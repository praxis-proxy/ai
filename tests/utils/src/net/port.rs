// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Port allocation utilities for integration tests.

use std::{
    collections::HashSet,
    net::TcpListener,
    sync::{
        LazyLock, Mutex, PoisonError,
        atomic::{AtomicU16, Ordering},
    },
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Number of ports in the reserved band below the OS ephemeral range.
///
/// Ports are never returned to the band, so this bounds how many a single
/// test binary can allocate over its whole run. The integration suite uses
/// roughly a third of it.
const BAND_WIDTH: u16 = 8192;

/// Ephemeral floor assumed when the OS range cannot be read.
///
/// The Linux default is `32768`; macOS starts higher at `49152`. Assuming the
/// lower of the two keeps the reserved band below both.
const DEFAULT_EPHEMERAL_FLOOR: u16 = 32768;

/// Ceiling imposed to keep allocated ports from aliasing an example-config port.
///
/// Tests patch example addresses by plain substring replacement, so a port
/// whose decimal form *begins* with an example port is rewritten along with
/// the address it was meant to leave alone: with a listener on `30011`,
/// replacing `127.0.0.1:3001` turns `127.0.0.1:30011` into
/// `127.0.0.1:<new>1`. Example configs use `3000`-`3004`, whose five-digit
/// extensions are `30000`-`30049`, and those are the only example ports whose
/// extensions land inside the `u16` range (`8000` and `9000` extend past it).
/// Staying below `30000` keeps every allocated port unaliasable.
///
/// [`crate::patch_yaml`] is boundary-aware and does not need this, but the
/// ad-hoc `.replace()` chains in individual tests are not.
const ALIAS_CEILING: u16 = 30000;

// -----------------------------------------------------------------------------
// Statics
// -----------------------------------------------------------------------------

/// Process-wide set of ports this process has claimed.
///
/// A port is inserted before its bind is attempted and is never removed, so a
/// candidate that proved unusable is not retried.
static ALLOCATED_PORTS: LazyLock<Mutex<HashSet<u16>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// First port of the reserved band.
///
/// Ports are handed out from `[BAND_START, BAND_START + BAND_WIDTH)`, which
/// sits entirely below the OS ephemeral range (and below [`ALIAS_CEILING`]).
/// A listener bound to port `0` therefore cannot be assigned a port that this
/// allocator also hands out: the two pools are disjoint by construction.
///
/// This matters because test servers bind `127.0.0.1:0` directly, without
/// consulting [`ALLOCATED_PORTS`]. Drawing proxy ports from the ephemeral
/// range let the OS hand one of those servers a port that a test had already
/// reserved for its proxy, after [`free_port`] closed the probe listener but
/// before the proxy bound. The proxy's bind then failed on its own thread
/// while the readiness probe was satisfied by the squatting server, so the
/// test ran its whole scenario against an unrelated backend.
static BAND_START: LazyLock<u16> = LazyLock::new(|| band_end() - BAND_WIDTH);

/// Rotating offset into the reserved band.
///
/// Seeded per process so that concurrently running test binaries start at
/// different points in the band instead of contending over its first ports.
static CURSOR: LazyLock<AtomicU16> = LazyLock::new(|| AtomicU16::new(seed_offset()));

// -----------------------------------------------------------------------------
// Band Selection
// -----------------------------------------------------------------------------

/// One past the last port of the reserved band.
///
/// Below the OS ephemeral range so that a bind to port `0` cannot collide,
/// and below [`ALIAS_CEILING`] so that substring address patching cannot
/// corrupt an allocated port.
fn band_end() -> u16 {
    ephemeral_floor().min(ALIAS_CEILING)
}

/// Lowest port the OS assigns when asked to bind port `0`.
///
/// Reads the Linux range from `/proc`, falling back to
/// [`DEFAULT_EPHEMERAL_FLOOR`] on any platform or parse failure.
fn ephemeral_floor() -> u16 {
    let Ok(range) = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range") else {
        return DEFAULT_EPHEMERAL_FLOOR;
    };
    range
        .split_whitespace()
        .next()
        .and_then(|low| low.parse::<u16>().ok())
        // A floor at or below the band width leaves no room beneath it.
        .filter(|low| *low > BAND_WIDTH)
        .unwrap_or(DEFAULT_EPHEMERAL_FLOOR)
}

/// A per-process starting offset into the reserved band.
fn seed_offset() -> u16 {
    let pid = u16::try_from(std::process::id() % u32::from(BAND_WIDTH)).unwrap_or(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            u16::try_from(since.subsec_nanos() % u32::from(BAND_WIDTH)).unwrap_or(0)
        });
    (pid ^ nanos) % BAND_WIDTH
}

// -----------------------------------------------------------------------------
// Port Allocation
// -----------------------------------------------------------------------------

/// Bind a port from the reserved band that no other caller in this
/// process has claimed, registering it before the bind is attempted.
///
/// # Panics
///
/// Panics if no port in the band can be bound.
pub fn bind_unique_port() -> (TcpListener, u16) {
    for _ in 0..BAND_WIDTH {
        let offset = CURSOR.fetch_add(1, Ordering::Relaxed) % BAND_WIDTH;
        let port = *BAND_START + offset;

        // Claim the port before binding so a bind failure (a service outside
        // this process holds it) retires the candidate instead of spinning.
        if !ALLOCATED_PORTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(port)
        {
            continue;
        }
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            return (listener, port);
        }
    }
    panic!(
        "failed to bind a port in the reserved band {}..{}",
        *BAND_START,
        *BAND_START + BAND_WIDTH
    );
}

/// A held port that keeps its [`TcpListener`] open until dropped or released.
///
/// Call [`release`] to drop the listener and obtain the
/// port number just before starting the server under test.
///
/// [`release`]: PortGuard::release
pub struct PortGuard {
    /// The allocated port number.
    port: u16,

    /// Held listener that prevents port reuse until dropped.
    _listener: TcpListener,
}

impl PortGuard {
    /// The allocated port number.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Consume the guard, releasing the held listener so the
    /// port can be rebound by the server under test.
    pub fn release(self) -> u16 {
        self.port
    }
}

impl std::fmt::Display for PortGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.port)
    }
}

/// Allocate a free port from the reserved band and release it.
///
/// The returned port is unbound, so the caller races anything that binds an
/// explicit port. It does not race a listener bound to port `0`, because the
/// band sits below the OS ephemeral range.
pub fn free_port() -> u16 {
    let (_listener, port) = bind_unique_port();
    port
}

/// Like [`free_port`] but returns a [`PortGuard`].
pub fn free_port_guard() -> PortGuard {
    let (listener, port) = bind_unique_port();
    PortGuard {
        port,
        _listener: listener,
    }
}

// -----------------------------------------------------------------------------
// IPv6
// -----------------------------------------------------------------------------

/// Attempt to bind to `[::1]:0`. Returns `true` if IPv6
/// loopback is available in this environment.
pub fn ipv6_available() -> bool {
    TcpListener::bind("[::1]:0").is_ok()
}

/// Allocate a free port on the IPv6 loopback interface.
///
/// # Panics
///
/// Panics if binding to `[::1]:0` fails (caller must check
/// [`ipv6_available`] first).
pub fn free_port_v6() -> u16 {
    TcpListener::bind("[::1]:0").unwrap().local_addr().unwrap().port()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn allocated_ports_sit_below_the_ephemeral_range() {
        let floor = ephemeral_floor();
        for _ in 0..64 {
            let port = free_port();
            assert!(
                port < floor,
                "port {port} must sit below the ephemeral floor {floor} so that a \
                 bind to port 0 can never be assigned it"
            );
            assert!(port >= *BAND_START, "port {port} must sit inside the reserved band");
        }
    }

    /// No port in the band may begin with a port that example configs use, or
    /// the substring address patching in individual tests would rewrite it.
    #[test]
    fn no_band_port_aliases_an_example_config_port() {
        let example_ports = example_config_ports();
        assert!(
            example_ports.contains(&3001),
            "sanity: the example configs should still use port 3001"
        );

        for example in example_ports {
            for suffix in 0..10_u32 {
                let Ok(alias) = u16::try_from(u32::from(example) * 10 + suffix) else {
                    // Extends past the port range, so it can never be allocated.
                    continue;
                };
                assert!(
                    alias < *BAND_START || alias >= band_end(),
                    "band {}..{} contains {alias}, which begins with the example-config \
                     port {example}; a test patching \"127.0.0.1:{example}\" by substring \
                     would corrupt an address bound to {alias}",
                    *BAND_START,
                    band_end()
                );
            }
        }
    }

    /// Every port that an example config binds or dials.
    fn example_config_ports() -> HashSet<u16> {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/configs");
        let mut ports = HashSet::new();
        let mut pending = vec![std::path::PathBuf::from(root)];

        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).expect("read examples/configs").flatten() {
                let path = entry.path();
                if path.is_dir() {
                    pending.push(path);
                    continue;
                }
                if path.extension().is_none_or(|ext| ext != "yaml") {
                    continue;
                }
                let yaml = std::fs::read_to_string(&path).expect("read example config");
                ports.extend(address_ports(&yaml));
            }
        }
        ports
    }

    /// Ports that `yaml` pairs with an IPv4 literal.
    ///
    /// Only IPv4 addresses count: those are what tests rewrite by substring,
    /// and a looser scan picks up unrelated `:digits` text such as the source
    /// line references in the visualizer configs.
    fn address_ports(yaml: &str) -> Vec<u16> {
        // Anything outside an address breaks the token, so `127.0.0.1:3001`
        // survives surrounding quotes, keys, and list markers intact.
        yaml.split(|c: char| !matches!(c, '0'..='9' | '.' | ':'))
            .filter_map(|token| {
                let (host, port) = token.rsplit_once(':')?;
                is_ipv4_literal(host).then_some(port)?.parse::<u16>().ok()
            })
            .collect()
    }

    /// Whether `host` is four dot-separated decimal octets.
    fn is_ipv4_literal(host: &str) -> bool {
        let mut parts = host.split('.');
        let octets = [parts.next(), parts.next(), parts.next(), parts.next()];
        parts.next().is_none() && octets.iter().all(|part| part.is_some_and(|p| p.parse::<u8>().is_ok()))
    }

    #[test]
    fn a_released_port_is_not_reassigned_to_an_ephemeral_bind() {
        // The regression: `free_port` closes its probe listener, so the port is
        // unbound when it is returned. A server that binds port 0 must still be
        // unable to receive it.
        let released: HashSet<u16> = (0..32).map(|_| free_port()).collect();

        let ephemeral: Vec<TcpListener> = (0..64).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();

        for listener in &ephemeral {
            let port = listener.local_addr().unwrap().port();
            assert!(
                !released.contains(&port),
                "an ephemeral bind was assigned port {port}, which was already handed \
                 out by free_port"
            );
        }
    }

    #[test]
    fn bind_unique_port_returns_distinct_ports() {
        let (listener_a, port_a) = bind_unique_port();
        let (listener_b, port_b) = bind_unique_port();

        assert_ne!(port_a, 0, "first port should be non-zero");
        assert_ne!(port_b, 0, "second port should be non-zero");
        assert_ne!(port_a, port_b, "two calls should return distinct ports");

        assert_ne!(
            listener_a.local_addr().unwrap().port(),
            0,
            "first listener should be bound to a valid port"
        );
        assert_ne!(
            listener_b.local_addr().unwrap().port(),
            0,
            "second listener should be bound to a valid port"
        );
    }
}
