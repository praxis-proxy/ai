// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `PostgreSQL` container lifecycle for integration tests.
//!
//! Spawns a `PostgreSQL` container on a random host port and
//! removes it on drop. Requires `docker` or `podman` on `$PATH`
//! (or set `CONTAINER_ENGINE` to override detection).

use std::{
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair, SanType};
use tempfile::TempDir;

use super::port::free_port;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Container image to use for `PostgreSQL`.
const PG_IMAGE: &str = "docker.io/library/postgres:17-alpine";

/// Default database user.
const PG_USER: &str = "praxis";

/// Default database password.
const PG_PASSWORD: &str = "praxis";

/// Default database name.
const PG_DATABASE: &str = "praxis";

/// Maximum time to wait for the container to accept connections.
const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum time to wait for the TLS/cert-auth container, which does
/// extra work (writing certificate material and starting with SSL
/// enabled) before it accepts connections.
const CERT_AUTH_READY_TIMEOUT: Duration = Duration::from_secs(60);

/// Interval between readiness polls.
const READY_POLL_INTERVAL: Duration = Duration::from_millis(100);

// -----------------------------------------------------------------------------
// PostgresGuard
// -----------------------------------------------------------------------------

/// RAII guard that manages a `PostgreSQL` container lifecycle.
///
/// Spawns a `postgres:17-alpine` container on a random host
/// port, waits for TCP readiness, and kills the container on
/// drop. The container runs with `--rm` so it is also removed
/// after being killed.
///
/// # Panics
///
/// Panics if no container engine is found, if the container
/// fails to start, or if `PostgreSQL` does not become ready
/// within the timeout.
pub struct PostgresGuard {
    /// Container ID (short hash from `docker run`).
    container_id: String,

    /// Host port mapped to container port 5432.
    port: u16,

    /// Container engine command (`docker` or `podman`).
    engine: String,
}

impl PostgresGuard {
    /// Connection URL for this container.
    pub fn url(&self) -> String {
        format!(
            "postgres://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{}/{PG_DATABASE}",
            self.port
        )
    }

    /// Host port mapped to `PostgreSQL`'s 5432.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for PostgresGuard {
    fn drop(&mut self) {
        let _ = Command::new(&self.engine).args(["kill", &self.container_id]).output();
    }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// Start a `PostgreSQL` container and wait for it to accept
/// connections.
///
/// Uses `CONTAINER_ENGINE` env var if set, otherwise probes
/// for `podman` then `docker` on `$PATH`.
///
/// # Panics
///
/// Panics if no container engine is available, the container
/// fails to start, or `PostgreSQL` does not become ready within
/// 30 seconds.
pub fn start_postgres() -> PostgresGuard {
    let engine = detect_container_engine();
    let port = free_port();
    let container_id = run_container(&engine, port);
    let guard = PostgresGuard {
        container_id,
        port,
        engine,
    };

    wait_for_postgres(&guard.engine, &guard.container_id, READY_TIMEOUT);

    guard
}

// -----------------------------------------------------------------------------
// PostgresCertAuthGuard (TLS + certificate authentication)
// -----------------------------------------------------------------------------

/// RAII guard for a `PostgreSQL` container that requires TLS with
/// certificate authentication.
///
/// The server is started with `ssl=on` and a `pg_hba.conf` that permits
/// TCP connections only over TLS with a client certificate (`hostssl ...
/// cert`), so a client authenticates by presenting a certificate whose
/// Common Name matches the database role. No password is configured, so
/// the connection exercises the same certificate-authentication path the
/// `require_certificate_authentication` compliance profile depends on.
///
/// The CA, client certificate, and client key are written to a host
/// temporary directory (cleaned up on drop) for the proxy to load via
/// `ssl_root_cert`, `ssl_client_cert`, and `ssl_client_key`.
///
/// # Panics
///
/// Panics if no container engine is found, if certificate generation or
/// container startup fails, or if `PostgreSQL` does not become ready.
pub struct PostgresCertAuthGuard {
    /// Host directory holding the CA and client certificate material;
    /// removed on drop.
    cert_dir: TempDir,

    /// Container ID (short hash from `docker run`).
    container_id: String,

    /// Container engine command (`docker` or `podman`).
    engine: String,

    /// Host port mapped to container port 5432.
    port: u16,
}

impl PostgresCertAuthGuard {
    /// Host path to the PEM-encoded CA certificate (`ssl_root_cert`).
    pub fn ca_cert_path(&self) -> PathBuf {
        self.cert_dir.path().join("ca.crt")
    }

    /// Host path to the PEM-encoded client certificate (`ssl_client_cert`).
    pub fn client_cert_path(&self) -> PathBuf {
        self.cert_dir.path().join("client.crt")
    }

    /// Host path to the PEM-encoded client private key (`ssl_client_key`).
    pub fn client_key_path(&self) -> PathBuf {
        self.cert_dir.path().join("client.key")
    }

    /// Host port mapped to `PostgreSQL`'s 5432.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Password-less connection URL for this container.
    ///
    /// The URL carries no password and no TLS parameters, so it is
    /// suitable for the certificate-authentication compliance profile,
    /// which configures TLS exclusively through the filter fields.
    pub fn url_without_password(&self) -> String {
        format!("postgres://{PG_USER}@127.0.0.1:{}/{PG_DATABASE}", self.port)
    }
}

impl Drop for PostgresCertAuthGuard {
    fn drop(&mut self) {
        let _ = Command::new(&self.engine).args(["kill", &self.container_id]).output();
    }
}

/// Start a `PostgreSQL` container that requires TLS with certificate
/// authentication and wait for it to accept connections.
///
/// Generates a fresh CA that signs both the server certificate (with a
/// `127.0.0.1` IP SAN so `verify-full` succeeds on loopback) and a client
/// certificate whose Common Name is the database role. Certificate
/// material is passed to the container through environment variables and
/// written inside the container by a small shell wrapper, avoiding
/// bind-mount ownership and `SELinux` pitfalls.
///
/// # Panics
///
/// Panics if no container engine is available, certificate generation or
/// container startup fails, or `PostgreSQL` does not become ready within
/// the timeout.
pub fn start_postgres_cert_auth() -> PostgresCertAuthGuard {
    let engine = detect_container_engine();
    let port = free_port();
    let certs = CertMaterial::generate();
    let cert_dir = certs.write_client_files();

    let container_id = run_cert_auth_container(&engine, port, &certs);
    let guard = PostgresCertAuthGuard {
        cert_dir,
        container_id,
        engine,
        port,
    };

    wait_for_postgres(&guard.engine, &guard.container_id, CERT_AUTH_READY_TIMEOUT);

    guard
}

// -----------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------

/// Spawn a detached `PostgreSQL` container on the given port.
fn run_container(engine: &str, port: u16) -> String {
    let output = Command::new(engine)
        .args([
            "run",
            "-d",
            "--rm",
            "-e",
            &format!("POSTGRES_USER={PG_USER}"),
            "-e",
            &format!("POSTGRES_PASSWORD={PG_PASSWORD}"),
            "-e",
            &format!("POSTGRES_DB={PG_DATABASE}"),
            "-p",
            &format!("{port}:5432"),
            PG_IMAGE,
        ])
        .output()
        .expect("failed to execute container engine");

    assert!(
        output.status.success(),
        "container engine failed to start postgres: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("container ID should be valid UTF-8")
        .trim()
        .to_owned()
}

// -----------------------------------------------------------------------------
// Certificate-Authentication Container
// -----------------------------------------------------------------------------

/// PEM-encoded certificate material for the cert-auth container.
///
/// A single CA signs both the server certificate (presented to clients)
/// and the client certificate (presented for authentication), so the
/// same CA verifies both ends.
struct CertMaterial {
    /// PEM-encoded CA certificate (trusts server and client).
    ca_cert_pem: String,

    /// PEM-encoded client certificate (CN = database role).
    client_cert_pem: String,

    /// PEM-encoded client private key.
    client_key_pem: String,

    /// PEM-encoded server certificate (`127.0.0.1` IP SAN).
    server_cert_pem: String,

    /// PEM-encoded server private key.
    server_key_pem: String,
}

impl CertMaterial {
    /// Generate a CA plus server and client certificates.
    ///
    /// # Panics
    ///
    /// Panics if key or certificate generation fails.
    #[expect(
        clippy::too_many_lines,
        reason = "linear CA/server/client certificate assembly reads best as one sequence"
    )]
    fn generate() -> Self {
        let ca_key = KeyPair::generate().expect("CA key generation");
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("CA params");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "Praxis Postgres Test CA");
        let ca_cert = ca_params.self_signed(&ca_key).expect("CA self-sign");
        let issuer = Issuer::from_params(&ca_params, &ca_key);

        // Server certificate: loopback IP SAN so sqlx verify-full succeeds
        // when connecting to 127.0.0.1.
        let server_key = KeyPair::generate().expect("server key generation");
        let mut server_params = CertificateParams::new(vec!["localhost".to_owned()]).expect("server params");
        server_params.distinguished_name.push(DnType::CommonName, "localhost");
        server_params
            .subject_alt_names
            .push(SanType::IpAddress(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        server_params
            .extended_key_usages
            .push(ExtendedKeyUsagePurpose::ServerAuth);
        let server_cert = server_params.signed_by(&server_key, &issuer).expect("server cert sign");

        // Client certificate: CN must equal the database role for `cert`
        // authentication to map it to that role.
        let client_key = KeyPair::generate().expect("client key generation");
        let mut client_params = CertificateParams::new(Vec::<String>::new()).expect("client params");
        client_params.distinguished_name.push(DnType::CommonName, PG_USER);
        client_params
            .extended_key_usages
            .push(ExtendedKeyUsagePurpose::ClientAuth);
        let client_cert = client_params.signed_by(&client_key, &issuer).expect("client cert sign");

        Self {
            ca_cert_pem: ca_cert.pem(),
            client_cert_pem: client_cert.pem(),
            client_key_pem: client_key.serialize_pem(),
            server_cert_pem: server_cert.pem(),
            server_key_pem: server_key.serialize_pem(),
        }
    }

    /// Write the CA and client certificate/key to a fresh host temp dir.
    ///
    /// # Panics
    ///
    /// Panics if the temp directory or file writes fail.
    fn write_client_files(&self) -> TempDir {
        let dir = TempDir::new().expect("cert tempdir creation");
        write_file(&dir.path().join("ca.crt"), &self.ca_cert_pem);
        write_file(&dir.path().join("client.crt"), &self.client_cert_pem);
        let client_key = dir.path().join("client.key");
        write_file(&client_key, &self.client_key_pem);
        // The store filter fails closed on a group- or world-accessible client
        // key, so mirror the documented mode 0600 the harness must satisfy.
        restrict_key_permissions(&client_key);
        dir
    }
}

/// Write `contents` to `path`, panicking on failure.
fn write_file(path: &Path, contents: &str) {
    std::fs::write(path, contents).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Restrict a private key file to owner-only access (mode 0600) on Unix.
#[cfg(unix)]
fn restrict_key_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .unwrap_or_else(|e| panic!("chmod {}: {e}", path.display()));
}

/// No-op on non-Unix targets, which do not expose POSIX permission bits.
#[cfg(not(unix))]
fn restrict_key_permissions(_path: &Path) {}

/// `pg_hba.conf` that permits only TLS + client-certificate TCP access.
///
/// The `local ... trust` line lets the image entrypoint bootstrap the
/// database over the Unix socket. The `hostssl ... cert` lines are the
/// only TCP rules, so any non-TLS connection or any TLS connection
/// without a valid client certificate is refused.
const CERT_AUTH_PG_HBA: &str = "\
local   all   all                 trust
hostssl all   all   0.0.0.0/0     cert
hostssl all   all   ::/0          cert
";

/// Shell wrapper (run as root inside the container) that materializes the
/// certificate files with the ownership and permissions `PostgreSQL`
/// requires, then hands off to the stock image entrypoint with SSL on.
const CERT_AUTH_ENTRYPOINT: &str = "\
set -e
d=/etc/praxis-pg
mkdir -p \"$d\"
printf '%s' \"$PG_CA_CERT\" > \"$d/ca.crt\"
printf '%s' \"$PG_SERVER_CERT\" > \"$d/server.crt\"
printf '%s' \"$PG_SERVER_KEY\" > \"$d/server.key\"
printf '%s' \"$PG_HBA\" > \"$d/pg_hba.conf\"
chown postgres:postgres \"$d/ca.crt\" \"$d/server.crt\" \"$d/server.key\" \"$d/pg_hba.conf\"
chmod 600 \"$d/server.key\"
chmod 644 \"$d/ca.crt\" \"$d/server.crt\" \"$d/pg_hba.conf\"
exec docker-entrypoint.sh postgres \
  -c ssl=on \
  -c ssl_cert_file=\"$d/server.crt\" \
  -c ssl_key_file=\"$d/server.key\" \
  -c ssl_ca_file=\"$d/ca.crt\" \
  -c hba_file=\"$d/pg_hba.conf\"
";

/// Spawn a detached TLS + cert-auth `PostgreSQL` container.
#[expect(
    clippy::too_many_lines,
    reason = "single `docker run` argument vector for the cert-auth container"
)]
fn run_cert_auth_container(engine: &str, port: u16, certs: &CertMaterial) -> String {
    let output = Command::new(engine)
        .args([
            "run",
            "-d",
            "--rm",
            "-e",
            &format!("POSTGRES_USER={PG_USER}"),
            "-e",
            &format!("POSTGRES_DB={PG_DATABASE}"),
            // No password: the client authenticates with a certificate.
            "-e",
            "POSTGRES_HOST_AUTH_METHOD=trust",
            "-e",
            &format!("PG_CA_CERT={}", certs.ca_cert_pem),
            "-e",
            &format!("PG_SERVER_CERT={}", certs.server_cert_pem),
            "-e",
            &format!("PG_SERVER_KEY={}", certs.server_key_pem),
            "-e",
            &format!("PG_HBA={CERT_AUTH_PG_HBA}"),
            "-p",
            &format!("{port}:5432"),
            "--entrypoint",
            "/bin/sh",
            PG_IMAGE,
            "-c",
            CERT_AUTH_ENTRYPOINT,
        ])
        .output()
        .expect("failed to execute container engine");

    assert!(
        output.status.success(),
        "container engine failed to start cert-auth postgres: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("container ID should be valid UTF-8")
        .trim()
        .to_owned()
}

/// Detect the container engine to use.
fn detect_container_engine() -> String {
    if let Ok(engine) = std::env::var("CONTAINER_ENGINE") {
        return engine;
    }

    for candidate in ["podman", "docker"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return candidate.to_owned();
        }
    }

    panic!("no container engine found — install podman or docker, or set CONTAINER_ENGINE");
}

/// Poll until `PostgreSQL` is ready to accept queries.
///
/// First waits for `pg_isready`, then verifies the target
/// database is queryable via `psql`. The two-phase check
/// guards against a race in the Docker entrypoint where the
/// server accepts connections before the `POSTGRES_DB`
/// database has been created.
fn wait_for_postgres(engine: &str, container_id: &str, timeout: Duration) {
    let deadline = Instant::now() + timeout;

    poll_until(deadline, timeout, || {
        Command::new(engine)
            .args(["exec", container_id, "pg_isready", "-U", PG_USER])
            .output()
            .is_ok_and(|o| o.status.success())
    });

    poll_until(deadline, timeout, || {
        Command::new(engine)
            .args([
                "exec",
                container_id,
                "psql",
                "-U",
                PG_USER,
                "-d",
                PG_DATABASE,
                "-c",
                "SELECT 1",
            ])
            .output()
            .is_ok_and(|o| o.status.success())
    });
}

/// Repeatedly call `check` until it returns `true` or the deadline passes.
fn poll_until(deadline: Instant, timeout: Duration, check: impl Fn() -> bool) {
    loop {
        if check() {
            return;
        }
        if Instant::now() >= deadline {
            break;
        }
        thread::sleep(READY_POLL_INTERVAL);
    }
    panic!("`PostgreSQL` container did not become ready within {timeout:?}");
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn poll_until_checks_readiness_after_sleep_crosses_deadline() {
        let attempts = Cell::new(0);

        poll_until(
            Instant::now() + Duration::from_millis(1),
            Duration::from_millis(1),
            || {
                let next_attempt = attempts.get() + 1;
                attempts.set(next_attempt);
                next_attempt == 2
            },
        );

        assert_eq!(attempts.get(), 2);
    }
}
