// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Signed gateway `Mcp-Session-Id` values and ended-session tombstones.
//!
//! With `backend_sessions`, the broker only accepts gateway session ids it
//! issued through `initialize` and that no `DELETE` has ended (#2126). An id
//! is `mcp-<payload>.<tag>`: a 16-byte payload (an 8-byte issue sequence
//! number followed by 8 random bytes) and its HMAC-SHA256 tag, both base64url
//! without padding. Verifying the tag needs no per-session state, so
//! `initialize` cannot be used to grow memory.
//!
//! The signing key is drawn from OpenSSL's private generator when the filter is
//! built and never leaves the process. A config reload or restart therefore
//! invalidates every outstanding id: clients get `404` and send a new
//! `initialize`, as the MCP Streamable HTTP transport requires. Several
//! broker replicas behind one address are not supported until session state
//! is shared (#156).
//!
//! `DELETE` records the id's sequence number in a bounded list. When the list
//! is full, the lowest entry is dropped and every id issued up to it is
//! rejected from then on, live or ended. Forgetting state can therefore only
//! end ids, never revive one; a client whose live id expires this way gets
//! `404` and re-initializes.

use std::{
    collections::BTreeSet,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use praxis_ai_apis::hash::{HmacSha256, SHA256_LEN, private_random_bytes, random_bytes};
use zeroize::Zeroizing;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Prefix shared with ids issued without `backend_sessions`.
const PREFIX: &str = "mcp-";

/// Separator between the encoded payload and tag.
const SEPARATOR: char = '.';

/// Issue sequence number length, big-endian at the start of the payload.
const SEQUENCE_LEN: usize = 8;

/// Signed payload: sequence number plus random bytes.
const PAYLOAD_LEN: usize = 16;

/// HMAC key length (the SHA-256 block-size guidance: at least the output).
const KEY_LEN: usize = 32;

/// Most ended ids remembered per filter instance.
const MAX_ENDED: usize = 16_384;

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// Signed part of a gateway session id.
type Payload = [u8; PAYLOAD_LEN];

/// Ended-session state.
#[derive(Default)]
struct Ended {
    /// Sequence numbers of ended sessions at or above `floor`.
    ended: BTreeSet<u64>,
    /// Ids with a lower sequence number are rejected.
    floor: u64,
}

/// Issues and verifies gateway session ids for one broker filter instance.
pub(crate) struct GatewaySessions {
    /// Ended sessions.
    ended: Mutex<Ended>,
    /// HMAC key, zeroed on drop.
    key: Zeroizing<[u8; KEY_LEN]>,
    /// Ended-list capacity.
    max_ended: usize,
    /// Next issue sequence number.
    next: AtomicU64,
}

/// Why a gateway session id could not be issued.
#[derive(Debug)]
pub(crate) struct IssueError;

impl std::fmt::Display for IssueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("could not issue a gateway session id")
    }
}

impl std::error::Error for IssueError {}

// -----------------------------------------------------------------------------
// GatewaySessions
// -----------------------------------------------------------------------------

impl GatewaySessions {
    /// Draw a fresh signing key.
    ///
    /// # Errors
    ///
    /// Returns [`IssueError`] if OpenSSL's generator cannot supply the key.
    pub(crate) fn new() -> Result<Self, IssueError> {
        Self::with_capacity(MAX_ENDED)
    }

    /// [`Self::new`] with an explicit ended-list capacity.
    fn with_capacity(max_ended: usize) -> Result<Self, IssueError> {
        let mut key = Zeroizing::new([0; KEY_LEN]);
        private_random_bytes(key.as_mut()).map_err(|_error| IssueError)?;
        Ok(Self {
            ended: Mutex::new(Ended::default()),
            key,
            max_ended,
            next: AtomicU64::new(0),
        })
    }

    /// Issue a new gateway session id.
    ///
    /// # Errors
    ///
    /// Returns [`IssueError`] if the payload or its tag cannot be produced;
    /// the caller must not hand out an unsigned id.
    pub(crate) fn issue(&self) -> Result<String, IssueError> {
        let mut payload: Payload = [0; PAYLOAD_LEN];
        let (sequence, random) = payload.split_at_mut(SEQUENCE_LEN);
        sequence.copy_from_slice(&self.next.fetch_add(1, Ordering::Relaxed).to_be_bytes());
        random_bytes(random).map_err(|_error| IssueError)?;
        let tag = HmacSha256::mac(self.key.as_ref(), &payload).map_err(|_error| IssueError)?;
        let mut id = String::with_capacity(PREFIX.len() + 1 + 22 + 43);
        id.push_str(PREFIX);
        URL_SAFE_NO_PAD.encode_string(payload, &mut id);
        id.push(SEPARATOR);
        URL_SAFE_NO_PAD.encode_string(tag, &mut id);
        Ok(id)
    }

    /// Whether `id` was issued by this instance and has not been ended.
    pub(crate) fn is_live(&self, id: &str) -> bool {
        self.verified_sequence(id)
            .is_some_and(|sequence| self.lock().is_live(sequence))
    }

    /// End a live session, returning `false` when `id` was never issued or
    /// was already ended.
    pub(crate) fn end(&self, id: &str) -> bool {
        let Some(sequence) = self.verified_sequence(id) else {
            return false;
        };
        let mut state = self.lock();
        if !state.is_live(sequence) {
            return false;
        }
        state.ended.insert(sequence);
        if state.ended.len() > self.max_ended
            && let Some(lowest) = state.ended.pop_first()
        {
            state.floor = lowest.saturating_add(1);
        }
        true
    }

    /// The sequence number of an id whose tag verifies under this instance's
    /// key. Malformed ids and ids signed by another key yield `None`, as does
    /// a MAC failure (fail closed).
    fn verified_sequence(&self, id: &str) -> Option<u64> {
        let (payload, tag) = id.strip_prefix(PREFIX)?.split_once(SEPARATOR)?;
        let payload: Payload = decode_exact(payload)?;
        let tag: [u8; SHA256_LEN] = decode_exact(tag)?;
        if !HmacSha256::verify(self.key.as_ref(), &payload, &tag).ok()? {
            return None;
        }
        let (sequence, _) = payload.split_first_chunk::<SEQUENCE_LEN>()?;
        Some(u64::from_be_bytes(*sequence))
    }

    /// Lock the ended state, recovering from a poisoned lock: every update
    /// leaves it consistent.
    fn lock(&self) -> MutexGuard<'_, Ended> {
        self.ended.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Decode unpadded base64url into exactly `N` bytes on the stack.
fn decode_exact<const N: usize>(encoded: &str) -> Option<[u8; N]> {
    // Unpadded base64 carries 6 bits per character.
    if encoded.len() != (N * 8).div_ceil(6) {
        return None;
    }
    // The decoder reserves room for whole 3-byte groups.
    let mut buf = [0; 48];
    let decoded = buf.get_mut(..N.div_ceil(3) * 3)?;
    let len = URL_SAFE_NO_PAD.decode_slice(encoded, decoded).ok()?;
    decoded.get(..len)?.try_into().ok()
}

impl Ended {
    /// Whether an id with `sequence` is neither below the floor nor ended.
    fn is_live(&self, sequence: u64) -> bool {
        sequence >= self.floor && !self.ended.contains(&sequence)
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn issued_ids_are_live_and_well_formed() {
        let sessions = GatewaySessions::new().unwrap();
        let id = sessions.issue().unwrap();

        assert!(sessions.is_live(&id), "an issued id is live");
        assert!(id.starts_with(PREFIX), "ids keep the mcp- prefix");
        assert!(
            id.bytes().all(|b| (0x21..=0x7E).contains(&b)),
            "ids are visible ASCII, as the transport requires"
        );
    }

    #[test]
    fn malformed_and_foreign_ids_are_not_live() {
        let sessions = GatewaySessions::new().unwrap();
        let foreign = GatewaySessions::new().unwrap().issue().unwrap();
        let id = sessions.issue().unwrap();
        let (payload, _) = id.split_once(SEPARATOR).unwrap();

        for candidate in [
            "",
            "mcp-",
            "mcp-abc",
            "mcp-abc.def",
            "other-prefix",
            payload,
            &format!("{payload}."),
            &format!("{id}A"),
            &id.replacen(PREFIX, "", 1),
            &foreign,
        ] {
            assert!(!sessions.is_live(candidate), "{candidate:?} must not be live");
        }
    }

    #[test]
    fn ended_ids_are_not_live_and_end_only_once() {
        let sessions = GatewaySessions::new().unwrap();
        let id = sessions.issue().unwrap();
        let other = sessions.issue().unwrap();

        assert!(sessions.end(&id), "the first end succeeds");
        assert!(!sessions.is_live(&id), "an ended id is not live");
        assert!(!sessions.end(&id), "a second end is refused");
        assert!(sessions.is_live(&other), "other ids stay live");
        assert!(!sessions.end("mcp-forged.x"), "a forged id cannot be ended");
    }

    #[test]
    fn full_ended_list_never_revives_an_ended_id() {
        let sessions = GatewaySessions::with_capacity(2).unwrap();
        let ids: [String; 3] = std::array::from_fn(|_| sessions.issue().unwrap());

        for id in &ids {
            assert!(sessions.end(id), "each id ends once");
        }

        let [oldest, middle, newest] = &ids;
        assert!(!sessions.is_live(oldest), "an evicted entry stays ended");
        assert!(!sessions.end(oldest), "an evicted entry cannot be ended again");
        assert!(!sessions.is_live(middle), "newer entries are kept");
        assert!(!sessions.is_live(newest), "the newest entry is kept");
        assert_eq!(sessions.lock().ended.len(), 2, "the list never exceeds capacity");
    }

    #[test]
    fn full_ended_list_expires_live_ids_issued_before_the_evicted_one() {
        let sessions = GatewaySessions::with_capacity(1).unwrap();
        let older_live = sessions.issue().unwrap();
        let first = sessions.issue().unwrap();
        let second = sessions.issue().unwrap();
        let newer_live = sessions.issue().unwrap();

        assert!(sessions.end(&first), "first end succeeds");
        assert!(sessions.end(&second), "second end evicts the first");

        assert!(!sessions.is_live(&older_live), "ids older than an evicted entry expire");
        assert!(!sessions.is_live(&first), "the evicted entry stays ended");
        assert!(sessions.is_live(&newer_live), "ids issued later stay live");
    }
}
