// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SHA-256 through the system OpenSSL.
//!
//! Every digest praxis-ai computes goes through the platform's OpenSSL rather
//! than a pure-Rust implementation. On a FIPS host that library is the
//! validated module, and Red Hat's release scanner fails a binary on the name
//! of a pure-Rust cryptography crate alone, whether or not the use is a
//! security function. None of the digests here is one (identifiers, bucket
//! and cache keys), which is exactly why they must not be the reason a build
//! fails the scan.

use openssl::hash::{Hasher, MessageDigest};

/// Length of a SHA-256 digest in bytes.
pub const SHA256_LEN: usize = 32;

/// An incremental SHA-256 computation.
///
/// SHA-256 is part of every OpenSSL build, the FIPS provider included, so the
/// library's error paths cannot be reached by any input: they would mean the
/// library itself is broken, and the methods treat them as such.
pub struct Sha256 {
    /// The OpenSSL digest context.
    hasher: Hasher,
}

impl Sha256 {
    /// Start a new digest.
    ///
    /// # Panics
    ///
    /// If OpenSSL cannot provide SHA-256, which no build of it lacks.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "SHA-256 exists in every OpenSSL build, the FIPS provider included"
    )]
    pub fn new() -> Self {
        Self {
            hasher: Hasher::new(MessageDigest::sha256()).expect("OpenSSL provides SHA-256"),
        }
    }

    /// Feed bytes into the digest.
    ///
    /// # Panics
    ///
    /// If OpenSSL fails to update the digest context, which it does not for
    /// any input.
    #[expect(clippy::expect_used, reason = "an update cannot fail for any input")]
    pub fn update(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes).expect("OpenSSL updates a SHA-256 context");
    }

    /// Finish the digest.
    ///
    /// # Panics
    ///
    /// If OpenSSL fails to finish the digest, which it does not for any input.
    #[must_use]
    #[expect(clippy::expect_used, reason = "finishing cannot fail for any input")]
    pub fn finish(mut self) -> [u8; SHA256_LEN] {
        let digest = self.hasher.finish().expect("OpenSSL finishes a SHA-256 context");
        let mut out = [0; SHA256_LEN];
        out.copy_from_slice(&digest);
        out
    }

    /// The digest of one byte string.
    #[must_use]
    pub fn digest(bytes: &[u8]) -> [u8; SHA256_LEN] {
        let mut hasher = Self::new();
        hasher.update(bytes);
        hasher.finish()
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Lowercase hexadecimal encoding of a digest.
#[must_use]
pub fn hex(digest: &[u8]) -> String {
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        hex.push(char::from_digit(u32::from(byte & 0x0F), 16).unwrap_or('0'));
    }
    hex
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-4 test vectors, as `sha256sum` prints them.
    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn one_shot_digests_match_the_known_vectors() {
        assert_eq!(hex(&Sha256::digest(b"")), EMPTY, "empty input");
        assert_eq!(hex(&Sha256::digest(b"abc")), ABC, "abc");
    }

    #[test]
    fn incremental_updates_equal_the_one_shot_digest() {
        let mut hasher = Sha256::new();
        hasher.update(b"a");
        hasher.update(b"");
        hasher.update(b"bc");
        assert_eq!(hasher.finish(), Sha256::digest(b"abc"), "segmentation does not matter");
        assert_eq!(
            Sha256::default().finish(),
            Sha256::digest(b""),
            "default is a fresh context"
        );
    }

    #[test]
    fn hex_is_lowercase_and_two_characters_per_byte() {
        assert_eq!(hex(&[0x00, 0x0F, 0xA5, 0xFF]), "000fa5ff", "lowercase, zero-padded");
        assert_eq!(hex(&[]), "", "empty");
    }
}
