// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SHA-256 and HMAC-SHA256 through the system OpenSSL.
//!
//! Every digest and MAC praxis-ai computes goes through the platform's
//! OpenSSL rather than a pure-Rust implementation. On a FIPS host that
//! library is the validated module, and Red Hat's release scanner fails a
//! binary on the name of a pure-Rust cryptography crate alone, whether or not
//! the use is a security function. None of the plain digests here is one
//! (identifiers, bucket and cache keys), which is exactly why they must not be
//! the reason a build fails the scan. [`HmacSha256`] is the exception: it
//! signs AWS requests (`aws_sigv4_sign`), so it must run inside the validated
//! module on a FIPS host, and it reports the library's refusal instead of
//! panicking so a signing failure fails closed at the request.

use openssl::{
    error::ErrorStack,
    hash::{Hasher, MessageDigest},
    pkey::PKey,
    sign::Signer,
};

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

/// HMAC-SHA256 (RFC 2104 with SHA-256) through OpenSSL's EVP signing API.
///
/// The key is wrapped in an `EVP_PKEY` and the MAC is computed with
/// `EVP_DigestSign*`, which dispatches through the loaded provider (the
/// FIPS provider on a FIPS host). The legacy `HMAC()` and `openssl::sha`
/// entry points bypass the provider and are not used.
///
/// Unlike [`Sha256`], the operations are fallible: the provider decides which
/// keys it accepts (a FIPS policy may refuse a key it considers too short),
/// and the caller must be able to refuse to sign rather than crash.
pub struct HmacSha256;

impl HmacSha256 {
    /// The MAC of `data` under `key`.
    ///
    /// # Errors
    ///
    /// Returns the OpenSSL error stack if the provider refuses the key or
    /// cannot supply HMAC-SHA256. The error carries no key material.
    pub fn mac(key: &[u8], data: &[u8]) -> Result<[u8; SHA256_LEN], ErrorStack> {
        let key = PKey::hmac(key)?;
        let mut signer = Signer::new(MessageDigest::sha256(), &key)?;
        let mut out = [0; SHA256_LEN];
        let written = signer.sign_oneshot(&mut out, data)?;
        if written != SHA256_LEN {
            // Cannot happen for SHA-256; treat it as the library misbehaving
            // rather than returning a truncated tag.
            return Err(ErrorStack::get());
        }
        Ok(out)
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
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, reason = "tests")]
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

    /// RFC 4231 test case 2 (key "Jefe", data "what do ya want for nothing?").
    const RFC4231_CASE2: &str = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
    /// RFC 4231 test case 1 (20-byte key of 0x0b, data "Hi There").
    const RFC4231_CASE1: &str = "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7";
    /// RFC 4231 test case 6 (131-byte key of 0xaa: longer than the block size).
    const RFC4231_CASE6: &str = "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54";

    #[test]
    fn hmac_matches_the_rfc_4231_vectors() {
        assert_eq!(
            hex(&HmacSha256::mac(b"Jefe", b"what do ya want for nothing?").expect("mac")),
            RFC4231_CASE2,
            "case 2"
        );
        assert_eq!(
            hex(&HmacSha256::mac(&[0x0B; 20], b"Hi There").expect("mac")),
            RFC4231_CASE1,
            "case 1"
        );
        assert_eq!(
            hex(
                &HmacSha256::mac(&[0xAA; 131], b"Test Using Larger Than Block-Size Key - Hash Key First").expect("mac")
            ),
            RFC4231_CASE6,
            "case 6: key longer than the block size is hashed first"
        );
    }

    #[test]
    fn hmac_output_is_not_a_plain_digest_and_depends_on_the_key() {
        let keyed = HmacSha256::mac(b"k1", b"abc").expect("mac");
        assert_ne!(hex(&keyed), ABC, "a MAC is not the unkeyed digest");
        assert_ne!(
            keyed,
            HmacSha256::mac(b"k2", b"abc").expect("mac"),
            "a different key gives a different tag"
        );
    }

    #[test]
    fn hex_is_lowercase_and_two_characters_per_byte() {
        assert_eq!(hex(&[0x00, 0x0F, 0xA5, 0xFF]), "000fa5ff", "lowercase, zero-padded");
        assert_eq!(hex(&[]), "", "empty");
    }
}
