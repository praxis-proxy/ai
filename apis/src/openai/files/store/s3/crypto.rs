// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! OpenSSL provider for `object_store` signing, adapted from [ModelExpress].
//!
//! [ModelExpress]: https://github.com/opendatahub-io/modelexpress/blob/main/modelexpress_common/src/providers/s3_crypto.rs

use std::{error::Error, fmt::Debug};

use object_store::client::{CryptoProvider, DigestAlgorithm, DigestContext, HmacContext, Signer, SigningAlgorithm};
use openssl::{
    error::ErrorStack,
    hash::{DigestBytes, Hasher, MessageDigest},
    md::Md,
    md_ctx::MdCtx,
    pkey::{Id, PKey, Private},
    rsa::Padding,
    sign::Signer as OpensslSigner,
};

use crate::hash::{HmacError, SHA256_LEN};

/// Dispatch signing and hashing through the system OpenSSL.
#[derive(Debug)]
pub(super) struct OpensslCryptoProvider;

impl CryptoProvider for OpensslCryptoProvider {
    fn digest(&self, algorithm: DigestAlgorithm) -> object_store::Result<Box<dyn DigestContext>> {
        require_sha256(&algorithm)?;
        let hasher = Hasher::new(MessageDigest::sha256()).map_err(crypto_error)?;
        Ok(Box::new(OpensslDigestContext {
            hasher,
            out: None,
            update_error: None,
        }))
    }

    fn hmac(&self, algorithm: DigestAlgorithm, secret: &[u8]) -> object_store::Result<Box<dyn HmacContext>> {
        require_sha256(&algorithm)?;
        let key = PKey::hmac(secret).map_err(crypto_error)?;
        // EVP retains the key in the context so each update can go directly to OpenSSL.
        let mut ctx = MdCtx::new().map_err(crypto_error)?;
        ctx.digest_sign_init(Some(Md::sha256()), &key).map_err(crypto_error)?;
        Ok(Box::new(OpensslHmacContext {
            ctx,
            out: [0; SHA256_LEN],
            update_error: None,
        }))
    }

    fn sign(&self, algorithm: SigningAlgorithm, pem: &[u8]) -> object_store::Result<Box<dyn Signer>> {
        if algorithm != SigningAlgorithm::RS256 {
            return Err(unsupported(&algorithm));
        }
        let key = PKey::private_key_from_pem(pem).map_err(crypto_error)?;
        if key.id() != Id::RSA {
            return Err(crypto_error("RS256 requires an RSA private key"));
        }
        Ok(Box::new(OpensslRsaKey(key)))
    }
}

/// Reject any future digest algorithms until the provider supports them.
fn require_sha256(algorithm: &DigestAlgorithm) -> object_store::Result<()> {
    if *algorithm == DigestAlgorithm::Sha256 {
        Ok(())
    } else {
        Err(unsupported(algorithm))
    }
}

/// Report an unsupported digest or signature algorithm.
fn unsupported(algorithm: &dyn Debug) -> object_store::Error {
    object_store::Error::NotSupported {
        source: format!("unsupported OpenSSL signing or digest algorithm {algorithm:?}").into(),
    }
}

/// Preserve the cause of an OpenSSL provider failure in the store error.
fn crypto_error(source: impl Into<Box<dyn Error + Send + Sync>>) -> object_store::Error {
    object_store::Error::Generic {
        store: "OpensslCryptoProvider",
        source: source.into(),
    }
}

/// Incremental SHA-256 with a borrowed result at finalization.
struct OpensslDigestContext {
    /// Active digest state.
    hasher: Hasher,
    /// OpenSSL's digest bytes, retained for [`DigestContext::finish`].
    out: Option<DigestBytes>,
    /// First update failure, returned by [`DigestContext::finish`].
    update_error: Option<ErrorStack>,
}

impl DigestContext for OpensslDigestContext {
    fn update(&mut self, data: &[u8]) {
        if self.update_error.is_none() {
            self.update_error = self.hasher.update(data).err();
        }
    }

    fn finish(&mut self) -> object_store::Result<&[u8]> {
        if let Some(source) = self.update_error.take() {
            return Err(crypto_error(source));
        }
        let digest = self.hasher.finish().map_err(crypto_error)?;
        Ok(self.out.insert(digest))
    }
}

/// Incremental HMAC-SHA256 through EVP digest signing.
struct OpensslHmacContext {
    /// Signing state holding the HMAC key.
    ctx: MdCtx,
    /// Space for the fixed-length tag returned by [`HmacContext::finish`].
    out: [u8; SHA256_LEN],
    /// First update failure, returned by [`HmacContext::finish`].
    update_error: Option<ErrorStack>,
}

impl HmacContext for OpensslHmacContext {
    fn update(&mut self, data: &[u8]) {
        if self.update_error.is_none() {
            self.update_error = self.ctx.digest_sign_update(data).err();
        }
    }

    fn finish(&mut self) -> object_store::Result<&[u8]> {
        if let Some(source) = self.update_error.take() {
            return Err(crypto_error(source));
        }
        let written = self.ctx.digest_sign_final(Some(&mut self.out)).map_err(crypto_error)?;
        if written != SHA256_LEN {
            return Err(crypto_error(HmacError::TagLength(written)));
        }
        Ok(&self.out)
    }
}

/// Parsed RSA private key for PKCS#1 v1.5 signatures.
struct OpensslRsaKey(PKey<Private>);

impl Signer for OpensslRsaKey {
    fn sign(&self, string_to_sign: &[u8]) -> object_store::Result<Vec<u8>> {
        let mut signer = OpensslSigner::new(MessageDigest::sha256(), &self.0).map_err(crypto_error)?;
        signer.set_rsa_padding(Padding::PKCS1).map_err(crypto_error)?;
        signer.sign_oneshot_to_vec(string_to_sign).map_err(crypto_error)
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use openssl::{
        ec::{EcGroup, EcKey},
        nid::Nid,
        rsa::Rsa,
        sign::Verifier,
    };

    use super::*;
    use crate::hash::hex;

    #[test]
    fn sha256_matches_nist_vector() {
        let mut ctx = OpensslCryptoProvider.digest(DigestAlgorithm::Sha256).unwrap();
        ctx.update(b"a");
        ctx.update(b"bc");
        assert_eq!(
            hex(ctx.finish().unwrap()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hmac_sha256_matches_rfc4231_case_1() {
        let mut ctx = OpensslCryptoProvider
            .hmac(DigestAlgorithm::Sha256, &[0x0B; 20])
            .unwrap();
        ctx.update(b"Hi ");
        ctx.update(b"There");
        assert_eq!(
            hex(ctx.finish().unwrap()),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn rs256_signature_verifies() {
        let key = PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap();
        let pem = key.private_key_to_pem_pkcs8().unwrap();
        let signer = OpensslCryptoProvider.sign(SigningAlgorithm::RS256, &pem).unwrap();
        let data = b"string to sign";
        let signature = signer.sign(data).unwrap();

        let mut verifier = Verifier::new(MessageDigest::sha256(), &key).unwrap();
        verifier.set_rsa_padding(Padding::PKCS1).unwrap();
        assert!(verifier.verify_oneshot(&signature, data).unwrap());
    }

    #[test]
    fn rs256_rejects_non_rsa_keys() {
        let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).unwrap();
        let key = PKey::from_ec_key(EcKey::generate(&group).unwrap()).unwrap();
        let pem = key.private_key_to_pem_pkcs8().unwrap();
        assert!(OpensslCryptoProvider.sign(SigningAlgorithm::RS256, &pem).is_err());
    }
}
