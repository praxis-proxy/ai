// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Propagate the OpenSSL version cfgs from `openssl-sys` so that
//! `crypto_readiness` can gate on `#[cfg(ossl300)]`.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ossl300)");

    if let Ok(version) = std::env::var("DEP_OPENSSL_VERSION_NUMBER")
        && let Ok(version) = u64::from_str_radix(&version, 16)
        && version >= 0x3000_0000
    {
        println!("cargo:rustc-cfg=ossl300");
    }
}
