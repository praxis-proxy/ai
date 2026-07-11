// SPDX-License-Identifier: MIT AND Apache-2.0 AND BSD-3-Clause
// Copyright (c) 2024 Praxis Contributors
// Vendored Envoy protos: Apache-2.0 (see NOTICE)
// Vendored Google protos: BSD-3-Clause (see NOTICE)

//! Protobuf and gRPC definitions for the Envoy external processing
//! protocol.
//!
//! Compiled from vendored `.proto` files at build time.

#[allow(
    clippy::allow_attributes,
    clippy::missing_docs_in_private_items,
    clippy::missing_errors_doc,
    reason = "generated protobuf module tree"
)]
/// Envoy protobuf type tree for `ext_proc`.
///
/// **`llm-d` only:** Exposed for `llm-d` integration tests. No API
/// stability is guaranteed.
pub mod envoy {
    /// Service definitions.
    pub mod service {
        /// Common types shared across Envoy services.
        pub mod common {
            #[allow(
                dead_code,
                missing_docs,
                unreachable_pub,
                trivial_casts,
                unused_qualifications,
                clippy::allow_attributes,
                clippy::allow_attributes_without_reason,
                clippy::clone_on_ref_ptr,
                clippy::default_trait_access,
                clippy::derive_partial_eq_without_eq,
                clippy::doc_lazy_continuation,
                clippy::doc_markdown,
                clippy::enum_variant_names,
                clippy::missing_docs_in_private_items,
                clippy::needless_borrows_for_generic_args,
                clippy::too_many_lines,
                clippy::trivially_copy_pass_by_ref,
                reason = "generated protobuf code"
            )]
            pub mod v3 {
                tonic::include_proto!("envoy.service.common.v3");
            }
        }

        /// External processing service types.
        pub mod ext_proc {
            #[allow(
                dead_code,
                missing_docs,
                unreachable_pub,
                trivial_casts,
                unused_qualifications,
                clippy::allow_attributes,
                clippy::allow_attributes_without_reason,
                clippy::clone_on_ref_ptr,
                clippy::default_trait_access,
                clippy::derive_partial_eq_without_eq,
                clippy::doc_lazy_continuation,
                clippy::doc_markdown,
                clippy::enum_variant_names,
                clippy::missing_docs_in_private_items,
                clippy::needless_borrows_for_generic_args,
                clippy::too_many_lines,
                clippy::trivially_copy_pass_by_ref,
                reason = "generated protobuf code"
            )]
            pub mod v3 {
                tonic::include_proto!("envoy.service.ext_proc.v3");
            }
        }
    }
}
