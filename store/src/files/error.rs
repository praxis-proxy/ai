// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Failures at the file catalog boundary.

use std::{error::Error, fmt};

/// Failures at the durable metadata boundary.
#[derive(Debug)]
#[non_exhaustive]
pub enum CatalogError {
    /// An ID or key already exists, or a conditional state transition failed.
    Conflict,

    /// The tenant has no remaining file capacity.
    QuotaExceeded,

    /// The list cursor is invalid in this tenant scope.
    InvalidCursor,

    /// The metadata backend failed; its original error remains available.
    Backend(Box<dyn Error + Send + Sync>),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conflict => write!(f, "file state conflict"),
            Self::QuotaExceeded => write!(f, "tenant file storage limit exceeded"),
            Self::InvalidCursor => write!(f, "invalid file list cursor"),
            Self::Backend(_) => write!(f, "file catalog backend failed"),
        }
    }
}

impl Error for CatalogError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend(source) => Some(source.as_ref()),
            _ => None,
        }
    }
}
