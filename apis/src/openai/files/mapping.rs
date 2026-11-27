// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Conversions between the service DTOs, the catalog records and the wire
//! contract.

use praxis_ai_store::{CatalogError, FileListQuery, FileRecord, FileRecordPage, SortDirection};

use super::{
    FileInfo, FileList, FileListOrder, FilePage, FilePurpose,
    contracts::{ListFilesQuery, ListFilesResponse, OpenAiFile},
};

impl From<FileListOrder> for SortDirection {
    fn from(order: FileListOrder) -> Self {
        match order {
            FileListOrder::Asc => Self::Asc,
            FileListOrder::Desc => Self::Desc,
        }
    }
}

impl From<&FileList> for FileListQuery {
    fn from(list: &FileList) -> Self {
        Self {
            purpose: list.purpose.clone(),
            limit: list.limit,
            order: list.order.into(),
            after: list.after.clone(),
        }
    }
}

impl TryFrom<FileRecord> for FileInfo {
    type Error = CatalogError;

    /// A stored purpose that does not parse is a catalog backend error, not
    /// a client error.
    fn try_from(record: FileRecord) -> Result<Self, Self::Error> {
        let purpose = record
            .purpose
            .parse::<FilePurpose>()
            .map_err(|err| CatalogError::Backend(Box::new(err)))?;
        Ok(Self {
            id: record.id,
            bytes: record.bytes,
            created_at: record.created_at,
            expires_at: record.expires_at,
            filename: record.filename,
            purpose,
        })
    }
}

impl TryFrom<FileRecordPage> for FilePage {
    type Error = CatalogError;

    fn try_from(page: FileRecordPage) -> Result<Self, Self::Error> {
        let data = page
            .data
            .into_iter()
            .map(FileInfo::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            data,
            has_more: page.has_more,
        })
    }
}

impl From<FileInfo> for OpenAiFile {
    fn from(info: FileInfo) -> Self {
        let file = Self::new(info.id, info.bytes, info.created_at, info.filename, info.purpose);
        match info.expires_at {
            Some(expires_at) => file.with_expires_at(expires_at),
            None => file,
        }
    }
}

/// `first_id` and `last_id` are derived from the page.
impl From<FilePage> for ListFilesResponse {
    fn from(page: FilePage) -> Self {
        Self::new(page.data.into_iter().map(OpenAiFile::from).collect(), page.has_more)
    }
}

impl From<ListFilesQuery> for FileList {
    fn from(query: ListFilesQuery) -> Self {
        Self {
            limit: query.effective_limit(),
            purpose: query.purpose,
            order: query.order,
            after: query.after,
        }
    }
}
