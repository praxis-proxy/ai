// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! SQLite test database helpers.

use std::path::{Path, PathBuf};

/// File-backed SQLite database URL that cleans up its files on drop.
#[must_use]
pub struct TempSqlite {
    /// SQLx-compatible database URL pointing at `path`.
    url: String,
    /// Database file path under `_dir`.
    path: PathBuf,
    /// Temporary directory that owns the database file and SQLite sidecars.
    _dir: tempfile::TempDir,
}

impl TempSqlite {
    /// Generate a unique file-backed SQLite URL for test isolation.
    ///
    /// # Panics
    ///
    /// Panics if the temporary database directory cannot be created.
    pub fn new(test_name: &str) -> Self {
        let dir = tempfile::Builder::new()
            .prefix(&format!("praxis_integ_{test_name}_"))
            .tempdir()
            .expect("failed to create temporary SQLite database directory");
        let path = dir.path().join("responses.db");
        let url = format!("sqlite://{}?mode=rwc", path.display());

        Self { url, path, _dir: dir }
    }

    /// Return the SQLx-compatible SQLite URL for this database.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Return the SQLite database file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_sqlite_url_uses_file_backed_sqlite() {
        let db = TempSqlite::new("url");

        assert!(db.url().starts_with("sqlite://"));
        assert!(db.url().ends_with("?mode=rwc"));
    }

    #[test]
    fn drop_removes_sqlite_files() {
        let db = TempSqlite::new("drop");
        let db_path = db.path().to_path_buf();
        let dir_path = db_path
            .parent()
            .expect("database path should have a parent")
            .to_path_buf();
        let shm_path = dir_path.join("responses.db-shm");
        let wal_path = dir_path.join("responses.db-wal");

        std::fs::write(&db_path, b"db").expect("should write database file");
        std::fs::write(&shm_path, b"shm").expect("should write shm file");
        std::fs::write(&wal_path, b"wal").expect("should write wal file");

        drop(db);

        assert!(!dir_path.exists(), "database directory should be deleted");
    }
}
