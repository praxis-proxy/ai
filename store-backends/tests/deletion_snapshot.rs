// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Item deletion returns metadata from its mutation lock, without a later read.

#![cfg(any(feature = "sqlite", feature = "_postgres"))]

#[cfg(test)]
mod tests {
    use praxis_ai_store::{ConversationItemRecord, ConversationItemStore, ConversationRecord, StateOwner, StoreError};
    use serde_json::json;

    #[tokio::test]
    async fn memory_delete_snapshot_is_owner_scoped() -> Result<(), Box<dyn std::error::Error>> {
        let store = praxis_ai_store::memory::InMemoryStore::new();
        let owner = seed(&store).await?;
        assert_snapshot(&store, &owner).await
    }

    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn sqlite_delete_snapshot_waits_for_metadata_writer() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let url = format!("sqlite://{}?mode=rwc", dir.path().join("snapshot.db").display());
        let store = praxis_ai_store_backends::SqliteResponseStore::new(
            &url,
            "snapshot_responses",
            "snapshot_conversations",
            Some("snapshot_items"),
            None,
            None,
        )
        .await?;
        let owner = seed(&store).await?;
        let pool = sqlx::SqlitePool::connect(&url).await?;
        let mut writer = pool.begin_with("BEGIN IMMEDIATE").await?;
        sqlx::query("UPDATE snapshot_conversations SET metadata = '{\"topic\":\"committed\"}'")
            .execute(&mut *writer)
            .await?;
        let delete = store.delete_item_and_get_conversation(&owner, "snapshot", "item");
        tokio::pin!(delete);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut delete)
                .await
                .is_err(),
            "delete must wait while the metadata writer holds the parent lock"
        );
        writer.commit().await?;
        check_committed(&store, &owner, delete.await?).await?;
        pool.close().await;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "_postgres")]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn postgres_delete_snapshot_waits_for_metadata_writer() -> Result<(), Box<dyn std::error::Error>> {
        let url = std::env::var("DATABASE_URL")?;
        let store = postgres_store(&url).await?;
        let owner = seed(&store).await?;
        let pool = sqlx::PgPool::connect(&url).await?;
        let mut writer = pool.begin().await?;
        sqlx::query("UPDATE snapshot_conversations SET metadata = '{\"topic\":\"committed\"}' WHERE conversation_id = 'snapshot'")
            .execute(&mut *writer).await?;
        let delete = store.delete_item_and_get_conversation(&owner, "snapshot", "item");
        tokio::pin!(delete);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut delete)
                .await
                .is_err(),
            "delete must wait while the metadata writer holds the parent lock"
        );
        writer.commit().await?;
        check_committed(&store, &owner, delete.await?).await?;
        store.delete_conversation(&owner, "snapshot").await?;
        pool.close().await;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    #[cfg(feature = "_postgres")]
    async fn postgres_store(url: &str) -> Result<praxis_ai_store_backends::PostgresResponseStore, StoreError> {
        praxis_ai_store_backends::PostgresResponseStore::new(
            url,
            "snapshot_responses",
            "snapshot_conversations",
            Some("snapshot_items"),
            &praxis_ai_store_backends::PgTlsConfig {
                ssl_mode: Some(praxis_ai_store::SslMode::Disable),
                ..Default::default()
            },
            None,
            None,
        )
        .await
    }

    async fn check_committed(
        store: &dyn ConversationItemStore,
        owner: &StateOwner,
        snapshot: Option<praxis_ai_store::ConversationSnapshot>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let snapshot = snapshot.ok_or("missing deletion snapshot")?;
        assert_eq!(snapshot.metadata, json!({"topic": "committed"}));
        assert_eq!(snapshot.created_at, 42);
        assert!(store.get_conversation_item(owner, "snapshot", "item").await?.is_none());
        store.delete_conversation(owner, "snapshot").await?;
        let owner = seed(store).await?;
        assert_snapshot(store, &owner).await
    }

    async fn seed(store: &dyn ConversationItemStore) -> Result<StateOwner, StoreError> {
        let owner = StateOwner::from_trusted_parts("snapshot", "issuer", "alice")
            .map_err(|e| StoreError::InvalidInput(e.to_string()))?;
        store
            .create_conversation_with_items(
                &ConversationRecord {
                    conversation_id: "snapshot".to_owned(),
                    owner: owner.clone(),
                    created_at: 42,
                    metadata: json!({"topic": "initial"}),
                    messages: json!([]),
                },
                &[ConversationItemRecord {
                    conversation_id: "snapshot".to_owned(),
                    owner: owner.clone(),
                    item_id: "item".to_owned(),
                    created_at: 42,
                    position: 0,
                    item_data: json!({"id": "item"}),
                }],
            )
            .await?;
        Ok(owner)
    }

    async fn assert_rejected_deletions(
        store: &dyn ConversationItemStore,
        owner: &StateOwner,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let foreign = StateOwner::from_trusted_parts("snapshot", "issuer", "bob")?;
        assert!(matches!(
            store
                .delete_item_and_get_conversation(&foreign, "snapshot", "item")
                .await,
            Err(StoreError::NotFound)
        ));
        assert!(store.get_conversation_item(owner, "snapshot", "item").await?.is_some());
        assert!(
            store
                .delete_item_and_get_conversation(owner, "snapshot", "absent")
                .await?
                .is_none()
        );
        Ok(())
    }

    async fn assert_snapshot(
        store: &dyn ConversationItemStore,
        owner: &StateOwner,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_rejected_deletions(store, owner).await?;
        store
            .update_conversation_metadata(owner, "snapshot", &json!({"topic": "updated"}))
            .await?;
        let snapshot = store
            .delete_item_and_get_conversation(owner, "snapshot", "item")
            .await?
            .ok_or("missing snapshot")?;
        assert_eq!(snapshot.conversation_id, "snapshot");
        assert_eq!(snapshot.created_at, 42);
        assert_eq!(snapshot.metadata, json!({"topic": "updated"}));
        assert!(
            store
                .delete_item_and_get_conversation(owner, "snapshot", "item")
                .await?
                .is_none()
        );
        Ok(())
    }
}
