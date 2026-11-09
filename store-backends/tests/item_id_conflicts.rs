// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Transactional conversation item ID conflict classification.

#![cfg(any(feature = "sqlite", feature = "_postgres"))]

#[cfg(test)]
mod tests {
    use praxis_ai_store::{ConversationItemRecord, ConversationItemStore, ConversationRecord, StateOwner, StoreError};
    use serde_json::json;

    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn sqlite_item_id_conflicts_are_invalid_input() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let url = format!("sqlite://{}?mode=rwc", directory.path().join("conflicts.db").display());
        let store = praxis_ai_store_backends::SqliteResponseStore::new(
            &url,
            "conflict_responses",
            "conflict_conversations",
            Some("conflict_items"),
            None,
            None,
        )
        .await?;
        assert_item_id_conflicts(&store).await
    }

    #[tokio::test]
    #[cfg(feature = "_postgres")]
    #[ignore = "requires disposable PostgreSQL via DATABASE_URL"]
    async fn postgres_item_id_conflicts_are_invalid_input() -> Result<(), Box<dyn std::error::Error>> {
        let store = praxis_ai_store_backends::PostgresResponseStore::new(
            &std::env::var("DATABASE_URL")?,
            "conflict_responses",
            "conflict_conversations",
            Some("conflict_items"),
            &praxis_ai_store_backends::PgTlsConfig {
                ssl_mode: Some(praxis_ai_store::SslMode::Disable),
                ..Default::default()
            },
            None,
            None,
        )
        .await?;
        assert_item_id_conflicts(&store).await
    }

    // -----------------------------------------------------------------------------
    // Test Utilities
    // -----------------------------------------------------------------------------

    /// Exercise bounded and unbounded appends, including partial-batch rollback.
    #[expect(
        clippy::too_many_lines,
        reason = "shared regression matrix keeps rollback and history assertions together"
    )]
    async fn assert_item_id_conflicts(store: &dyn ConversationItemStore) -> Result<(), Box<dyn std::error::Error>> {
        let run = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        for bounded in [false, true] {
            let owner = StateOwner::from_trusted_parts(format!("conflict-{run}-{bounded}"), "issuer", "subject")?;
            let first = format!("first-{run}-{bounded}");
            let second = format!("second-{run}-{bounded}");
            for id in [&first, &second] {
                store
                    .upsert_conversation(&ConversationRecord {
                        conversation_id: id.to_owned(),
                        owner: owner.clone(),
                        created_at: 1,
                        metadata: json!({}),
                        messages: json!([{"role": "user", "content": "legacy"}]),
                    })
                    .await?;
            }
            write_items(store, bounded, &owner, &first, &[item(&owner, &first, "shared", 1)]).await?;
            let first_history = store.conversation_history(&owner, &first).await?;
            let second_history = store.conversation_history(&owner, &second).await?;
            for target in [&second, &first] {
                let result = write_items(
                    store,
                    bounded,
                    &owner,
                    target,
                    &[
                        item(&owner, target, "before-conflict", 2),
                        item(&owner, target, "shared", 3),
                    ],
                )
                .await;
                assert!(
                    matches!(result, Err(StoreError::InvalidInput(_))),
                    "duplicate ID must be invalid input: {result:?}"
                );
                assert_eq!(
                    store.conversation_history(&owner, &first).await?,
                    first_history,
                    "source history must survive"
                );
                assert_eq!(
                    store.conversation_history(&owner, &second).await?,
                    second_history,
                    "target legacy history must survive"
                );
                assert!(
                    store
                        .get_conversation_item(&owner, target, "before-conflict")
                        .await?
                        .is_none(),
                    "earlier insert must roll back"
                );
            }
            assert_eq!(
                store
                    .get_conversation(&owner, &second)
                    .await?
                    .map(|record| record.messages),
                second_history.map(serde_json::Value::Array),
                "legacy cache conversion must roll back"
            );
            write_items(
                store,
                bounded,
                &owner,
                &second,
                &[item(&owner, &second, "before-conflict", 2)],
            )
            .await?;

            Box::pin(assert_owner_isolation(store, bounded, &owner)).await?;
            assert_concurrent_conflict(store, &owner, &first, &second).await;

            assert_position_conflict(store, bounded, &owner, &first).await?;
        }
        Ok(())
    }

    /// Unrelated uniqueness violations must not be reported as duplicate item IDs.
    async fn assert_position_conflict(
        store: &dyn ConversationItemStore,
        bounded: bool,
        owner: &StateOwner,
        conversation: &str,
    ) -> Result<(), StoreError> {
        store
            .create_conversation_items(&[item(owner, conversation, "last-position", i64::MAX)])
            .await?;
        let result = write_items(
            store,
            bounded,
            owner,
            conversation,
            &[item(owner, conversation, "position-conflict", 0)],
        )
        .await;
        assert!(
            matches!(result, Err(StoreError::Database(_))),
            "position conflicts must remain database errors: {result:?}"
        );
        Ok(())
    }

    /// Permit identical item IDs when any component of the owner differs.
    async fn assert_owner_isolation(
        store: &dyn ConversationItemStore,
        bounded: bool,
        owner: &StateOwner,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let other_tenant = format!("other-tenant-{}", owner.tenant_id());
        for (index, other) in [
            StateOwner::from_trusted_parts(other_tenant, owner.issuer(), owner.subject())?,
            StateOwner::from_trusted_parts(owner.tenant_id(), "other-issuer", owner.subject())?,
            StateOwner::from_trusted_parts(owner.tenant_id(), owner.issuer(), "other-subject")?,
        ]
        .into_iter()
        .enumerate()
        {
            let id = format!("other-{}-{index}", owner.tenant_id());
            store
                .upsert_conversation(&ConversationRecord {
                    conversation_id: id.clone(),
                    owner: other.clone(),
                    created_at: 1,
                    metadata: json!({}),
                    messages: json!([]),
                })
                .await?;
            write_items(store, bounded, &other, &id, &[item(&other, &id, "shared", 1)]).await?;
            assert!(
                store.get_conversation_item(&other, &id, "shared").await?.is_some(),
                "each owner component isolates IDs"
            );
        }

        Ok(())
    }

    /// Concurrent appends in different conversations must have exactly one winner.
    async fn assert_concurrent_conflict(
        store: &dyn ConversationItemStore,
        owner: &StateOwner,
        first: &str,
        second: &str,
    ) {
        let first_items = [item(owner, first, "racing", 0)];
        let second_items = [item(owner, second, "racing", 0)];
        let results = futures::join!(
            store.create_items_and_sync_messages(owner, first, &first_items),
            store.create_items_and_sync_messages(owner, second, &second_items),
        );
        assert!(
            matches!(
                &results,
                (Ok(()), Err(StoreError::InvalidInput(_))) | (Err(StoreError::InvalidInput(_)), Ok(()))
            ),
            "one concurrent append succeeds and the other is invalid input: {results:?}"
        );
    }

    /// Select the bounded or unbounded transactional append path.
    async fn write_items(
        store: &dyn ConversationItemStore,
        bounded: bool,
        owner: &StateOwner,
        conversation: &str,
        items: &[ConversationItemRecord],
    ) -> Result<(), StoreError> {
        if bounded {
            store
                .create_items_and_sync_messages_bounded(owner, conversation, items, usize::MAX)
                .await
        } else {
            store.create_items_and_sync_messages(owner, conversation, items).await
        }
    }

    /// Build a record with an explicit owner-scoped ID and raw position.
    fn item(owner: &StateOwner, conversation: &str, id: &str, position: i64) -> ConversationItemRecord {
        ConversationItemRecord {
            item_id: id.to_owned(),
            owner: owner.clone(),
            conversation_id: conversation.to_owned(),
            item_data: json!({"id": id, "type": "message", "role": "user", "content": "original"}),
            created_at: 1,
            position,
        }
    }
}
