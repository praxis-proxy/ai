// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! A2A-specific extraction from parsed JSON-RPC values and A2A request headers.

use std::collections::BTreeMap;

use serde_json::Value;

// -----------------------------------------------------------------------------
// A2aMethod
// -----------------------------------------------------------------------------

/// A2A method classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum A2aMethod {
    /// `SendMessage` message delivery method.
    SendMessage,

    /// `SendStreamingMessage` streaming message method.
    SendStreamingMessage,

    /// `GetTask` task retrieval method.
    GetTask,

    /// `ListTasks` task listing method.
    ListTasks,

    /// `CancelTask` task cancellation method.
    CancelTask,

    /// `SubscribeToTask` task subscription method.
    SubscribeToTask,

    /// `CreateTaskPushNotificationConfig` push notification config creation.
    CreateTaskPushNotificationConfig,

    /// `GetTaskPushNotificationConfig` push notification config retrieval.
    GetTaskPushNotificationConfig,

    /// `ListTaskPushNotificationConfigs` push notification config listing.
    ListTaskPushNotificationConfigs,

    /// `DeleteTaskPushNotificationConfig` push notification config deletion.
    DeleteTaskPushNotificationConfig,

    /// `GetExtendedAgentCard` agent card retrieval.
    GetExtendedAgentCard,

    /// Any other method string not in the known set.
    Unknown(String),
}

/// A2A method family classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum A2aFamily {
    /// Message methods: `SendMessage`, `SendStreamingMessage`.
    Message,

    /// Task methods: `GetTask`, `ListTasks`, `CancelTask`,
    /// `SubscribeToTask`.
    Task,

    /// Push notification config methods.
    PushNotification,

    /// Agent card methods: `GetExtendedAgentCard`.
    AgentCard,

    /// Unknown methods.
    Unknown,
}

impl A2aMethod {
    /// Parse an A2A method from the JSON-RPC method string, with alias support.
    ///
    /// A2A JSON-RPC method strings are matched exactly. Legacy slash-delimited
    /// names are accepted only when configured explicitly in `method_aliases`.
    pub(crate) fn from_method_str(s: &str, aliases: &BTreeMap<String, String>) -> Self {
        // First check if this is an alias
        let canonical_method = aliases.get(s).map_or(s, String::as_str);

        match canonical_method {
            "SendMessage" => Self::SendMessage,
            "SendStreamingMessage" => Self::SendStreamingMessage,
            "GetTask" => Self::GetTask,
            "ListTasks" => Self::ListTasks,
            "CancelTask" => Self::CancelTask,
            "SubscribeToTask" => Self::SubscribeToTask,
            "CreateTaskPushNotificationConfig" => Self::CreateTaskPushNotificationConfig,
            "GetTaskPushNotificationConfig" => Self::GetTaskPushNotificationConfig,
            "ListTaskPushNotificationConfigs" => Self::ListTaskPushNotificationConfigs,
            "DeleteTaskPushNotificationConfig" => Self::DeleteTaskPushNotificationConfig,
            "GetExtendedAgentCard" => Self::GetExtendedAgentCard,
            other => Self::Unknown(other.to_owned()),
        }
    }

    /// String representation for headers and metadata (canonical form).
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::SendMessage => "SendMessage",
            Self::SendStreamingMessage => "SendStreamingMessage",
            Self::GetTask => "GetTask",
            Self::ListTasks => "ListTasks",
            Self::CancelTask => "CancelTask",
            Self::SubscribeToTask => "SubscribeToTask",
            Self::CreateTaskPushNotificationConfig => "CreateTaskPushNotificationConfig",
            Self::GetTaskPushNotificationConfig => "GetTaskPushNotificationConfig",
            Self::ListTaskPushNotificationConfigs => "ListTaskPushNotificationConfigs",
            Self::DeleteTaskPushNotificationConfig => "DeleteTaskPushNotificationConfig",
            Self::GetExtendedAgentCard => "GetExtendedAgentCard",
            Self::Unknown(s) => s,
        }
    }

    /// Get the family classification for this method.
    pub(crate) fn family(&self) -> A2aFamily {
        match self {
            Self::SendMessage | Self::SendStreamingMessage => A2aFamily::Message,
            Self::GetTask | Self::ListTasks | Self::CancelTask | Self::SubscribeToTask => A2aFamily::Task,
            Self::CreateTaskPushNotificationConfig
            | Self::GetTaskPushNotificationConfig
            | Self::ListTaskPushNotificationConfigs
            | Self::DeleteTaskPushNotificationConfig => A2aFamily::PushNotification,
            Self::GetExtendedAgentCard => A2aFamily::AgentCard,
            Self::Unknown(_) => A2aFamily::Unknown,
        }
    }

    /// Whether this method supports streaming responses.
    pub(crate) fn is_streaming(&self) -> bool {
        matches!(self, Self::SendStreamingMessage | Self::SubscribeToTask)
    }

    /// Whether this method should extract task ID from `params.id`.
    pub(crate) fn extracts_task_id(&self) -> bool {
        matches!(self, Self::GetTask | Self::CancelTask | Self::SubscribeToTask)
    }

    /// Whether a follow-up request with this method should be routed
    /// by stored task ownership.
    pub(crate) fn is_task_routable(&self) -> bool {
        matches!(
            self,
            Self::GetTask
                | Self::CancelTask
                | Self::SubscribeToTask
                | Self::CreateTaskPushNotificationConfig
                | Self::GetTaskPushNotificationConfig
                | Self::ListTaskPushNotificationConfigs
                | Self::DeleteTaskPushNotificationConfig
        )
    }

    /// Whether this method should extract task ID from `params.taskId`.
    pub(crate) fn extracts_task_id_from_params(&self) -> bool {
        matches!(
            self,
            Self::CreateTaskPushNotificationConfig
                | Self::GetTaskPushNotificationConfig
                | Self::ListTaskPushNotificationConfigs
                | Self::DeleteTaskPushNotificationConfig
        )
    }
}

impl A2aFamily {
    /// String representation for headers and metadata.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::Task => "task",
            Self::PushNotification => "push_notification",
            Self::AgentCard => "agent_card",
            Self::Unknown => "unknown",
        }
    }
}

// -----------------------------------------------------------------------------
// A2aEnvelope
// -----------------------------------------------------------------------------

/// Extracted A2A envelope metadata.
#[derive(Debug, Clone)]
pub(crate) struct A2aEnvelope {
    /// Context ID from the request, when present.
    pub context_id: Option<String>,
    /// Method family classification.
    pub family: A2aFamily,
    /// Classified A2A method (canonical after alias resolution).
    pub method: A2aMethod,
    /// Original method string before alias resolution (if different).
    pub original_method: Option<String>,
    /// Whether the method supports streaming.
    pub streaming: bool,
    /// Task ID extracted from params, when present.
    pub task_id: Option<String>,
    /// A2A version from `A2A-Version` header, when present.
    pub version: Option<String>,
}

// -----------------------------------------------------------------------------
// Extraction
// -----------------------------------------------------------------------------

/// Extract A2A-specific metadata from a pre-parsed JSON value and request headers.
pub(crate) fn extract_a2a_envelope(
    value: &Value,
    method_str: &str,
    aliases: &BTreeMap<String, String>,
    request_headers: &http::HeaderMap,
) -> A2aEnvelope {
    let method = A2aMethod::from_method_str(method_str, aliases);

    // Track if alias resolution changed the method
    let original_method = aliases.get(method_str).map(|_| method_str.to_owned());

    let family = method.family();
    let streaming = method.is_streaming();
    let task_id = extract_task_id(value, &method);
    let context_id = extract_context_id(value, &method);
    let version = extract_version(request_headers);

    A2aEnvelope {
        context_id,
        family,
        method,
        original_method,
        streaming,
        task_id,
        version,
    }
}

/// Extract task ID from params based on method requirements.
fn extract_task_id(value: &Value, method: &A2aMethod) -> Option<String> {
    let params = value.get("params")?;

    if method.extracts_task_id() {
        // Extract from params.id for task methods
        params.get("id").and_then(|v| v.as_str()).map(str::to_owned)
    } else if method.extracts_task_id_from_params() {
        // Extract from params.taskId for push notification config methods
        params.get("taskId").and_then(|v| v.as_str()).map(str::to_owned)
    } else {
        // No task ID extraction for other methods
        None
    }
}

/// A2A places context IDs at different JSON depths depending on the
/// method, so extraction must be method-aware.
fn extract_context_id(value: &Value, method: &A2aMethod) -> Option<String> {
    let params = value.get("params")?;

    match method {
        A2aMethod::SendMessage | A2aMethod::SendStreamingMessage => params
            .get("message")
            .and_then(|m| m.get("contextId"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        A2aMethod::ListTasks => params.get("contextId").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

/// Extract A2A version from `A2A-Version` request header.
fn extract_version(headers: &http::HeaderMap) -> Option<String> {
    headers
        .get("a2a-version")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use serde_json::json;

    use super::*;

    fn empty_aliases() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    // -------------------------------------------------------------------------
    // A2aMethod::from_method_str — known methods
    // -------------------------------------------------------------------------

    #[test]
    fn from_method_str_send_message() {
        let m = A2aMethod::from_method_str("SendMessage", &empty_aliases());
        assert_eq!(m, A2aMethod::SendMessage);
    }

    #[test]
    fn from_method_str_send_streaming_message() {
        let m = A2aMethod::from_method_str("SendStreamingMessage", &empty_aliases());
        assert_eq!(m, A2aMethod::SendStreamingMessage);
    }

    #[test]
    fn from_method_str_get_task() {
        let m = A2aMethod::from_method_str("GetTask", &empty_aliases());
        assert_eq!(m, A2aMethod::GetTask);
    }

    #[test]
    fn from_method_str_list_tasks() {
        let m = A2aMethod::from_method_str("ListTasks", &empty_aliases());
        assert_eq!(m, A2aMethod::ListTasks);
    }

    #[test]
    fn from_method_str_cancel_task() {
        let m = A2aMethod::from_method_str("CancelTask", &empty_aliases());
        assert_eq!(m, A2aMethod::CancelTask);
    }

    #[test]
    fn from_method_str_subscribe_to_task() {
        let m = A2aMethod::from_method_str("SubscribeToTask", &empty_aliases());
        assert_eq!(m, A2aMethod::SubscribeToTask);
    }

    #[test]
    fn from_method_str_create_task_push_notification_config() {
        let m = A2aMethod::from_method_str("CreateTaskPushNotificationConfig", &empty_aliases());
        assert_eq!(m, A2aMethod::CreateTaskPushNotificationConfig);
    }

    #[test]
    fn from_method_str_get_task_push_notification_config() {
        let m = A2aMethod::from_method_str("GetTaskPushNotificationConfig", &empty_aliases());
        assert_eq!(m, A2aMethod::GetTaskPushNotificationConfig);
    }

    #[test]
    fn from_method_str_list_task_push_notification_configs() {
        let m = A2aMethod::from_method_str("ListTaskPushNotificationConfigs", &empty_aliases());
        assert_eq!(m, A2aMethod::ListTaskPushNotificationConfigs);
    }

    #[test]
    fn from_method_str_delete_task_push_notification_config() {
        let m = A2aMethod::from_method_str("DeleteTaskPushNotificationConfig", &empty_aliases());
        assert_eq!(m, A2aMethod::DeleteTaskPushNotificationConfig);
    }

    #[test]
    fn from_method_str_get_extended_agent_card() {
        let m = A2aMethod::from_method_str("GetExtendedAgentCard", &empty_aliases());
        assert_eq!(m, A2aMethod::GetExtendedAgentCard);
    }

    #[test]
    fn from_method_str_unknown() {
        let m = A2aMethod::from_method_str("DoSomethingElse", &empty_aliases());
        assert_eq!(m, A2aMethod::Unknown("DoSomethingElse".to_owned()));
    }

    #[test]
    fn from_method_str_alias_resolution() {
        let mut aliases = BTreeMap::new();
        aliases.insert("tasks/send".to_owned(), "SendMessage".to_owned());

        let m = A2aMethod::from_method_str("tasks/send", &aliases);
        assert_eq!(m, A2aMethod::SendMessage, "alias should resolve to canonical method");
    }

    #[test]
    fn from_method_str_alias_to_unknown_canonical() {
        let mut aliases = BTreeMap::new();
        aliases.insert("legacy/foo".to_owned(), "NotARealMethod".to_owned());

        let m = A2aMethod::from_method_str("legacy/foo", &aliases);
        assert_eq!(
            m,
            A2aMethod::Unknown("NotARealMethod".to_owned()),
            "alias resolving to unknown canonical stays Unknown"
        );
    }

    // -------------------------------------------------------------------------
    // A2aMethod::as_str — round-trip
    // -------------------------------------------------------------------------

    #[test]
    fn as_str_round_trip_all_known_methods() {
        let cases = [
            ("SendMessage", A2aMethod::SendMessage),
            ("SendStreamingMessage", A2aMethod::SendStreamingMessage),
            ("GetTask", A2aMethod::GetTask),
            ("ListTasks", A2aMethod::ListTasks),
            ("CancelTask", A2aMethod::CancelTask),
            ("SubscribeToTask", A2aMethod::SubscribeToTask),
            (
                "CreateTaskPushNotificationConfig",
                A2aMethod::CreateTaskPushNotificationConfig,
            ),
            (
                "GetTaskPushNotificationConfig",
                A2aMethod::GetTaskPushNotificationConfig,
            ),
            (
                "ListTaskPushNotificationConfigs",
                A2aMethod::ListTaskPushNotificationConfigs,
            ),
            (
                "DeleteTaskPushNotificationConfig",
                A2aMethod::DeleteTaskPushNotificationConfig,
            ),
            ("GetExtendedAgentCard", A2aMethod::GetExtendedAgentCard),
        ];

        for (expected, method) in &cases {
            assert_eq!(method.as_str(), *expected, "as_str mismatch for {method:?}");
        }
    }

    #[test]
    fn as_str_unknown_preserves_original() {
        let m = A2aMethod::Unknown("CustomMethod".to_owned());
        assert_eq!(m.as_str(), "CustomMethod");
    }

    #[test]
    fn from_method_str_as_str_round_trip() {
        let methods = [
            "SendMessage",
            "SendStreamingMessage",
            "GetTask",
            "ListTasks",
            "CancelTask",
            "SubscribeToTask",
            "CreateTaskPushNotificationConfig",
            "GetTaskPushNotificationConfig",
            "ListTaskPushNotificationConfigs",
            "DeleteTaskPushNotificationConfig",
            "GetExtendedAgentCard",
        ];

        for name in &methods {
            let parsed = A2aMethod::from_method_str(name, &empty_aliases());
            assert_eq!(parsed.as_str(), *name, "round-trip failed for {name}");
        }
    }

    // -------------------------------------------------------------------------
    // A2aMethod::family
    // -------------------------------------------------------------------------

    #[test]
    fn family_message_methods() {
        assert_eq!(A2aMethod::SendMessage.family(), A2aFamily::Message);
        assert_eq!(A2aMethod::SendStreamingMessage.family(), A2aFamily::Message);
    }

    #[test]
    fn family_task_methods() {
        assert_eq!(A2aMethod::GetTask.family(), A2aFamily::Task);
        assert_eq!(A2aMethod::ListTasks.family(), A2aFamily::Task);
        assert_eq!(A2aMethod::CancelTask.family(), A2aFamily::Task);
        assert_eq!(A2aMethod::SubscribeToTask.family(), A2aFamily::Task);
    }

    #[test]
    fn family_push_notification_methods() {
        assert_eq!(
            A2aMethod::CreateTaskPushNotificationConfig.family(),
            A2aFamily::PushNotification
        );
        assert_eq!(
            A2aMethod::GetTaskPushNotificationConfig.family(),
            A2aFamily::PushNotification
        );
        assert_eq!(
            A2aMethod::ListTaskPushNotificationConfigs.family(),
            A2aFamily::PushNotification
        );
        assert_eq!(
            A2aMethod::DeleteTaskPushNotificationConfig.family(),
            A2aFamily::PushNotification
        );
    }

    #[test]
    fn family_agent_card_method() {
        assert_eq!(A2aMethod::GetExtendedAgentCard.family(), A2aFamily::AgentCard);
    }

    #[test]
    fn family_unknown_method() {
        assert_eq!(A2aMethod::Unknown("Foo".to_owned()).family(), A2aFamily::Unknown);
    }

    // -------------------------------------------------------------------------
    // A2aFamily::as_str
    // -------------------------------------------------------------------------

    #[test]
    fn family_as_str_all_variants() {
        assert_eq!(A2aFamily::Message.as_str(), "message");
        assert_eq!(A2aFamily::Task.as_str(), "task");
        assert_eq!(A2aFamily::PushNotification.as_str(), "push_notification");
        assert_eq!(A2aFamily::AgentCard.as_str(), "agent_card");
        assert_eq!(A2aFamily::Unknown.as_str(), "unknown");
    }

    // -------------------------------------------------------------------------
    // is_streaming
    // -------------------------------------------------------------------------

    #[test]
    fn is_streaming_true_cases() {
        assert!(A2aMethod::SendStreamingMessage.is_streaming());
        assert!(A2aMethod::SubscribeToTask.is_streaming());
    }

    #[test]
    fn is_streaming_false_cases() {
        let non_streaming = [
            A2aMethod::SendMessage,
            A2aMethod::GetTask,
            A2aMethod::ListTasks,
            A2aMethod::CancelTask,
            A2aMethod::CreateTaskPushNotificationConfig,
            A2aMethod::GetTaskPushNotificationConfig,
            A2aMethod::ListTaskPushNotificationConfigs,
            A2aMethod::DeleteTaskPushNotificationConfig,
            A2aMethod::GetExtendedAgentCard,
            A2aMethod::Unknown("Anything".to_owned()),
        ];

        for m in &non_streaming {
            assert!(!m.is_streaming(), "{m:?} should not be streaming");
        }
    }

    // -------------------------------------------------------------------------
    // extracts_task_id
    // -------------------------------------------------------------------------

    #[test]
    fn extracts_task_id_true_cases() {
        assert!(A2aMethod::GetTask.extracts_task_id());
        assert!(A2aMethod::CancelTask.extracts_task_id());
        assert!(A2aMethod::SubscribeToTask.extracts_task_id());
    }

    #[test]
    fn extracts_task_id_false_cases() {
        let no_extract = [
            A2aMethod::SendMessage,
            A2aMethod::SendStreamingMessage,
            A2aMethod::ListTasks,
            A2aMethod::CreateTaskPushNotificationConfig,
            A2aMethod::GetTaskPushNotificationConfig,
            A2aMethod::ListTaskPushNotificationConfigs,
            A2aMethod::DeleteTaskPushNotificationConfig,
            A2aMethod::GetExtendedAgentCard,
            A2aMethod::Unknown("X".to_owned()),
        ];

        for m in &no_extract {
            assert!(!m.extracts_task_id(), "{m:?} should not extract task_id");
        }
    }

    // -------------------------------------------------------------------------
    // is_task_routable
    // -------------------------------------------------------------------------

    #[test]
    fn is_task_routable_true_cases() {
        let routable = [
            A2aMethod::GetTask,
            A2aMethod::CancelTask,
            A2aMethod::SubscribeToTask,
            A2aMethod::CreateTaskPushNotificationConfig,
            A2aMethod::GetTaskPushNotificationConfig,
            A2aMethod::ListTaskPushNotificationConfigs,
            A2aMethod::DeleteTaskPushNotificationConfig,
        ];

        for m in &routable {
            assert!(m.is_task_routable(), "{m:?} should be task-routable");
        }
    }

    #[test]
    fn is_task_routable_false_cases() {
        let not_routable = [
            A2aMethod::SendMessage,
            A2aMethod::SendStreamingMessage,
            A2aMethod::ListTasks,
            A2aMethod::GetExtendedAgentCard,
            A2aMethod::Unknown("X".to_owned()),
        ];

        for m in &not_routable {
            assert!(!m.is_task_routable(), "{m:?} should not be task-routable");
        }
    }

    // -------------------------------------------------------------------------
    // extracts_task_id_from_params
    // -------------------------------------------------------------------------

    #[test]
    fn extracts_task_id_from_params_true_cases() {
        let from_params = [
            A2aMethod::CreateTaskPushNotificationConfig,
            A2aMethod::GetTaskPushNotificationConfig,
            A2aMethod::ListTaskPushNotificationConfigs,
            A2aMethod::DeleteTaskPushNotificationConfig,
        ];

        for m in &from_params {
            assert!(
                m.extracts_task_id_from_params(),
                "{m:?} should extract task_id from params"
            );
        }
    }

    #[test]
    fn extracts_task_id_from_params_false_cases() {
        let not_from_params = [
            A2aMethod::SendMessage,
            A2aMethod::SendStreamingMessage,
            A2aMethod::GetTask,
            A2aMethod::ListTasks,
            A2aMethod::CancelTask,
            A2aMethod::SubscribeToTask,
            A2aMethod::GetExtendedAgentCard,
            A2aMethod::Unknown("X".to_owned()),
        ];

        for m in &not_from_params {
            assert!(
                !m.extracts_task_id_from_params(),
                "{m:?} should not extract task_id from params"
            );
        }
    }

    // -------------------------------------------------------------------------
    // extract_task_id
    // -------------------------------------------------------------------------

    #[test]
    fn extract_task_id_from_params_id() {
        let value = json!({
            "params": { "id": "task-123" }
        });

        let result = extract_task_id(&value, &A2aMethod::GetTask);
        assert_eq!(result.as_deref(), Some("task-123"));
    }

    #[test]
    fn extract_task_id_from_params_task_id() {
        let value = json!({
            "params": { "taskId": "task-456" }
        });

        let result = extract_task_id(&value, &A2aMethod::CreateTaskPushNotificationConfig);
        assert_eq!(result.as_deref(), Some("task-456"));
    }

    #[test]
    fn extract_task_id_no_params() {
        let value = json!({});

        let result = extract_task_id(&value, &A2aMethod::GetTask);
        assert_eq!(result, None, "missing params should return None");
    }

    #[test]
    fn extract_task_id_missing_id_field() {
        let value = json!({
            "params": { "other": "value" }
        });

        let result = extract_task_id(&value, &A2aMethod::GetTask);
        assert_eq!(result, None, "missing id field should return None");
    }

    #[test]
    fn extract_task_id_non_extracting_method() {
        let value = json!({
            "params": { "id": "task-789", "taskId": "task-789" }
        });

        let result = extract_task_id(&value, &A2aMethod::SendMessage);
        assert_eq!(result, None, "SendMessage should not extract task_id");
    }

    #[test]
    fn extract_task_id_cancel_task() {
        let value = json!({
            "params": { "id": "task-cancel-1" }
        });

        let result = extract_task_id(&value, &A2aMethod::CancelTask);
        assert_eq!(result.as_deref(), Some("task-cancel-1"));
    }

    #[test]
    fn extract_task_id_subscribe_to_task() {
        let value = json!({
            "params": { "id": "task-sub-1" }
        });

        let result = extract_task_id(&value, &A2aMethod::SubscribeToTask);
        assert_eq!(result.as_deref(), Some("task-sub-1"));
    }

    #[test]
    fn extract_task_id_push_notification_methods() {
        let value = json!({
            "params": { "taskId": "task-push-1" }
        });

        let push_methods = [
            A2aMethod::GetTaskPushNotificationConfig,
            A2aMethod::ListTaskPushNotificationConfigs,
            A2aMethod::DeleteTaskPushNotificationConfig,
        ];

        for method in &push_methods {
            let result = extract_task_id(&value, method);
            assert_eq!(
                result.as_deref(),
                Some("task-push-1"),
                "{method:?} should extract from params.taskId"
            );
        }
    }

    // -------------------------------------------------------------------------
    // extract_context_id
    // -------------------------------------------------------------------------

    #[test]
    fn extract_context_id_send_message() {
        let value = json!({
            "params": {
                "message": {
                    "contextId": "ctx-send-1"
                }
            }
        });

        let result = extract_context_id(&value, &A2aMethod::SendMessage);
        assert_eq!(result.as_deref(), Some("ctx-send-1"));
    }

    #[test]
    fn extract_context_id_send_streaming_message() {
        let value = json!({
            "params": {
                "message": {
                    "contextId": "ctx-stream-1"
                }
            }
        });

        let result = extract_context_id(&value, &A2aMethod::SendStreamingMessage);
        assert_eq!(result.as_deref(), Some("ctx-stream-1"));
    }

    #[test]
    fn extract_context_id_list_tasks() {
        let value = json!({
            "params": {
                "contextId": "ctx-list-1"
            }
        });

        let result = extract_context_id(&value, &A2aMethod::ListTasks);
        assert_eq!(result.as_deref(), Some("ctx-list-1"));
    }

    #[test]
    fn extract_context_id_other_method_returns_none() {
        let value = json!({
            "params": {
                "contextId": "ctx-ignored",
                "message": { "contextId": "also-ignored" }
            }
        });

        let result = extract_context_id(&value, &A2aMethod::GetTask);
        assert_eq!(result, None, "GetTask should not extract context_id");
    }

    #[test]
    fn extract_context_id_no_params() {
        let value = json!({});

        let result = extract_context_id(&value, &A2aMethod::SendMessage);
        assert_eq!(result, None, "missing params should return None");
    }

    #[test]
    fn extract_context_id_missing_message_field() {
        let value = json!({
            "params": { "other": "value" }
        });

        let result = extract_context_id(&value, &A2aMethod::SendMessage);
        assert_eq!(result, None, "missing message field should return None");
    }

    #[test]
    fn extract_context_id_missing_context_id_in_message() {
        let value = json!({
            "params": {
                "message": { "role": "user" }
            }
        });

        let result = extract_context_id(&value, &A2aMethod::SendMessage);
        assert_eq!(result, None, "missing contextId in message should return None");
    }

    // -------------------------------------------------------------------------
    // extract_version
    // -------------------------------------------------------------------------

    #[test]
    fn extract_version_present() {
        let mut headers = http::HeaderMap::new();
        headers.insert("a2a-version", http::HeaderValue::from_static("0.2.1"));

        let result = extract_version(&headers);
        assert_eq!(result.as_deref(), Some("0.2.1"));
    }

    #[test]
    fn extract_version_missing() {
        let headers = http::HeaderMap::new();

        let result = extract_version(&headers);
        assert_eq!(result, None, "missing header should return None");
    }

    #[test]
    fn extract_version_case_insensitive() {
        let mut headers = http::HeaderMap::new();
        headers.insert("A2A-Version", http::HeaderValue::from_static("1.0.0"));

        let result = extract_version(&headers);
        assert_eq!(result.as_deref(), Some("1.0.0"), "HTTP headers are case-insensitive");
    }

    // -------------------------------------------------------------------------
    // extract_a2a_envelope — full integration
    // -------------------------------------------------------------------------

    #[test]
    fn envelope_send_message_with_context_and_version() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "id": 1,
            "params": {
                "message": {
                    "role": "user",
                    "contextId": "ctx-abc"
                }
            }
        });

        let mut headers = http::HeaderMap::new();
        headers.insert("a2a-version", http::HeaderValue::from_static("0.2.1"));

        let env = extract_a2a_envelope(&value, "SendMessage", &empty_aliases(), &headers);

        assert_eq!(env.method, A2aMethod::SendMessage);
        assert_eq!(env.family, A2aFamily::Message);
        assert!(!env.streaming);
        assert_eq!(env.context_id.as_deref(), Some("ctx-abc"));
        assert_eq!(env.task_id, None);
        assert_eq!(env.version.as_deref(), Some("0.2.1"));
        assert_eq!(env.original_method, None, "no alias used");
    }

    #[test]
    fn envelope_get_task_with_task_id() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "GetTask",
            "id": 2,
            "params": { "id": "task-42" }
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "GetTask", &empty_aliases(), &headers);

        assert_eq!(env.method, A2aMethod::GetTask);
        assert_eq!(env.family, A2aFamily::Task);
        assert!(!env.streaming);
        assert_eq!(env.task_id.as_deref(), Some("task-42"));
        assert_eq!(env.context_id, None);
        assert_eq!(env.version, None);
    }

    #[test]
    fn envelope_streaming_subscribe() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "SubscribeToTask",
            "id": 3,
            "params": { "id": "task-99" }
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "SubscribeToTask", &empty_aliases(), &headers);

        assert_eq!(env.method, A2aMethod::SubscribeToTask);
        assert_eq!(env.family, A2aFamily::Task);
        assert!(env.streaming, "SubscribeToTask should be streaming");
        assert_eq!(env.task_id.as_deref(), Some("task-99"));
    }

    #[test]
    fn envelope_push_notification_config_extracts_task_id() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "CreateTaskPushNotificationConfig",
            "id": 4,
            "params": { "taskId": "task-push-42" }
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "CreateTaskPushNotificationConfig", &empty_aliases(), &headers);

        assert_eq!(env.method, A2aMethod::CreateTaskPushNotificationConfig);
        assert_eq!(env.family, A2aFamily::PushNotification);
        assert!(!env.streaming);
        assert_eq!(env.task_id.as_deref(), Some("task-push-42"));
    }

    #[test]
    fn envelope_unknown_method() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "CustomExtension",
            "id": 5,
            "params": {}
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "CustomExtension", &empty_aliases(), &headers);

        assert_eq!(env.method, A2aMethod::Unknown("CustomExtension".to_owned()));
        assert_eq!(env.family, A2aFamily::Unknown);
        assert!(!env.streaming);
        assert_eq!(env.task_id, None);
        assert_eq!(env.context_id, None);
    }

    // -------------------------------------------------------------------------
    // Alias tracking — original_method
    // -------------------------------------------------------------------------

    #[test]
    fn envelope_alias_sets_original_method() {
        let mut aliases = BTreeMap::new();
        aliases.insert("tasks/get".to_owned(), "GetTask".to_owned());

        let value = json!({
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "id": 6,
            "params": { "id": "task-aliased" }
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "tasks/get", &aliases, &headers);

        assert_eq!(env.method, A2aMethod::GetTask, "alias should resolve to GetTask");
        assert_eq!(
            env.original_method.as_deref(),
            Some("tasks/get"),
            "original method should be preserved"
        );
        assert_eq!(env.task_id.as_deref(), Some("task-aliased"));
    }

    #[test]
    fn envelope_no_alias_leaves_original_method_none() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "GetTask",
            "id": 7,
            "params": { "id": "task-direct" }
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "GetTask", &empty_aliases(), &headers);

        assert_eq!(env.original_method, None, "no alias means no original_method");
    }

    #[test]
    fn envelope_list_tasks_with_context_id() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "ListTasks",
            "id": 8,
            "params": { "contextId": "ctx-list-all" }
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "ListTasks", &empty_aliases(), &headers);

        assert_eq!(env.method, A2aMethod::ListTasks);
        assert_eq!(env.family, A2aFamily::Task);
        assert_eq!(env.context_id.as_deref(), Some("ctx-list-all"));
        assert_eq!(env.task_id, None);
    }

    #[test]
    fn envelope_agent_card() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "GetExtendedAgentCard",
            "id": 9,
            "params": {}
        });

        let headers = http::HeaderMap::new();
        let env = extract_a2a_envelope(&value, "GetExtendedAgentCard", &empty_aliases(), &headers);

        assert_eq!(env.method, A2aMethod::GetExtendedAgentCard);
        assert_eq!(env.family, A2aFamily::AgentCard);
        assert!(!env.streaming);
        assert_eq!(env.task_id, None);
        assert_eq!(env.context_id, None);
    }
}
