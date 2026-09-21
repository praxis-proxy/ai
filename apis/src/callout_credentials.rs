//! Per-user callout credentials captured at the trust boundary.
//!
//! The [`crate::callout_credentials::CalloutCredentialsFilter`] establishing filter reads
//! configured ingress headers, stores their values here as [`SecretString`] slots keyed by a
//! config-static slot id, and strips the ingress headers so they never reach an upstream.
//! Callout adapters read a slot through [`crate::callout_identity::stage_callout_identity`]
//! and stage a per-user credential into the nested subrequest instead of a shared provider key.

use std::collections::BTreeMap;

use secrecy::SecretString;

/// Request-scoped map of per-user callout credentials, keyed by config-static slot id.
///
/// Inserted into `RequestExtensions` by the `callout_credentials` filter. Slot ids are safe to
/// log (they come from static config); slot values are secret and never rendered.
#[derive(Default, Clone)]
pub struct CalloutCredentials {
    slots: BTreeMap<String, SecretString>,
}

impl CalloutCredentials {
    /// Create an empty credential set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a secret under `slot`, replacing any prior value for that slot.
    pub fn insert(&mut self, slot: String, value: SecretString) {
        self.slots.insert(slot, value);
    }

    /// Look up the secret staged for `slot`, if any.
    pub fn get(&self, slot: &str) -> Option<&SecretString> {
        self.slots.get(slot)
    }

    /// True when no slots are populated.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Number of populated slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
}

impl std::fmt::Debug for CalloutCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalloutCredentials")
            .field("slots", &self.slots.keys().collect::<Vec<_>>())
            .field("values", &"[REDACTED]")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    #[test]
    fn get_returns_inserted_secret() {
        let mut creds = CalloutCredentials::new();
        creds.insert("brave".to_owned(), SecretString::from("tok-123"));
        assert_eq!(creds.get("brave").unwrap().expose_secret(), "tok-123");
        assert!(creds.get("absent").is_none());
        assert!(!creds.is_empty());
        assert_eq!(creds.len(), 1);
    }

    #[test]
    fn debug_redacts_secret_values_but_lists_slot_names() {
        let mut creds = CalloutCredentials::new();
        creds.insert("brave".to_owned(), SecretString::from("super-secret"));
        let rendered = format!("{creds:?}");
        assert!(rendered.contains("brave"), "slot name should be visible: {rendered}");
        assert!(rendered.contains("REDACTED"), "must mark redaction: {rendered}");
        assert!(
            !rendered.contains("super-secret"),
            "secret value must never appear in Debug: {rendered}"
        );
    }

    #[test]
    fn empty_by_default() {
        let creds = CalloutCredentials::new();
        assert!(creds.is_empty());
        assert_eq!(creds.len(), 0);
    }
}
