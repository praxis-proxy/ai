#![allow(
    clippy::missing_docs_in_private_items,
    clippy::too_many_lines,
    reason = "private credential internals"
)]

use std::{collections::HashMap, fs};

use praxis_ai_apis::hash::Sha256;
use praxis_filter::FilterError;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
struct KeyEntry {
    id: String,
    key: String,
}

impl std::fmt::Debug for KeyEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyEntry")
            .field("id", &self.id)
            .field("key", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyFile {
    #[serde(default)]
    premium: Vec<KeyEntry>,
    #[serde(default)]
    main: Vec<KeyEntry>,
    #[serde(default)]
    ratelimit: Vec<KeyEntry>,
}

#[derive(Clone, Debug)]
pub(crate) struct SelectedKey {
    pub id: String,
    pub header_value: String,
}

#[derive(Default)]
pub(crate) struct KeyRing {
    aliases: HashMap<String, SelectedKey>,
    classes: HashMap<String, Vec<SelectedKey>>,
    default_key: Option<String>,
}

impl KeyRing {
    pub(crate) fn load(
        path: Option<&str>,
        header_prefix: &str,
        default_key: Option<String>,
    ) -> Result<Self, FilterError> {
        let mut ring = Self {
            aliases: HashMap::new(),
            classes: HashMap::new(),
            default_key,
        };
        let Some(path) = path else {
            return Ok(ring);
        };
        let raw = fs::read_to_string(path)
            .map_err(|error| -> FilterError { format!("enmaas_policy: key_file could not be read: {error}").into() })?;
        let file: KeyFile = serde_json::from_str(&raw)
            .map_err(|error| -> FilterError { format!("enmaas_policy: key_file is invalid JSON: {error}").into() })?;

        for (class, entries) in [
            ("premium", file.premium),
            ("main", file.main),
            ("ratelimit", file.ratelimit),
        ] {
            for entry in entries {
                if entry.id.trim().is_empty() {
                    return Err(format!("enmaas_policy: {class} key id must not be empty").into());
                }
                if entry.key.is_empty() || entry.key.chars().any(char::is_control) {
                    return Err(format!("enmaas_policy: {class} key '{}' is invalid", entry.id).into());
                }
                let header_value = format!("{header_prefix}{}", entry.key);
                let selected = SelectedKey {
                    id: entry.id.clone(),
                    header_value,
                };
                ring.aliases.insert(format!("{class}:{}", entry.id), selected.clone());
                ring.classes.entry(class.to_owned()).or_default().push(selected);
            }
        }
        Ok(ring)
    }

    pub(crate) fn resolve(&self, alias: Option<&str>, subject: &str) -> Option<SelectedKey> {
        let alias = alias?;
        if let Some(key) = self.aliases.get(alias) {
            return Some(key.clone());
        }
        self.classes
            .get(alias)
            .and_then(|entries| Self::pick(subject, entries))
            .cloned()
    }

    pub(crate) fn default_key(&self) -> Option<SelectedKey> {
        self.default_key.as_ref().map(|value| SelectedKey {
            id: "default".to_owned(),
            header_value: value.clone(),
        })
    }

    fn pick<'a>(subject: &str, entries: &'a [SelectedKey]) -> Option<&'a SelectedKey> {
        entries
            .iter()
            .map(|entry| {
                let mut digest = Sha256::new();
                digest.update(subject.as_bytes());
                digest.update(&[0x1f]);
                digest.update(entry.id.as_bytes());
                (digest.finish(), entry)
            })
            .max_by(|(left, _), (right, _)| left.cmp(right))
            .map(|(_, entry)| entry)
    }
}
