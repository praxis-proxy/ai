// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Principal matching for catalog entry visibility.

use std::collections::{BTreeSet, HashSet};

use praxis_filter::AuthenticatedIdentity;

use super::config::{DefaultVisibility, VisibleToConfig};

/// Borrowed view of the caller's authenticated identity.
///
/// Kept separate from [`AuthenticatedIdentity`] so unit tests can drive
/// matching without the crate-private core constructor.
#[derive(Debug, Clone, Copy)]
pub(super) struct PrincipalView<'a> {
    /// Authenticated subject ID.
    pub subject: &'a str,
    /// Authenticated roles.
    pub roles: &'a BTreeSet<String>,
    /// Authenticated teams.
    pub teams: &'a BTreeSet<String>,
}

impl<'a> From<&'a AuthenticatedIdentity> for PrincipalView<'a> {
    fn from(identity: &'a AuthenticatedIdentity) -> Self {
        Self {
            subject: identity.subject_id(),
            roles: identity.roles(),
            teams: identity.teams(),
        }
    }
}

/// Compiled visibility rule for one catalog entry.
#[derive(Debug)]
pub(super) enum Visibility {
    /// Visible to every caller.
    Public,
    /// Visible to any authenticated caller.
    Authenticated,
    /// Visible to callers matching any listed subject, role, or team.
    Restricted(Box<AllowedPrincipals>),
}

/// Principals allowed to see a restricted entry.
#[derive(Debug)]
pub(super) struct AllowedPrincipals {
    /// Allowed subject IDs.
    subjects: HashSet<String>,
    /// Allowed roles.
    roles: HashSet<String>,
    /// Allowed teams.
    teams: HashSet<String>,
}

impl Visibility {
    /// Compile an entry's rule, falling back to the catalog default.
    pub(super) fn compile(rule: Option<VisibleToConfig>, default: DefaultVisibility) -> Self {
        match rule {
            Some(rule) => Self::Restricted(Box::new(AllowedPrincipals {
                subjects: rule.subjects.into_iter().collect(),
                roles: rule.roles.into_iter().collect(),
                teams: rule.teams.into_iter().collect(),
            })),
            None => match default {
                DefaultVisibility::Public => Self::Public,
                DefaultVisibility::Authenticated => Self::Authenticated,
            },
        }
    }

    /// Whether `principal` may see the entry.
    pub(super) fn allows(&self, principal: Option<PrincipalView<'_>>) -> bool {
        match (self, principal) {
            (Self::Public, _) => true,
            (Self::Authenticated, principal) => principal.is_some(),
            (Self::Restricted(_), None) => false,
            (Self::Restricted(allowed), Some(principal)) => {
                allowed.subjects.contains(principal.subject)
                    || principal.roles.iter().any(|role| allowed.roles.contains(role))
                    || principal.teams.iter().any(|team| allowed.teams.contains(team))
            },
        }
    }
}
