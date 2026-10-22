// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Generation-scoped Sentinel primary-cache state machine.

/// Result of beginning a primary discovery attempt.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum DiscoveryStart {
    /// The caller owns the discovery attempt.
    Started,
    /// Another caller is already discovering this generation's primary.
    InProgress,
}

/// Result of publishing a discovered primary.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum PublishResult {
    /// The primary belongs to the current discovery generation.
    Accepted,
    /// The discovery completed after its generation was invalidated.
    Discarded,
}

/// A lookup result for the generation-scoped primary cache.
pub(crate) enum Lookup<'a, T> {
    /// A primary client cached for the requested generation.
    Hit(&'a T),
    /// No primary is cached for the requested generation.
    Miss,
    /// Discovery for the requested generation is already in progress.
    InProgress,
}

/// Internal state for one generation-scoped primary cache.
enum State<T> {
    /// No primary or discovery attempt is retained.
    Empty,
    /// A discovery attempt owns this generation.
    Discovering {
        /// Generation being discovered.
        generation: u64,
    },
    /// A primary is ready for this generation.
    Ready {
        /// Generation that produced the primary.
        generation: u64,
        /// Discovered primary client.
        primary: T,
    },
}

/// State machine for coalescing and fencing Sentinel primary discovery.
pub(crate) struct PrimaryCache<T> {
    /// Current cache state.
    state: State<T>,
}

impl<T> PrimaryCache<T> {
    /// Create an empty primary cache.
    pub(crate) fn new() -> Self {
        Self { state: State::Empty }
    }

    /// Look up a primary for `generation`.
    pub(crate) fn lookup(&self, generation: u64) -> Lookup<'_, T> {
        match &self.state {
            State::Ready {
                generation: cached_generation,
                primary,
            } if *cached_generation == generation => Lookup::Hit(primary),
            State::Discovering {
                generation: discovery_generation,
            } if *discovery_generation == generation => Lookup::InProgress,
            State::Empty | State::Discovering { .. } | State::Ready { .. } => Lookup::Miss,
        }
    }

    /// Begin discovery for `generation`, coalescing concurrent callers.
    pub(crate) fn begin_discovery(&mut self, generation: u64) -> DiscoveryStart {
        match &self.state {
            State::Discovering {
                generation: discovery_generation,
            } if *discovery_generation == generation => DiscoveryStart::InProgress,
            State::Ready {
                generation: cached_generation,
                ..
            } if *cached_generation == generation => DiscoveryStart::InProgress,
            State::Empty | State::Discovering { .. } | State::Ready { .. } => {
                self.state = State::Discovering { generation };
                DiscoveryStart::Started
            },
        }
    }

    /// Publish a discovery result only if its generation is still current.
    pub(crate) fn publish(&mut self, generation: u64, primary: T) -> PublishResult {
        if matches!(
            &self.state,
            State::Discovering {
                generation: discovery_generation,
            } if *discovery_generation == generation
        ) {
            self.state = State::Ready { generation, primary };
            PublishResult::Accepted
        } else {
            PublishResult::Discarded
        }
    }

    /// Invalidate the cached primary only when `generation` owns it.
    pub(crate) fn invalidate(&mut self, generation: u64) -> bool {
        let owns_generation = matches!(
            &self.state,
            State::Discovering {
                generation: state_generation,
            }
                | State::Ready {
                    generation: state_generation,
                    ..
                } if *state_generation == generation
        );
        if owns_generation {
            self.state = State::Empty;
        }
        owns_generation
    }
}

#[cfg(test)]
mod tests {
    use super::{DiscoveryStart, Lookup, PrimaryCache, PublishResult};

    #[test]
    fn empty_cache_starts_discovery_on_miss() {
        let mut cache = PrimaryCache::<&str>::new();

        assert!(matches!(cache.lookup(0), Lookup::Miss));
        assert_eq!(cache.begin_discovery(0), DiscoveryStart::Started);
    }

    #[test]
    fn concurrent_discovery_is_coalesced_for_one_generation() {
        let mut cache = PrimaryCache::<&str>::new();

        assert_eq!(cache.begin_discovery(7), DiscoveryStart::Started);
        assert_eq!(cache.begin_discovery(7), DiscoveryStart::InProgress);
    }

    #[test]
    fn current_generation_lookup_reports_discovery_in_progress() {
        let mut cache = PrimaryCache::<&str>::new();

        assert_eq!(cache.begin_discovery(7), DiscoveryStart::Started);
        assert!(matches!(cache.lookup(7), Lookup::InProgress));
        assert!(matches!(cache.lookup(8), Lookup::Miss));
    }

    #[test]
    fn published_primary_is_a_hit_for_its_generation() {
        let mut cache = PrimaryCache::<&str>::new();
        cache.begin_discovery(3);

        assert_eq!(cache.publish(3, "primary-a"), PublishResult::Accepted);
        assert!(matches!(cache.lookup(3), Lookup::Hit(&"primary-a")));
    }

    #[test]
    fn a_generation_change_turns_a_cached_primary_into_a_miss() {
        let mut cache = PrimaryCache::<&str>::new();
        cache.begin_discovery(1);
        cache.publish(1, "primary-a");

        assert!(matches!(cache.lookup(2), Lookup::Miss));
        assert_eq!(cache.begin_discovery(2), DiscoveryStart::Started);
    }

    #[test]
    fn a_stale_discovery_result_is_discarded_after_invalidation() {
        let mut cache = PrimaryCache::<&str>::new();
        cache.begin_discovery(4);
        assert!(cache.invalidate(4));

        assert_eq!(cache.publish(4, "old-primary"), PublishResult::Discarded);
        assert!(matches!(cache.lookup(5), Lookup::Miss));
    }

    #[test]
    fn stale_invalidation_does_not_evict_a_newer_primary() {
        let mut cache = PrimaryCache::<&str>::new();
        cache.begin_discovery(9);
        cache.publish(9, "primary-b");

        assert!(!cache.invalidate(8));
        assert!(matches!(cache.lookup(9), Lookup::Hit(&"primary-b")));
    }
}
