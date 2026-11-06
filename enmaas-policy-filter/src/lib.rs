#![allow(
    clippy::missing_docs_in_private_items,
    clippy::doc_markdown,
    reason = "private filter internals"
)]

//! EnMaaS request authorization and QoS policy filter.

mod buckets;
mod config;
mod events;
mod filter;
mod keys;
mod policy;

#[cfg(test)]
mod tests;

use praxis_filter::export_filters;

export_filters! {
    http "enmaas_policy" => filter::EnmaasPolicyFilter::from_config,
}
