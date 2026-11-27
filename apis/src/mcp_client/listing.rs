// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded `tools/list` pagination shared by every MCP discovery path.
//!
//! [`paginate_tools`] follows `nextCursor` until a page has none, and stops
//! with an error once the listing crosses [`MAX_PAGES`] pages, a tool-count
//! cap, or a cumulative byte cap. The caller supplies the step that fetches
//! one page, so the hosted connectors (an rmcp `Peer`) and the `mcp` broker
//! (its own session exchanges) share one set of bounds.

use std::future::Future;

use serde::Serialize;

use super::measure_tools_json_bytes;

/// Cap on pagination rounds, so a server returning empty pages with valid
/// cursors cannot loop forever.
pub const MAX_PAGES: usize = 100;

/// One page of a `tools/list` result.
#[derive(Debug)]
pub struct McpToolPage<T> {
    /// Tools on this page.
    pub tools: Vec<T>,
    /// Cursor for the next page; `None` on the last page.
    pub next_cursor: Option<String>,
}

/// Bounds for one complete `tools/list` listing.
#[derive(Clone, Copy, Debug)]
pub struct McpListingLimits {
    /// Most tools across all pages.
    pub max_tools: usize,
    /// Most serialized tool bytes across all pages.
    pub max_bytes: usize,
}

/// Why a paginated listing stopped before completing.
#[derive(Debug, Eq, PartialEq)]
pub enum McpListingError<E> {
    /// Fetching one page failed.
    Page(E),
    /// The listing holds more tools than allowed, or did not end within
    /// [`MAX_PAGES`] pages.
    TooManyTools {
        /// Tools collected when the listing stopped.
        count: usize,
        /// Configured tool cap.
        max: usize,
    },
    /// The serialized tools across pages exceed the byte cap.
    TooLarge {
        /// Bytes collected when the cap was crossed.
        bytes: usize,
        /// Configured byte cap.
        max: usize,
    },
}

/// Fetch a complete `tools/list` listing, one page at a time.
///
/// `fetch_page` receives the cursor to request (`None` for the first page).
/// Each page's tools count toward `limits` before the next page is
/// requested; a page body is bounded by the caller's transport before it is
/// decoded, and the byte cap stops a server from multiplying that bound over
/// many pages.
///
/// # Errors
///
/// Returns [`McpListingError::Page`] when a page cannot be fetched,
/// [`McpListingError::TooManyTools`] past the tool cap or [`MAX_PAGES`], and
/// [`McpListingError::TooLarge`] past the byte cap.
pub async fn paginate_tools<T, E, F, Fut>(
    limits: McpListingLimits,
    mut fetch_page: F,
) -> Result<Vec<T>, McpListingError<E>>
where
    T: Serialize,
    F: FnMut(Option<String>) -> Fut,
    Fut: Future<Output = Result<McpToolPage<T>, E>>,
{
    let mut all_tools = Vec::new();
    let mut total_bytes: usize = 0;
    let mut cursor = None;
    for _ in 0..MAX_PAGES {
        let page = fetch_page(cursor.take()).await.map_err(McpListingError::Page)?;
        total_bytes = total_bytes.saturating_add(measure_tools_json_bytes(&page.tools));
        if total_bytes > limits.max_bytes {
            return Err(McpListingError::TooLarge {
                bytes: total_bytes,
                max: limits.max_bytes,
            });
        }
        all_tools.extend(page.tools);
        if all_tools.len() > limits.max_tools {
            return Err(McpListingError::TooManyTools {
                count: all_tools.len(),
                max: limits.max_tools,
            });
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => return Ok(all_tools),
        }
    }
    Err(McpListingError::TooManyTools {
        count: all_tools.len(),
        max: limits.max_tools,
    })
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use std::sync::Mutex;

    use super::{MAX_PAGES, McpListingError, McpListingLimits, McpToolPage, paginate_tools};

    const LIMITS: McpListingLimits = McpListingLimits {
        max_tools: 16,
        max_bytes: 4_096,
    };

    /// Serve `pages` in order, recording each requested cursor.
    async fn run(
        limits: McpListingLimits,
        pages: Vec<Result<McpToolPage<String>, &'static str>>,
    ) -> (Result<Vec<String>, McpListingError<&'static str>>, Vec<Option<String>>) {
        let requested = Mutex::new(Vec::new());
        let pages = Mutex::new(pages.into_iter());
        let result = paginate_tools(limits, |cursor| {
            requested.lock().unwrap().push(cursor);
            let page = pages.lock().unwrap().next().unwrap();
            async move { page }
        })
        .await;
        (result, requested.into_inner().unwrap())
    }

    #[expect(clippy::unnecessary_wraps, reason = "pages are served as fetch results")]
    fn page(tools: &[&str], next: Option<&str>) -> Result<McpToolPage<String>, &'static str> {
        Ok(McpToolPage {
            tools: tools.iter().map(|t| (*t).to_owned()).collect(),
            next_cursor: next.map(str::to_owned),
        })
    }

    #[tokio::test]
    async fn follows_next_cursor_until_the_last_page() {
        let (result, requested) = run(LIMITS, vec![page(&["a"], Some("c1")), page(&["b", "c"], None)]).await;
        assert_eq!(result.unwrap(), ["a", "b", "c"]);
        assert_eq!(requested, [None, Some("c1".to_owned())]);
    }

    #[tokio::test]
    async fn page_failure_stops_the_listing() {
        let (result, requested) = run(LIMITS, vec![page(&["a"], Some("c1")), Err("down")]).await;
        assert_eq!(result.unwrap_err(), McpListingError::Page("down"));
        assert_eq!(requested.len(), 2);
    }

    #[tokio::test]
    async fn tool_cap_counts_across_pages() {
        let limits = McpListingLimits { max_tools: 2, ..LIMITS };
        let (result, _) = run(limits, vec![page(&["a", "b"], Some("c1")), page(&["c"], None)]).await;
        assert_eq!(result.unwrap_err(), McpListingError::TooManyTools { count: 3, max: 2 });
    }

    #[tokio::test]
    async fn byte_cap_counts_across_pages() {
        // Each `"xxxxxxxx"` page serializes to 12 bytes (`["xxxxxxxx"]`).
        let limits = McpListingLimits {
            max_bytes: 20,
            ..LIMITS
        };
        let (result, requested) = run(limits, vec![page(&["xxxxxxxx"], Some("c1")), page(&["xxxxxxxx"], None)]).await;
        assert_eq!(result.unwrap_err(), McpListingError::TooLarge { bytes: 24, max: 20 });
        assert_eq!(requested.len(), 2);
    }

    #[tokio::test]
    async fn endless_cursors_stop_at_the_page_cap() {
        let pages = (0..MAX_PAGES).map(|_| page(&[], Some("again"))).collect();
        let (result, requested) = run(LIMITS, pages).await;
        assert_eq!(result.unwrap_err(), McpListingError::TooManyTools { count: 0, max: 16 });
        assert_eq!(requested.len(), MAX_PAGES);
    }
}
