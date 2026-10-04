// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded model-context construction for file search results.

use std::collections::HashMap;

use serde_json::Value;

use super::client::SearchResult;
use crate::openai::responses::bounded_json_size;

/// Maximum content chunks rendered into one model-facing search context.
pub(super) const MAX_FORMATTED_CHUNKS: usize = 4_096;

/// Maximum model-facing context generated for one file-search call.
pub(super) const MAX_MODEL_CONTEXT_BYTES: usize = 10_485_760;

/// Maximum file ID length retained for citation provenance.
pub(super) const MAX_FILE_ID_BYTES: usize = 512;

/// Maximum filename length retained for citation provenance.
pub(super) const MAX_FILENAME_BYTES: usize = 1_024;

/// Fixed per-chunk format used by the private continuation bridge.
const ANNOTATION_TEMPLATE: &str = "[{index}] {filename} (score: {score}) cite as <|{file_id}|>\n{content}\n";

/// Fixed wrapper used by the private continuation bridge.
const CONTEXT_TEMPLATE: &str = "file_search found {num_chunks} chunks for \"{query}\":\n{results}";

/// Templates used to build model-facing file search context.
pub(super) struct FormatTemplates<'a> {
    /// Template rendered once for every returned content chunk.
    pub annotation: &'a str,
    /// Template wrapped around all rendered chunks.
    pub context: &'a str,
}

/// Internal continuation format; this is not part of the filter's YAML API.
pub(super) const MODEL_CONTEXT_TEMPLATES: FormatTemplates<'static> = FormatTemplates {
    annotation: ANNOTATION_TEMPLATE,
    context: CONTEXT_TEMPLATE,
};

/// Per-continuation limits and already-owned citation metadata.
pub(super) struct FormatLimits<'a> {
    /// Remaining compact-JSON string-content bytes for model-visible context.
    pub max_model_context_bytes: usize,

    /// Number of new file mappings that may be retained.
    pub max_new_citation_files: usize,

    /// Citation mappings accumulated before this result set.
    pub known_citation_files: &'a HashMap<String, String>,

    /// Citation mappings staged by earlier calls in the current transaction.
    pub staged_citation_files: &'a HashMap<String, String>,

    /// Whether canonical raw results must be materialized for the response.
    pub include_public_results: bool,
}

/// Separate bounded public and model-facing forms of one search result set.
pub(super) struct FormattedSearchResults {
    /// File metadata available for citation marker resolution.
    pub citation_files: HashMap<String, String>,
    /// Private context sent to the next inference round.
    pub model_context: String,
    /// Canonical raw results optionally exposed to the caller.
    pub public_results: Vec<Value>,
    /// Whether chunk or byte limits omitted any model-facing context.
    pub truncated: bool,
}

/// One brace-delimited placeholder parsed from a template.
struct TemplatePlaceholder<'a> {
    /// Placeholder name without braces.
    name: &'a str,
    /// Literal prefix before the placeholder.
    prefix: &'a str,
    /// Complete placeholder including braces.
    raw: &'a str,
    /// Template suffix after the placeholder.
    rest: &'a str,
}

/// Incremental bounded builder for private model context.
struct ModelContextBuilder {
    /// Number of chunks successfully rendered.
    chunk_count: usize,
    /// Whether a hard byte or chunk limit prevents further rendering.
    exhausted: bool,
    /// Maximum compact-JSON string-content bytes for the finished context.
    max_json_bytes: usize,
    /// Maximum escaped bytes available for rendered result annotations.
    max_rendered_json_bytes: usize,
    /// Rendered per-chunk annotations.
    rendered: String,
    /// Reused annotation buffer; one allocation can serve every result chunk.
    annotation: String,
    /// Width of the default wrapper's reserved count, or zero for a custom template.
    default_count_width: usize,
    /// Compact-JSON string-content bytes used by rendered annotations.
    rendered_json_bytes: usize,
    /// Whether any content exceeded a formatting bound.
    truncated: bool,
}

impl ModelContextBuilder {
    /// Create a builder after reserving escaped outer-template bytes.
    fn new(max_bytes: usize, query: &str, template: &str, max_chunk_count: usize) -> Self {
        let max_json_bytes = max_bytes.min(MAX_MODEL_CONTEXT_BYTES);
        let max_chunk_count = max_chunk_count.min(MAX_FORMATTED_CHUNKS).to_string();
        let wrapper = render_template_bounded(
            template,
            &[("query", query), ("num_chunks", &max_chunk_count), ("results", "")],
            max_json_bytes,
        );
        let result_placeholders = count_template_placeholders(template, "results");
        let wrapper_json_bytes = wrapper.as_deref().and_then(json_string_content_bytes);
        let wrapper_fits = wrapper_json_bytes.is_some_and(|bytes| bytes <= max_json_bytes);
        let max_rendered_json_bytes = wrapper_json_bytes
            .and_then(|bytes| max_json_bytes.checked_sub(bytes))
            .map_or(0, |bytes| bytes / result_placeholders.max(1));
        let default_wrapper = template == CONTEXT_TEMPLATE && wrapper_fits;
        Self {
            chunk_count: 0,
            exhausted: !wrapper_fits || result_placeholders == 0,
            max_json_bytes,
            max_rendered_json_bytes,
            rendered: if default_wrapper {
                wrapper.unwrap_or_default()
            } else {
                String::new()
            },
            annotation: String::new(),
            default_count_width: if default_wrapper { max_chunk_count.len() } else { 0 },
            rendered_json_bytes: 0,
            truncated: !wrapper_fits || result_placeholders == 0,
        }
    }

    /// Record that at least one result could not be represented safely.
    fn mark_truncated(&mut self) {
        self.truncated = true;
    }

    /// Mark the builder full and report that the current chunk was omitted.
    fn exhaust(&mut self) -> bool {
        self.exhausted = true;
        self.truncated = true;
        false
    }

    /// Append one result chunk when both formatting budgets permit it.
    #[expect(
        clippy::too_many_lines,
        reason = "check both encoded size limits before appending a chunk"
    )]
    fn append_chunk(&mut self, result: &SearchResult, content: &str, template: &str) -> bool {
        if self.exhausted || self.chunk_count >= MAX_FORMATTED_CHUNKS {
            return self.exhaust();
        }
        let next_index = self.chunk_count.saturating_add(1);
        let remaining_bytes = self.max_rendered_json_bytes.saturating_sub(self.rendered_json_bytes);
        self.annotation.clear();
        if render_annotation_bounded_into(
            &mut self.annotation,
            result,
            content,
            next_index,
            template,
            remaining_bytes,
        )
        .is_none()
        {
            return self.exhaust();
        }
        let Some(next_json_bytes) = json_string_content_bytes(&self.annotation)
            .and_then(|annotation_json_bytes| self.rendered_json_bytes.checked_add(annotation_json_bytes))
        else {
            return self.exhaust();
        };
        if next_json_bytes > self.max_rendered_json_bytes {
            return self.exhaust();
        }
        let rendered_file = !self.annotation.is_empty();
        self.rendered.push_str(&self.annotation);
        self.rendered_json_bytes = next_json_bytes;
        self.chunk_count = next_index;
        rendered_file
    }

    /// Apply the outer context template and report whether it overflowed.
    fn finish(mut self, query: &str, template: &str) -> (String, bool) {
        let chunk_count = self.chunk_count.to_string();
        if self.default_count_width > 0 {
            // `new` reserved the widest count possible for this result set.
            // Replacing it with the actual count can only shrink the wrapper. The
            // annotations stay in the same allocation instead of being copied
            // into a second complete model-context string.
            let count_start = "file_search found ".len();
            let count_end = count_start + self.default_count_width;
            self.rendered.replace_range(count_start..count_end, &chunk_count);
            return (self.rendered, self.truncated);
        }
        let rendered = render_template_bounded(
            template,
            &[
                ("query", query),
                ("num_chunks", &chunk_count),
                ("results", &self.rendered),
            ],
            self.max_json_bytes,
        );
        match rendered {
            Some(context) if json_string_content_bytes(&context).is_some_and(|bytes| bytes <= self.max_json_bytes) => {
                (context, self.truncated)
            },
            None => (String::new(), true),
            Some(_context) => (String::new(), true),
        }
    }
}

/// Count compact JSON bytes inside a serialized string's quote delimiters.
fn json_string_content_bytes(value: &str) -> Option<usize> {
    bounded_json_size(value, usize::MAX).ok().flatten()?.checked_sub(2)
}

/// Render a template by replacing known `{variable}` placeholders.
///
/// Unknown placeholders remain unchanged so future variables can be added
/// without making existing configurations fail to load.
#[cfg(test)]
pub(super) fn render_template(template: &str, variables: &[(&str, &str)]) -> String {
    render_template_bounded(template, variables, usize::MAX).unwrap_or_else(|| template.to_owned())
}

/// Count exact parsed placeholders using the same tokenization as rendering.
pub(super) fn count_template_placeholders(template: &str, name: &str) -> usize {
    let mut count = 0_usize;
    let mut remaining = template;
    while let Some(placeholder) = next_template_placeholder(remaining) {
        if placeholder.name == name {
            count = count.saturating_add(1);
        }
        remaining = placeholder.rest;
    }
    count
}

/// Parse the next complete brace-delimited placeholder.
fn next_template_placeholder(template: &str) -> Option<TemplatePlaceholder<'_>> {
    let open = template.find('{')?;
    let (prefix, candidate) = template.split_at(open);
    let close = candidate.find('}')?;
    let (raw, rest) = candidate.split_at(close.saturating_add(1));
    let name = raw.strip_prefix('{')?.strip_suffix('}')?;
    Some(TemplatePlaceholder {
        name,
        prefix,
        raw,
        rest,
    })
}

/// Render a template without allowing replacement values to exceed `max_bytes`.
fn render_template_bounded(template: &str, variables: &[(&str, &str)], max_bytes: usize) -> Option<String> {
    let mut rendered = String::with_capacity(template.len().min(max_bytes));
    render_template_bounded_into(&mut rendered, template, variables, max_bytes)?;
    Some(rendered)
}

/// Render into caller-owned storage so each chunk can reuse one allocation.
fn render_template_bounded_into(
    rendered: &mut String,
    template: &str,
    variables: &[(&str, &str)],
    max_bytes: usize,
) -> Option<()> {
    let mut remaining = template;
    while let Some(placeholder) = next_template_placeholder(remaining) {
        push_bounded(rendered, placeholder.prefix, max_bytes)?;
        if let Some(value) = variables
            .iter()
            .find_map(|(variable, value)| (*variable == placeholder.name).then_some(*value))
        {
            push_bounded(rendered, value, max_bytes)?;
        } else {
            push_bounded(rendered, placeholder.raw, max_bytes)?;
        }
        remaining = placeholder.rest;
    }
    push_bounded(rendered, remaining, max_bytes)?;
    Some(())
}

/// Append one string when it fits the remaining byte budget.
fn push_bounded(rendered: &mut String, value: &str, max_bytes: usize) -> Option<()> {
    let next_len = rendered.len().checked_add(value.len())?;
    if next_len > max_bytes {
        return None;
    }
    rendered.push_str(value);
    Some(())
}

/// Format ranked search results into separate public and model-facing forms.
///
/// Public results retain canonical raw chunk text. Templates are applied only
/// to the private model context so prompt instructions and citation markers do
/// not leak into `file_search_call.results`.
#[expect(
    clippy::too_many_lines,
    reason = "formats public, citation, and private projections in one pass"
)]
pub(super) fn format_search_results(
    results: &[SearchResult],
    query: &str,
    templates: &FormatTemplates<'_>,
    limits: &FormatLimits<'_>,
) -> FormattedSearchResults {
    let mut citation_files = HashMap::new();
    let mut public_results = Vec::with_capacity(results.len());
    let max_chunk_count = results
        .iter()
        .fold(0_usize, |total, result| total.saturating_add(result.content.len()))
        .min(MAX_FORMATTED_CHUNKS);
    let mut context = ModelContextBuilder::new(
        limits.max_model_context_bytes,
        query,
        templates.context,
        max_chunk_count,
    );
    let emits_citation_marker = templates.annotation.contains("<|{file_id}|>");

    for result in results {
        let citation_compatible = citation_metadata_compatible(
            result,
            limits.known_citation_files,
            limits.staged_citation_files,
            &citation_files,
            limits.max_new_citation_files,
        );
        let render_context = !emits_citation_marker || citation_compatible;
        if !render_context && !result.content.is_empty() {
            context.mark_truncated();
        }
        let (public_result, rendered_file) = format_result(
            result,
            templates.annotation,
            &mut context,
            render_context,
            limits.include_public_results,
        );
        if rendered_file
            && emits_citation_marker
            && !limits.known_citation_files.contains_key(&result.file_id)
            && !limits.staged_citation_files.contains_key(&result.file_id)
        {
            citation_files.insert(result.file_id.clone(), result.filename.clone());
        }
        public_results.extend(public_result);
    }

    let (model_context, truncated) = context.finish(query, templates.context);
    if truncated && model_context.is_empty() {
        citation_files.clear();
    }
    FormattedSearchResults {
        citation_files,
        model_context,
        public_results,
        truncated,
    }
}

/// Check one mapping against syntax, conflict, and capacity limits.
fn citation_metadata_compatible(
    result: &SearchResult,
    known_citation_files: &HashMap<String, String>,
    staged_citation_files: &HashMap<String, String>,
    new_citation_files: &HashMap<String, String>,
    max_new_citation_files: usize,
) -> bool {
    let metadata_compatible = result.file_id.len() <= MAX_FILE_ID_BYTES
        && is_valid_file_id(&result.file_id)
        && is_valid_filename(&result.filename);
    let known_compatible = known_citation_files
        .get(&result.file_id)
        .is_none_or(|filename| filename == &result.filename);
    let staged_compatible = staged_citation_files
        .get(&result.file_id)
        .is_none_or(|filename| filename == &result.filename);
    let local_compatible = new_citation_files
        .get(&result.file_id)
        .is_none_or(|filename| filename == &result.filename);
    let has_capacity = known_citation_files.contains_key(&result.file_id)
        || staged_citation_files.contains_key(&result.file_id)
        || new_citation_files.contains_key(&result.file_id)
        || new_citation_files.len() < max_new_citation_files;
    metadata_compatible && known_compatible && staged_compatible && local_compatible && has_capacity
}

/// Build one canonical public result while appending its private context.
fn format_result(
    result: &SearchResult,
    annotation_template: &str,
    context: &mut ModelContextBuilder,
    render_context: bool,
    include_public_result: bool,
) -> (Option<Value>, bool) {
    let mut raw_text = include_public_result.then(String::new);
    let mut rendered_file = false;
    for (index, chunk) in result.content.iter().enumerate() {
        if let Some(raw_text) = &mut raw_text {
            if index > 0 {
                raw_text.push('\n');
            }
            raw_text.push_str(&chunk.text);
        }
        if render_context {
            rendered_file |= context.append_chunk(result, &chunk.text, annotation_template);
        }
    }
    (
        raw_text.map(|raw_text| {
            serde_json::json!({
                "attributes": result.attributes,
                "file_id": result.file_id,
                "filename": result.filename,
                "score": result.score,
                "text": raw_text,
            })
        }),
        rendered_file,
    )
}

/// Render one annotation without exceeding its remaining context budget.
#[cfg(test)]
fn render_annotation_bounded(
    result: &SearchResult,
    content: &str,
    index: usize,
    template: &str,
    max_bytes: usize,
) -> Option<String> {
    let mut rendered = String::new();
    render_annotation_bounded_into(&mut rendered, result, content, index, template, max_bytes)?;
    Some(rendered)
}

/// Render one annotation into the reusable chunk buffer.
#[expect(
    clippy::too_many_arguments,
    reason = "reuse caller-owned storage with the bounded template fields"
)]
fn render_annotation_bounded_into(
    rendered: &mut String,
    result: &SearchResult,
    content: &str,
    index: usize,
    template: &str,
    max_bytes: usize,
) -> Option<()> {
    let index = index.to_string();
    let score = result.score.to_string();
    render_template_bounded_into(
        rendered,
        template,
        &[
            ("index", &index),
            ("file_id", &result.file_id),
            ("filename", &result.filename),
            ("score", &score),
            ("content", content),
        ],
        max_bytes,
    )
}

/// Return whether an identifier matches `file-[A-Za-z0-9_-]+`.
pub(super) fn is_valid_file_id(file_id: &str) -> bool {
    file_id.strip_prefix("file-").is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    })
}

/// Return whether a filename is bounded and safe to persist in citation metadata.
pub(super) fn is_valid_filename(filename: &str) -> bool {
    !filename.trim().is_empty()
        && filename.len() <= MAX_FILENAME_BYTES
        && filename.chars().all(|character| !character.is_control())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::indexing_slicing, clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;
    use crate::openai::responses::file_search_callout::client::{ContentChunk, ContentChunkType};

    /// Fixed pre-refactor path: allocate one annotation per chunk, then copy
    /// the complete rendered set into the wrapper at the end.
    fn legacy_default_context(result: &SearchResult) -> String {
        let mut rendered = String::new();
        for (index, chunk) in result.content.iter().enumerate() {
            let annotation = render_annotation_bounded(
                result,
                &chunk.text,
                index + 1,
                ANNOTATION_TEMPLATE,
                MAX_MODEL_CONTEXT_BYTES,
            )
            .unwrap();
            rendered.push_str(&annotation);
        }
        render_template_bounded(
            CONTEXT_TEMPLATE,
            &[
                ("query", "allocation fixture"),
                ("num_chunks", &result.content.len().to_string()),
                ("results", &rendered),
            ],
            MAX_MODEL_CONTEXT_BYTES,
        )
        .unwrap()
    }

    #[test]
    #[expect(clippy::print_stderr, reason = "record the fixed-baseline allocation result")]
    #[expect(
        clippy::too_many_lines,
        reason = "compare a full high-cardinality context with its fixed baseline"
    )]
    fn large_citation_context_allocation_reuses_chunk_storage_and_moves_final_text() {
        // Reproducible high-cardinality input: 2048 citations, 256 ASCII
        // content bytes each, one file. The legacy fixture above models the
        // old per-chunk String and full-context final copy.
        let result = SearchResult {
            attributes: None,
            content: (0..2048)
                .map(|_| ContentChunk {
                    _chunk_type: ContentChunkType::Text,
                    text: "x".repeat(256),
                })
                .collect(),
            file_id: "file-allocation".to_owned(),
            filename: "allocation.txt".to_owned(),
            score: 0.75,
        };
        let optimized_context = || {
            let mut context = ModelContextBuilder::new(
                MAX_MODEL_CONTEXT_BYTES,
                "allocation fixture",
                CONTEXT_TEMPLATE,
                result.content.len(),
            );
            for chunk in &result.content {
                assert!(
                    context.append_chunk(&result, &chunk.text, ANNOTATION_TEMPLATE),
                    "fixture chunks must fit the context limit"
                );
            }
            context.finish("allocation fixture", CONTEXT_TEMPLATE).0
        };
        assert_eq!(optimized_context(), legacy_default_context(&result));

        let optimized = allocation_counter::measure(|| {
            std::hint::black_box(optimized_context());
        });
        let legacy = allocation_counter::measure(|| {
            std::hint::black_box(legacy_default_context(&result));
        });
        eprintln!("hosted citation context allocation fixture: optimized={optimized:?}, legacy={legacy:?}");
        assert!(
            optimized.count_total < legacy.count_total,
            "reused annotation must remove per-chunk allocation"
        );
        assert!(
            optimized.bytes_max < legacy.bytes_max,
            "in-place wrapper must remove the full-context copy"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "check both sides of the exact escaped JSON boundary"
    )]
    fn default_context_accepts_its_exact_json_boundary() {
        let result = SearchResult {
            attributes: None,
            content: vec![ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: "quoted \"source\"".to_owned(),
            }],
            file_id: "file-boundary".to_owned(),
            filename: "boundary.txt".to_owned(),
            score: 0.9,
        };
        let results = [result];
        let known = HashMap::new();
        let formatted_at = |limit| {
            format_search_results(
                &results,
                "boundary query",
                &MODEL_CONTEXT_TEMPLATES,
                &FormatLimits {
                    max_model_context_bytes: limit,
                    max_new_citation_files: 1,
                    known_citation_files: &known,
                    staged_citation_files: &known,
                    include_public_results: false,
                },
            )
        };
        let full = formatted_at(MAX_MODEL_CONTEXT_BYTES);
        let exact = json_string_content_bytes(&full.model_context).unwrap();
        let below = formatted_at(exact - 1);
        let at = formatted_at(exact);
        let above = formatted_at(exact + 1);
        assert!(below.truncated, "one byte below the rendered size must truncate");
        assert!(!at.truncated, "the exact rendered size must fit");
        assert_eq!(at.model_context, full.model_context);
        assert_eq!(above.model_context, full.model_context);
        assert_eq!(at.citation_files.get("file-boundary"), Some(&"boundary.txt".to_owned()));
    }

    #[test]
    fn default_context_rejects_escaped_wrapper_above_limit() {
        let query = "\n".repeat(64);
        let raw_wrapper = render_template(
            CONTEXT_TEMPLATE,
            &[("query", &query), ("num_chunks", "0"), ("results", "")],
        );
        let escaped_bytes = json_string_content_bytes(&raw_wrapper).unwrap();
        assert!(
            raw_wrapper.len() < escaped_bytes,
            "escaped newlines must cost more than their raw bytes"
        );

        let context = ModelContextBuilder::new(escaped_bytes - 1, &query, CONTEXT_TEMPLATE, 0);
        let (rendered, truncated) = context.finish(&query, CONTEXT_TEMPLATE);
        assert!(truncated, "an escaped wrapper above the limit must be rejected");
        assert!(rendered.is_empty(), "no over-limit context may be committed");
    }

    #[test]
    fn template_values_are_not_reinterpreted_as_placeholders() {
        let rendered = render_template(
            "query={query}; results={results}",
            &[("query", "{results}"), ("results", "actual")],
        );
        assert_eq!(rendered, "query={results}; results=actual");
    }

    #[test]
    fn placeholder_count_matches_renderer_tokenization() {
        assert_eq!(count_template_placeholders("{{results}}", "results"), 0);
        assert_eq!(count_template_placeholders("before {results} after", "results"), 1);
        assert_eq!(render_template("{{results}}", &[("results", "actual")]), "{{results}}");
    }

    #[test]

    fn bounded_template_renderer_rejects_oversized_expansion() {
        assert_eq!(
            render_template_bounded("{value}", &[("value", "1234")], 4),
            Some("1234".to_owned())
        );
        assert!(render_template_bounded("{value}", &[("value", "12345")], 4).is_none());
    }

    #[test]

    fn citation_metadata_limits_block_only_marker_templates() {
        let results = [SearchResult {
            attributes: None,
            content: vec![ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: "search text".to_owned(),
            }],
            file_id: "file-a".to_owned(),
            filename: "x".repeat(MAX_FILENAME_BYTES + 1),
            score: 0.9,
        }];
        let limits = FormatLimits {
            max_model_context_bytes: MAX_MODEL_CONTEXT_BYTES,
            max_new_citation_files: 0,
            known_citation_files: &HashMap::new(),
            staged_citation_files: &HashMap::new(),
            include_public_results: true,
        };

        let marked = format_search_results(
            &results,
            "query",
            &FormatTemplates {
                annotation: "<|{file_id}|>{content}",
                context: "{results}",
            },
            &limits,
        );

        assert!(marked.truncated);
        assert!(marked.model_context.is_empty());
        assert!(marked.citation_files.is_empty());
        assert_eq!(marked.public_results.len(), 1);
    }

    #[test]
    fn citation_free_templates_ignore_mapping_capacity() {
        let results = [SearchResult {
            attributes: None,
            content: vec![ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: "search text".to_owned(),
            }],
            file_id: "not-an-openai-file-id".to_owned(),
            filename: "x".repeat(MAX_FILENAME_BYTES + 1),
            score: 0.9,
        }];

        let formatted = format_search_results(
            &results,
            "query",
            &FormatTemplates {
                annotation: "{content}",
                context: "{results}",
            },
            &FormatLimits {
                max_model_context_bytes: MAX_MODEL_CONTEXT_BYTES,
                max_new_citation_files: 0,
                known_citation_files: &HashMap::new(),
                staged_citation_files: &HashMap::new(),
                include_public_results: true,
            },
        );

        assert!(!formatted.truncated);
        assert_eq!(formatted.model_context, "search text");
        assert!(formatted.citation_files.is_empty());
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "compares two complete escaped wrapper projections")]
    fn model_context_budget_counts_wrapper_and_json_escaping_once() {
        let make_result = |chunks: &[&str]| SearchResult {
            attributes: None,
            content: chunks
                .iter()
                .map(|text| ContentChunk {
                    _chunk_type: ContentChunkType::Text,
                    text: (*text).to_owned(),
                })
                .collect(),
            file_id: "file-a".to_owned(),
            filename: "a.txt".to_owned(),
            score: 0.9,
        };
        let templates = FormatTemplates {
            annotation: "{content}",
            context: "Q:{results}",
        };
        let limits = FormatLimits {
            max_model_context_bytes: 10,
            max_new_citation_files: 0,
            known_citation_files: &HashMap::new(),
            staged_citation_files: &HashMap::new(),
            include_public_results: false,
        };

        let ascii = format_search_results(&[make_result(&["abcd", "efgh"])], "", &templates, &limits);
        let escaped = format_search_results(&[make_result(&["\0", "\0"])], "", &templates, &limits);

        assert_eq!(ascii.model_context, "Q:abcdefgh");
        assert!(!ascii.truncated);
        assert_eq!(escaped.model_context, "Q:\0");
        assert!(escaped.truncated);
        assert_eq!(json_string_content_bytes(&escaped.model_context), Some(8));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "covers invalid and later valid metadata in one sequence"
    )]
    fn invalid_citation_metadata_does_not_hide_later_valid_context() {
        let result = |file_id: &str, text: &str| SearchResult {
            attributes: None,
            content: vec![ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: text.to_owned(),
            }],
            file_id: file_id.to_owned(),
            filename: "source.txt".to_owned(),
            score: 0.9,
        };
        let results = [result("", "invalid"), result("file-valid", "valid")];

        let formatted = format_search_results(
            &results,
            "query",
            &FormatTemplates {
                annotation: "<|{file_id}|>{content}",
                context: "{results}",
            },
            &FormatLimits {
                max_model_context_bytes: MAX_MODEL_CONTEXT_BYTES,
                max_new_citation_files: 1,
                known_citation_files: &HashMap::new(),
                staged_citation_files: &HashMap::new(),
                include_public_results: false,
            },
        );

        assert!(formatted.truncated);
        assert_eq!(formatted.model_context, "<|file-valid|>valid");
        assert_eq!(
            formatted.citation_files.get("file-valid").map(String::as_str),
            Some("source.txt")
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "constructs an over-limit chunk sequence")]
    fn model_context_caps_content_chunk_count() {
        let content = (0..=MAX_FORMATTED_CHUNKS)
            .map(|_| ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: "x".to_owned(),
            })
            .collect();
        let results = [SearchResult {
            attributes: None,
            content,
            file_id: "file-a".to_owned(),
            filename: "a.txt".to_owned(),
            score: 0.9,
        }];
        let templates = FormatTemplates {
            annotation: "{content}",
            context: "{results}",
        };

        let formatted = format_search_results(
            &results,
            "query",
            &templates,
            &FormatLimits {
                max_model_context_bytes: MAX_MODEL_CONTEXT_BYTES,
                max_new_citation_files: crate::openai::responses::state::MAX_CITATION_FILES,
                known_citation_files: &HashMap::new(),
                staged_citation_files: &HashMap::new(),
                include_public_results: true,
            },
        );

        assert!(formatted.truncated, "excess chunks must mark formatting incomplete");
        assert_eq!(formatted.model_context.len(), MAX_FORMATTED_CHUNKS);
        assert!(formatted.model_context.len() <= MAX_MODEL_CONTEXT_BYTES);
        assert_eq!(formatted.public_results.len(), 1, "public results remain canonical");
        assert!(formatted.citation_files.is_empty());
    }
}
