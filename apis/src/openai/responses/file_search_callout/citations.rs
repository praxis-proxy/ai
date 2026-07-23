// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Result formatting and citation extraction for file search.

use std::{collections::HashMap, fmt};

use serde_json::Value;

use super::client::SearchResult;
use crate::openai::responses::bounded_json_size;

/// Maximum marker candidates processed in one output-text part.
const MAX_CITATION_MARKERS: usize = 4_096;

/// Maximum content chunks rendered into one model-facing search context.
pub(super) const MAX_FORMATTED_CHUNKS: usize = 4_096;

/// Maximum model-facing context generated for one file-search call.
pub(super) const MAX_MODEL_CONTEXT_BYTES: usize = 10_485_760;

/// Maximum existing plus generated annotations processed in one response.
const MAX_CITATION_ANNOTATIONS: usize = 2_048;

/// Maximum file ID length copied into an annotation.
const MAX_FILE_ID_BYTES: usize = 512;

/// Maximum filename length copied into an annotation.
const MAX_FILENAME_BYTES: usize = 1_024;

/// Maximum complete citation marker length, including delimiters.
const MAX_CITATION_MARKER_BYTES: usize = MAX_FILE_ID_BYTES + "<||>".len();

/// A bounded citation rewrite could not be completed.
#[derive(Debug)]
pub(crate) struct CitationRewriteError {
    /// Name of the exhausted budget.
    budget: &'static str,

    /// Maximum entries allowed by that budget.
    limit: usize,
}

impl fmt::Display for CitationRewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "citation {} count exceeds {} limit", self.budget, self.limit)
    }
}

impl std::error::Error for CitationRewriteError {}

/// Character range removed from the original text.
#[derive(Clone, Copy)]
struct RemovedRange {
    /// Inclusive original character offset.
    start: usize,
    /// Exclusive original character offset.
    end: usize,
}

/// Bounded citation extraction output.
struct CitationExtraction {
    /// New file citation annotations.
    annotations: Vec<Value>,
    /// Cleaned output text.
    cleaned: String,
    /// Original character ranges removed from the text.
    removals: Vec<RemovedRange>,
}

/// Response-wide allocation and work budget for citation rewriting.
struct CitationBudget {
    /// Existing and generated annotations still allowed.
    annotations_remaining: usize,
    /// Marker candidates still allowed.
    markers_remaining: usize,
}

impl Default for CitationBudget {
    fn default() -> Self {
        Self {
            annotations_remaining: MAX_CITATION_ANNOTATIONS,
            markers_remaining: MAX_CITATION_MARKERS,
        }
    }
}

impl CitationBudget {
    /// Consume one entry from a named budget.
    fn consume(remaining: &mut usize, budget: &'static str, limit: usize) -> Result<(), CitationRewriteError> {
        *remaining = remaining.checked_sub(1).ok_or(CitationRewriteError { budget, limit })?;
        Ok(())
    }

    /// Account for one marker candidate.
    fn consume_marker(&mut self) -> Result<(), CitationRewriteError> {
        Self::consume(&mut self.markers_remaining, "marker", MAX_CITATION_MARKERS)
    }

    /// Account for one existing or generated annotation.
    fn consume_annotation(&mut self) -> Result<(), CitationRewriteError> {
        Self::consume(&mut self.annotations_remaining, "annotation", MAX_CITATION_ANNOTATIONS)
    }
}

/// Templates used to build model-facing file search context.
pub(super) struct FormatTemplates<'a> {
    /// Template rendered once for every returned content chunk.
    pub annotation: &'a str,
    /// Template wrapped around all rendered chunks.
    pub context: &'a str,
}

/// Per-continuation limits and already-owned citation metadata.
pub(super) struct FormatLimits<'a> {
    /// Remaining compact-JSON string-content bytes for model-visible context.
    pub max_model_context_bytes: usize,

    /// Number of new file mappings that may be retained.
    pub max_new_citation_files: usize,

    /// Citation mappings accumulated before this result set.
    pub known_citation_files: &'a HashMap<String, String>,

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
    /// Compact-JSON string-content bytes used by rendered annotations.
    rendered_json_bytes: usize,
    /// Whether any content exceeded a formatting bound.
    truncated: bool,
}

impl ModelContextBuilder {
    /// Create a builder after reserving escaped outer-template bytes.
    fn new(max_bytes: usize, query: &str, template: &str) -> Self {
        let max_json_bytes = max_bytes.min(MAX_MODEL_CONTEXT_BYTES);
        let max_chunk_count = MAX_FORMATTED_CHUNKS.to_string();
        let wrapper = render_template_bounded(
            template,
            &[("query", query), ("num_chunks", &max_chunk_count), ("results", "")],
            max_json_bytes,
        );
        let result_placeholders = count_template_placeholders(template, "results");
        let wrapper_json_bytes = wrapper.as_deref().and_then(json_string_content_bytes);
        let max_rendered_json_bytes = wrapper_json_bytes
            .and_then(|bytes| max_json_bytes.checked_sub(bytes))
            .map_or(0, |bytes| bytes / result_placeholders.max(1));
        Self {
            chunk_count: 0,
            exhausted: wrapper_json_bytes.is_none() || result_placeholders == 0,
            max_json_bytes,
            max_rendered_json_bytes,
            rendered: String::new(),
            rendered_json_bytes: 0,
            truncated: wrapper_json_bytes.is_none() || result_placeholders == 0,
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
    fn append_chunk(&mut self, result: &SearchResult, content: &str, template: &str) -> bool {
        if self.exhausted || self.chunk_count >= MAX_FORMATTED_CHUNKS {
            return self.exhaust();
        }
        let next_index = self.chunk_count.saturating_add(1);
        let remaining_bytes = self.max_rendered_json_bytes.saturating_sub(self.rendered_json_bytes);
        let Some(annotation) = render_annotation_bounded(result, content, next_index, template, remaining_bytes) else {
            return self.exhaust();
        };
        let Some(annotation_json_bytes) = json_string_content_bytes(&annotation) else {
            return self.exhaust();
        };
        let Some(next_json_bytes) = self.rendered_json_bytes.checked_add(annotation_json_bytes) else {
            return self.exhaust();
        };
        if next_json_bytes > self.max_rendered_json_bytes {
            return self.exhaust();
        }
        let rendered_file = !annotation.is_empty();
        self.rendered.push_str(&annotation);
        self.rendered_json_bytes = next_json_bytes;
        self.chunk_count = next_index;
        rendered_file
    }

    /// Apply the outer context template and report whether it overflowed.
    fn finish(self, query: &str, template: &str) -> (String, bool) {
        let chunk_count = self.chunk_count.to_string();
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
    let mut remaining = template;
    while let Some(placeholder) = next_template_placeholder(remaining) {
        push_bounded(&mut rendered, placeholder.prefix, max_bytes)?;
        if let Some(value) = variables
            .iter()
            .find_map(|(variable, value)| (*variable == placeholder.name).then_some(*value))
        {
            push_bounded(&mut rendered, value, max_bytes)?;
        } else {
            push_bounded(&mut rendered, placeholder.raw, max_bytes)?;
        }
        remaining = placeholder.rest;
    }
    push_bounded(&mut rendered, remaining, max_bytes)?;
    Some(rendered)
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
    let mut context = ModelContextBuilder::new(limits.max_model_context_bytes, query, templates.context);
    let emits_citation_marker = templates.annotation.contains("<|{file_id}|>");

    for result in results {
        let citation_compatible = citation_metadata_compatible(
            result,
            limits.known_citation_files,
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
        if rendered_file && emits_citation_marker && !limits.known_citation_files.contains_key(&result.file_id) {
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
    new_citation_files: &HashMap<String, String>,
    max_new_citation_files: usize,
) -> bool {
    let metadata_compatible = result.file_id.len() <= MAX_FILE_ID_BYTES
        && is_valid_file_id(&result.file_id)
        && is_valid_filename(&result.filename);
    let known_compatible = known_citation_files
        .get(&result.file_id)
        .is_none_or(|filename| filename == &result.filename);
    let local_compatible = new_citation_files
        .get(&result.file_id)
        .is_none_or(|filename| filename == &result.filename);
    let has_capacity = known_citation_files.contains_key(&result.file_id)
        || new_citation_files.contains_key(&result.file_id)
        || new_citation_files.len() < max_new_citation_files;
    metadata_compatible && known_compatible && local_compatible && has_capacity
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

/// Extract file markers, remove them from text, and build annotations.
///
/// Marker indices count Unicode scalar values in the cleaned text, matching
/// the provider behavior rather than using UTF-8 byte offsets. Syntactically
/// valid unknown markers are removed without producing an annotation.
#[cfg(test)]
fn extract_citations(text: &str, citation_files: &HashMap<String, String>) -> (String, Vec<Value>) {
    match extract_citations_bounded(text, citation_files, &mut CitationBudget::default()) {
        Ok(extraction) => (extraction.cleaned, extraction.annotations),
        Err(_error) => (text.to_owned(), Vec::new()),
    }
}

/// Extract citations while enforcing marker and annotation budgets.
#[expect(clippy::too_many_lines, reason = "bounded linear marker scanner")]
fn extract_citations_bounded(
    text: &str,
    citation_files: &HashMap<String, String>,
    budget: &mut CitationBudget,
) -> Result<CitationExtraction, CitationRewriteError> {
    let mut cleaned = String::with_capacity(text.len());
    let mut annotations = Vec::new();
    let mut removals = Vec::new();
    let mut remaining = text;
    let mut cleaned_chars = 0_usize;
    let mut original_chars = 0_usize;

    while let Some(marker_start) = remaining.find("<|file-") {
        budget.consume_marker()?;
        let (prefix, candidate) = remaining.split_at(marker_start);
        let prefix_chars = prefix.chars().count();
        let Some((file_id, after_marker)) = split_marker(candidate) else {
            cleaned.push_str(prefix);
            cleaned_chars = cleaned_chars.saturating_add(prefix_chars);
            let Some((rest, kept_chars)) = preserve_invalid_marker_prefix(candidate, &mut cleaned) else {
                cleaned.push_str(candidate);
                return Ok(CitationExtraction {
                    annotations,
                    cleaned,
                    removals,
                });
            };
            cleaned_chars = cleaned_chars.saturating_add(kept_chars);
            original_chars = original_chars.saturating_add(prefix_chars).saturating_add(kept_chars);
            remaining = rest;
            continue;
        };
        let marker_chars = candidate.len().saturating_sub(after_marker.len());
        let marker_chars = candidate.get(..marker_chars).map_or(0, |marker| marker.chars().count());

        if !is_valid_file_id(file_id) {
            cleaned.push_str(prefix);
            cleaned_chars = cleaned_chars.saturating_add(prefix_chars);
            let Some((rest, kept_chars)) = preserve_invalid_marker_prefix(candidate, &mut cleaned) else {
                cleaned.push_str(candidate);
                return Ok(CitationExtraction {
                    annotations,
                    cleaned,
                    removals,
                });
            };
            cleaned_chars = cleaned_chars.saturating_add(kept_chars);
            original_chars = original_chars.saturating_add(prefix_chars).saturating_add(kept_chars);
            remaining = rest;
            continue;
        }

        cleaned.push_str(prefix);
        cleaned_chars = cleaned_chars.saturating_add(prefix_chars);
        let marker_start = original_chars.saturating_add(prefix_chars);
        let removal_start = if cleaned.ends_with(' ') {
            cleaned.pop();
            cleaned_chars = cleaned_chars.saturating_sub(1);
            marker_start.saturating_sub(1)
        } else {
            marker_start
        };
        removals.push(RemovedRange {
            start: removal_start,
            end: marker_start.saturating_add(marker_chars),
        });
        record_valid_marker(file_id, citation_files, cleaned_chars, &mut annotations, budget)?;
        original_chars = original_chars.saturating_add(prefix_chars).saturating_add(marker_chars);
        remaining = after_marker;
    }

    cleaned.push_str(remaining);
    Ok(CitationExtraction {
        annotations,
        cleaned,
        removals,
    })
}

/// Retain enough of an invalid candidate to resume scanning after its prefix.
fn preserve_invalid_marker_prefix<'a>(candidate: &'a str, cleaned: &mut String) -> Option<(&'a str, usize)> {
    let keep_through = "<|file-".len();
    let (kept, rest) = candidate.get(..keep_through).zip(candidate.get(keep_through..))?;
    cleaned.push_str(kept);
    Some((rest, kept.chars().count()))
}

/// Append one resolved annotation when it fits the configured bounds.
fn record_valid_marker(
    file_id: &str,
    citation_files: &HashMap<String, String>,
    index: usize,
    annotations: &mut Vec<Value>,
    budget: &mut CitationBudget,
) -> Result<(), CitationRewriteError> {
    if file_id.len() <= MAX_FILE_ID_BYTES
        && let Some(filename) = citation_files.get(file_id)
        && filename.len() <= MAX_FILENAME_BYTES
    {
        budget.consume_annotation()?;
        annotations.push(serde_json::json!({
            "type": "file_citation",
            "file_id": file_id,
            "filename": filename,
            "index": index,
        }));
    }
    Ok(())
}

/// Replace citation markers in every assistant output-text part.
///
/// Returns whether the response was modified.
#[expect(
    dead_code,
    reason = "consumed when core continuation returns the final inference response"
)]
pub(crate) fn annotate_response(
    response: &mut Value,
    citation_files: &HashMap<String, String>,
) -> Result<bool, CitationRewriteError> {
    if citation_files.is_empty() {
        return Ok(false);
    }
    let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) else {
        return Ok(false);
    };
    annotate_output_items_with_budget(output, citation_files, &mut CitationBudget::default())
}

/// Replace citation markers in response output items.
#[cfg(test)]
fn annotate_output_items(
    output: &mut [Value],
    citation_files: &HashMap<String, String>,
) -> Result<bool, CitationRewriteError> {
    if citation_files.is_empty() {
        return Ok(false);
    }
    annotate_output_items_with_budget(output, citation_files, &mut CitationBudget::default())
}

/// Replace markers while sharing one response-wide allocation budget.
fn annotate_output_items_with_budget(
    output: &mut [Value],
    citation_files: &HashMap<String, String>,
    budget: &mut CitationBudget,
) -> Result<bool, CitationRewriteError> {
    let mut modified = false;
    for item in output {
        modified |= annotate_output_item(item, citation_files, budget)?;
    }
    Ok(modified)
}

/// Replace citation markers within one assistant message item.
fn annotate_output_item(
    item: &mut Value,
    citation_files: &HashMap<String, String>,
    budget: &mut CitationBudget,
) -> Result<bool, CitationRewriteError> {
    if item.get("type").and_then(Value::as_str) != Some("message")
        || item.get("role").and_then(Value::as_str) != Some("assistant")
    {
        return Ok(false);
    }
    let Some(content) = item.get_mut("content").and_then(Value::as_array_mut) else {
        return Ok(false);
    };
    let mut modified = false;
    for part in content {
        modified |= annotate_text_part(part, citation_files, budget)?;
    }
    Ok(modified)
}

/// Replace citation markers within one output-text part.
fn annotate_text_part(
    part: &mut Value,
    citation_files: &HashMap<String, String>,
    budget: &mut CitationBudget,
) -> Result<bool, CitationRewriteError> {
    if part.get("type").and_then(Value::as_str) != Some("output_text") {
        return Ok(false);
    }
    let Some(text) = part.get("text").and_then(Value::as_str) else {
        return Ok(false);
    };
    if !text.contains("<|file-") {
        return Ok(false);
    }
    let extraction = extract_citations_bounded(text, citation_files, budget)?;
    if extraction.cleaned == text {
        return Ok(false);
    }
    let Some(object) = part.as_object_mut() else {
        return Ok(false);
    };
    object.insert("text".to_owned(), Value::String(extraction.cleaned));
    merge_annotations(object, extraction.annotations, &extraction.removals, budget)?;
    if object.contains_key("logprobs") {
        object.insert("logprobs".to_owned(), Value::Array(Vec::new()));
    }
    Ok(true)
}

/// Merge generated annotations after remapping any existing offsets.
fn merge_annotations(
    object: &mut serde_json::Map<String, Value>,
    annotations: Vec<Value>,
    removals: &[RemovedRange],
    budget: &mut CitationBudget,
) -> Result<(), CitationRewriteError> {
    let existing = object
        .entry("annotations".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    if !existing.is_array() {
        *existing = Value::Array(Vec::new());
    }
    if let Value::Array(existing) = existing {
        for _annotation in existing.iter() {
            budget.consume_annotation()?;
        }
        remap_annotation_offsets(existing, removals);
        existing.extend(annotations);
    }
    Ok(())
}

/// Shift existing annotation offsets after marker removal.
fn remap_annotation_offsets(annotations: &mut [Value], removals: &[RemovedRange]) {
    for annotation in annotations {
        let Some(object) = annotation.as_object_mut() else {
            continue;
        };
        let fields: &[&str] = match object.get("type").and_then(Value::as_str) {
            Some("file_citation") => &["index"],
            Some("url_citation" | "container_file_citation") => &["start_index", "end_index"],
            _ => &[],
        };
        for field in fields {
            let Some(index) = object.get(*field).and_then(Value::as_u64) else {
                continue;
            };
            let index = usize::try_from(index).unwrap_or(usize::MAX);
            object.insert((*field).to_owned(), Value::from(remap_offset(index, removals)));
        }
    }
}

/// Map an original character offset into the cleaned text.
fn remap_offset(index: usize, removals: &[RemovedRange]) -> usize {
    let mut removed = 0_usize;
    for range in removals {
        if index >= range.end {
            removed = removed.saturating_add(range.end.saturating_sub(range.start));
        } else if index > range.start {
            return range.start.saturating_sub(removed);
        } else {
            break;
        }
    }
    index.saturating_sub(removed)
}

/// Render one annotation without exceeding its remaining context budget.
fn render_annotation_bounded(
    result: &SearchResult,
    content: &str,
    index: usize,
    template: &str,
    max_bytes: usize,
) -> Option<String> {
    let index = index.to_string();
    let score = result.score.to_string();
    render_template_bounded(
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
fn is_valid_file_id(file_id: &str) -> bool {
    file_id.strip_prefix("file-").is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    })
}

/// Return whether a filename is bounded and safe to persist in citation metadata.
fn is_valid_filename(filename: &str) -> bool {
    !filename.trim().is_empty()
        && filename.len() <= MAX_FILENAME_BYTES
        && filename.chars().all(|character| !character.is_control())
}

/// Split the first complete marker into its file ID and remaining text.
fn split_marker(candidate: &str) -> Option<(&str, &str)> {
    let delimiter = candidate
        .as_bytes()
        .windows(2)
        .take(MAX_CITATION_MARKER_BYTES.saturating_sub(1))
        .position(|bytes| bytes == b"|>")?;
    let marker_end = delimiter.saturating_add(2);
    let marker = candidate.get(..marker_end)?;
    let file_id = marker.strip_prefix("<|")?.strip_suffix("|>")?;
    Some((file_id, candidate.get(marker_end..)?))
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::indexing_slicing, clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::openai::responses::file_search_callout::client::{ContentChunk, ContentChunkType};

    #[test]
    fn extracts_provider_compatible_citations() {
        let text = "Start [not-a-file]. New source <|file-abc123|>. \
                    Other source <|file-def456|>? Repeat source <|file-abc123|>! No citation.";
        let files = HashMap::from([
            ("file-abc123".to_owned(), "doc1.pdf".to_owned()),
            ("file-def456".to_owned(), "doc2.txt".to_owned()),
        ]);

        let (cleaned, annotations) = extract_citations(text, &files);

        assert_eq!(
            cleaned,
            "Start [not-a-file]. New source. Other source? Repeat source! No citation."
        );
        assert_eq!(annotations.len(), 3);
        assert_eq!(
            annotations.first().and_then(|value| value.get("index")),
            Some(&json!(30))
        );
        assert_eq!(
            annotations.get(1).and_then(|value| value.get("index")),
            Some(&json!(44))
        );
        assert_eq!(
            annotations.get(2).and_then(|value| value.get("index")),
            Some(&json!(59))
        );
    }

    #[test]
    fn citation_index_counts_characters_not_utf8_bytes() {
        let files = HashMap::from([("file-a".to_owned(), "a.txt".to_owned())]);
        let (cleaned, annotations) = extract_citations("Café <|file-a|>.", &files);
        assert_eq!(cleaned, "Café.");
        assert_eq!(
            annotations.first().and_then(|value| value.get("index")),
            Some(&json!(4))
        );
    }

    #[test]
    fn unknown_valid_marker_is_removed_without_annotation() {
        let (cleaned, annotations) = extract_citations("Answer <|file-missing|>.", &HashMap::new());
        assert_eq!(cleaned, "Answer.");
        assert!(annotations.is_empty());
    }

    #[test]
    fn malformed_marker_is_preserved() {
        let (cleaned, annotations) = extract_citations("Keep <|file-bad.dot|> here", &HashMap::new());
        assert_eq!(cleaned, "Keep <|file-bad.dot|> here");
        assert!(annotations.is_empty());
    }

    #[test]
    fn malformed_existing_annotations_are_replaced_when_rewriting() {
        let files = HashMap::from([("file-a".to_owned(), "a.txt".to_owned())]);
        let mut output = vec![serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "Source <|file-a|>.",
                "annotations": "invalid"
            }]
        })];

        assert!(annotate_output_items(&mut output, &files).unwrap());
        assert_eq!(
            output.first().and_then(|item| item.pointer("/content/0/text")),
            Some(&json!("Source."))
        );
        assert_eq!(
            output
                .first()
                .and_then(|item| item.pointer("/content/0/annotations/0/file_id")),
            Some(&json!("file-a"))
        );
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
    fn overlong_marker_candidate_does_not_hide_later_valid_marker() {
        let malformed = format!("<|file-{}", "x".repeat(MAX_FILE_ID_BYTES));
        let text = format!("Keep {malformed} and cite <|file-a|>.");
        let files = HashMap::from([("file-a".to_owned(), "a.txt".to_owned())]);

        let (cleaned, annotations) = extract_citations(&text, &files);

        assert_eq!(cleaned, format!("Keep {malformed} and cite."));
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0]["file_id"], "file-a");
    }

    #[test]
    fn maximum_length_file_id_marker_is_accepted() {
        let file_id = format!("file-{}", "x".repeat(MAX_FILE_ID_BYTES - "file-".len()));
        let marker = format!("<|{file_id}|>");

        let parsed = split_marker(&marker);
        assert!(parsed.is_some(), "maximum-length marker should parse");
        let (parsed, remaining) = parsed.unwrap();

        assert_eq!(parsed, file_id);
        assert!(remaining.is_empty(), "the complete marker should have no suffix");
    }

    #[test]
    fn annotation_offsets_are_remapped_and_logprobs_cleared() {
        let files = HashMap::from([("file-a".to_owned(), "a.txt".to_owned())]);
        let mut output = vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "A <|file-a|> B link",
                "annotations": [{
                    "type": "url_citation",
                    "start_index": 15,
                    "end_index": 19,
                    "url": "https://example.com",
                    "title": "existing"
                }],
                "logprobs": [{"token":"stale"}]
            }]
        })];

        assert!(annotate_output_items(&mut output, &files).unwrap());
        let part = output.first().and_then(|item| item.pointer("/content/0")).unwrap();
        assert_eq!(part["text"], "A B link");
        assert_eq!(part["annotations"][0]["start_index"], 4);
        assert_eq!(part["annotations"][0]["end_index"], 8);
        assert_eq!(part["annotations"][1]["index"], 1);
        assert_eq!(part["logprobs"], json!([]));
    }

    #[test]
    fn empty_citation_map_leaves_literal_markers_unchanged() {
        let mut output = vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type":"output_text","text":"literal <|file-a|>"}]
        })];

        assert!(!annotate_output_items(&mut output, &HashMap::new()).unwrap());
        assert_eq!(
            output.first().and_then(|item| item.pointer("/content/0/text")),
            Some(&json!("literal <|file-a|>"))
        );
    }

    #[test]
    fn marker_budget_is_shared_across_response_parts() {
        let files = HashMap::from([("file-known".to_owned(), "known.txt".to_owned())]);
        let content: Vec<Value> = (0..=MAX_CITATION_MARKERS)
            .map(|_| json!({"type":"output_text","text":"x <|file-a|>"}))
            .collect();
        let mut output = vec![json!({
            "type": "message",
            "role": "assistant",
            "content": content
        })];

        let error = annotate_output_items(&mut output, &files).unwrap_err();
        assert!(error.to_string().contains("marker"));
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
    fn marker_removal_preserves_file_path_index_semantics() {
        let mut annotations = vec![
            json!({"type":"file_path","file_id":"file-generated","index":7}),
            json!({"type":"file_citation","file_id":"file-a","filename":"a.txt","index":15}),
            json!({"type":"url_citation","url":"https://example.com","title":"source","start_index":12,"end_index":18}),
        ];
        let removals = [RemovedRange { start: 2, end: 12 }];

        remap_annotation_offsets(&mut annotations, &removals);

        assert_eq!(annotations[0]["index"], 7, "file_path.index is a file-list index");
        assert_eq!(annotations[1]["index"], 5, "file citation text index must shift");
        assert_eq!(annotations[2]["start_index"], 2, "URL start offset must shift");
        assert_eq!(annotations[2]["end_index"], 8, "URL end offset must shift");
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
                include_public_results: true,
            },
        );

        assert!(!formatted.truncated);
        assert_eq!(formatted.model_context, "search text");
        assert!(formatted.citation_files.is_empty());
    }

    #[test]
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
