// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Final-response citation extraction and OpenAI annotation rewriting.

use std::{collections::HashMap, fmt};

use serde_json::Value;

use super::model_context::{MAX_FILE_ID_BYTES, MAX_FILENAME_BYTES, is_valid_file_id};

/// Maximum marker candidates processed in one output-text part.
const MAX_CITATION_MARKERS: usize = 4_096;

/// Maximum existing plus generated annotations processed in one response.
const MAX_CITATION_ANNOTATIONS: usize = 2_048;

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
}
