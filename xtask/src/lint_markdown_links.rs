// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask lint-markdown-links` checks local Markdown link targets.

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask lint-markdown-links`.
#[derive(Parser)]
pub(crate) struct Args;

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Check that relative links in tracked Markdown sources resolve on disk.
pub(crate) fn run(_args: Args) {
    let root = workspace_root();
    let files = tracked_markdown_files(&root);

    let mut violations = Vec::new();
    for path in &files {
        check_file(path, &root, &mut violations);
    }

    if violations.is_empty() {
        println!("all local links in {} Markdown files resolve", files.len());
        return;
    }

    eprintln!("Markdown links with missing local targets:");
    for violation in &violations {
        let path = violation.path.strip_prefix(&root).unwrap_or(&violation.path);
        eprintln!(
            "  {path}:{line}: {target}",
            path = path.display(),
            line = violation.line,
            target = violation.target,
        );
    }
    std::process::exit(1);
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// A Markdown link whose relative target does not exist.
struct Violation {
    /// Source file containing the link.
    path: PathBuf,

    /// One-based source line.
    line: usize,

    /// Link target as written.
    target: String,
}

/// Check one Markdown file and append missing-target violations.
fn check_file(path: &Path, root: &Path, violations: &mut Vec<Violation>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned());
    let parent = resolved.parent().unwrap_or(root);

    for (line, target) in extract_targets(&content) {
        let Some(local_target) = local_path(&target) else {
            continue;
        };
        if !parent.join(local_target).exists() {
            violations.push(Violation {
                line,
                path: path.to_owned(),
                target,
            });
        }
    }
}

/// Extract Markdown link destinations outside fenced code and HTML comments.
fn extract_targets(content: &str) -> Vec<(usize, String)> {
    let block_visible = mask_block_syntax(content);
    let visible = mask_inline_code(&block_visible);
    let mut targets = Vec::new();

    for (index, line) in visible.lines().enumerate() {
        let line_number = index + 1;
        extract_inline_targets(line, line_number, &mut targets);
        if let Some(target) = extract_reference_target(line.trim_start()) {
            targets.push((line_number, target));
        }
    }

    targets
}

/// Replace fenced code and HTML comments while preserving line boundaries.
fn mask_block_syntax(content: &str) -> String {
    let mut visible = String::with_capacity(content.len());
    let mut in_comment = false;
    let mut fence: Option<(u8, usize)> = None;
    let mut in_indented_code = false;
    let mut previous_blank = true;

    for line in content.split('\n') {
        let can_start_indented_code = previous_blank;
        previous_blank = line.trim().is_empty();
        if fence.is_some() && update_fence(line.trim_start(), &mut fence) {
            in_indented_code = false;
            push_masked_line(&mut visible, line);
            continue;
        }
        if in_comment {
            in_indented_code = false;
        } else if should_mask_indented_code(line, can_start_indented_code, &mut in_indented_code) {
            push_masked_line(&mut visible, line);
            continue;
        }
        let comment_visible = mask_html_comments(line, &mut in_comment);
        let trimmed = comment_visible.trim_start();
        if update_fence(trimmed, &mut fence) {
            push_masked_line(&mut visible, line);
            continue;
        }
        visible.push_str(&comment_visible);
        visible.push('\n');
    }

    visible
}

/// Append one fully masked line while retaining its line boundary.
fn push_masked_line(output: &mut String, line: &str) {
    output.push_str(&" ".repeat(line.len()));
    output.push('\n');
}

/// Whether a line uses Markdown's four-space or tab code indentation.
fn is_indented_code(line: &str) -> bool {
    line.starts_with('\t') || line.as_bytes().iter().take_while(|byte| **byte == b' ').count() >= 4
}

/// Update indented-code state and report whether the current line is code.
fn should_mask_indented_code(line: &str, can_start: bool, active: &mut bool) -> bool {
    if line.trim().is_empty() {
        return *active;
    }
    if is_indented_code(line) && (*active || can_start) {
        *active = true;
        return true;
    }
    *active = false;
    false
}

/// Update fenced-code state and report whether the current line is fenced.
fn update_fence(line: &str, fence: &mut Option<(u8, usize)>) -> bool {
    let Some(marker) = line.as_bytes().first().copied() else {
        return fence.is_some();
    };
    if !matches!(marker, b'`' | b'~') {
        return fence.is_some();
    }
    let length = line.as_bytes().iter().take_while(|byte| **byte == marker).count();

    if let Some((active_marker, active_length)) = fence {
        let suffix = line.get(length..).unwrap_or_default();
        if marker == *active_marker && length >= *active_length && suffix.trim().is_empty() {
            *fence = None;
        }
        return true;
    }

    if length >= 3 {
        *fence = Some((marker, length));
        return true;
    }
    false
}

/// Replace HTML comments with spaces while preserving visible text.
fn mask_html_comments<'a>(line: &'a str, in_comment: &mut bool) -> Cow<'a, str> {
    if !*in_comment && !line.contains("<!--") {
        return Cow::Borrowed(line);
    }

    let source = line.as_bytes();
    let mut masked = source.to_vec();
    let mut cursor = 0;
    while cursor < source.len() {
        if *in_comment {
            let Some(closing) = find_bytes(source, cursor, b"-->") else {
                if let Some(span) = masked.get_mut(cursor..) {
                    span.fill(b' ');
                }
                break;
            };
            let end = closing + 3;
            if let Some(span) = masked.get_mut(cursor..end) {
                span.fill(b' ');
            }
            cursor = end;
            *in_comment = false;
            continue;
        }

        let Some(opening) = find_bytes(source, cursor, b"<!--") else {
            break;
        };
        *in_comment = true;
        cursor = opening;
    }

    String::from_utf8(masked).map_or_else(|_| Cow::Borrowed(line), Cow::Owned)
}

/// Find an ASCII delimiter at or after a byte offset.
fn find_bytes(source: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    source
        .get(from..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| from + offset)
}

/// Extract inline `[label](target)` and `![alt](target)` destinations.
fn extract_inline_targets(line: &str, line_number: usize, targets: &mut Vec<(usize, String)>) {
    let mut remaining = line;
    while let Some(start) = remaining.find("](") {
        let Some(after_start) = remaining.get(start + 2..) else {
            break;
        };
        if find_link_opener(remaining, start).is_none() {
            remaining = after_start;
            continue;
        }
        let Some(end) = inline_destination_end(after_start) else {
            break;
        };
        let Some(raw) = after_start.get(..end) else {
            break;
        };
        if let Some(target) = destination(raw) {
            targets.push((line_number, target));
        }
        let Some(next) = after_start.get(end + 1..) else {
            break;
        };
        remaining = next;
    }
}

/// Find an unmatched, unescaped `[` before an inline link's closing bracket.
fn find_link_opener(value: &str, closing: usize) -> Option<usize> {
    let prefix = value.get(..closing)?;
    let mut openers = Vec::new();
    let mut escaped = false;
    for (index, character) in prefix.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '[' => openers.push(index),
            ']' => {
                openers.pop();
            },
            _ => {},
        }
    }
    openers.pop()
}

/// Replace inline code spans with spaces while preserving line boundaries.
fn mask_inline_code(content: &str) -> Cow<'_, str> {
    if !content.contains('`') {
        return Cow::Borrowed(content);
    }

    let source = content.as_bytes();
    let mut masked = source.to_vec();
    let mut cursor = 0;
    while cursor < source.len() {
        if source.get(cursor) != Some(&b'`') {
            cursor += 1;
            continue;
        }

        let opening = cursor;
        while source.get(cursor) == Some(&b'`') {
            cursor += 1;
        }
        let delimiter_length = cursor - opening;
        let Some(closing) = find_backtick_run(source, cursor, delimiter_length) else {
            continue;
        };
        let end = closing + delimiter_length;
        let Some(span) = masked.get_mut(opening..end) else {
            break;
        };
        mask_code_span(span);
        cursor = end;
    }

    String::from_utf8(masked).map_or_else(|_| Cow::Borrowed(content), Cow::Owned)
}

/// Mask one code span without changing document line boundaries.
fn mask_code_span(span: &mut [u8]) {
    for byte in span {
        if !matches!(*byte, b'\n' | b'\r') {
            *byte = b' ';
        }
    }
}

/// Find a backtick run whose length exactly matches the opening delimiter.
fn find_backtick_run(source: &[u8], mut cursor: usize, delimiter_length: usize) -> Option<usize> {
    while cursor < source.len() {
        if source.get(cursor) != Some(&b'`') {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while source.get(cursor) == Some(&b'`') {
            cursor += 1;
        }
        if cursor - start == delimiter_length {
            return Some(start);
        }
    }
    None
}

/// Find the closing delimiter while honoring balanced and escaped parentheses.
fn inline_destination_end(value: &str) -> Option<usize> {
    let mut depth = 0_u32;
    let mut escaped = false;

    for (index, character) in value.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '(' => depth += 1,
            ')' if depth == 0 => return Some(index),
            ')' => depth -= 1,
            _ => {},
        }
    }
    None
}

/// Extract a reference-style `[name]: target` destination.
fn extract_reference_target(line: &str) -> Option<String> {
    let rest = line.strip_prefix('[')?;
    if rest.starts_with('^') {
        return None;
    }
    let (_, raw) = rest.split_once("]: ").or_else(|| rest.split_once("]:"))?;
    destination(raw.trim_start())
}

/// Parse a Markdown destination, ignoring an optional title.
fn destination(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if let Some(angle) = trimmed.strip_prefix('<') {
        let end = angle.find('>')?;
        return angle.get(..end).map(unescape_destination);
    }
    trimmed.split_ascii_whitespace().next().map(unescape_destination)
}

/// Remove Markdown backslash escapes from punctuation in a destination.
fn unescape_destination(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\\' && characters.peek().is_some_and(char::is_ascii_punctuation) {
            result.push(characters.next().expect("peeked destination character"));
        } else {
            result.push(character);
        }
    }
    result
}

/// Return the filesystem portion of a relative Markdown target.
fn local_path(target: &str) -> Option<&str> {
    if target.starts_with('#') || target.starts_with('/') || has_uri_scheme(target) {
        return None;
    }

    let end = target.find(['#', '?']).unwrap_or(target.len());
    let path = target.get(..end)?;
    (!path.is_empty()).then_some(path)
}

/// Whether a target begins with an RFC 3986 URI scheme.
fn has_uri_scheme(target: &str) -> bool {
    let Some((scheme, _)) = target.split_once(':') else {
        return false;
    };
    let mut characters = scheme.chars();
    characters.next().is_some_and(|first| first.is_ascii_alphabetic())
        && characters.all(|character| character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.'))
}

// -----------------------------------------------------------------------------
// File Collection
// -----------------------------------------------------------------------------

/// Enumerate Markdown files in Git's index, including newly staged files.
/// Untracked drafts are intentionally excluded so unrelated local files do not
/// change `make lint`; stage new documentation before relying on this check.
fn tracked_markdown_files(root: &Path) -> Vec<PathBuf> {
    let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--cached", "-z", "--", "*.md"])
        .output()
    else {
        eprintln!("failed to run git ls-files for Markdown link checking");
        std::process::exit(1);
    };
    if !output.status.success() {
        eprintln!("git ls-files failed while collecting Markdown sources");
        std::process::exit(1);
    }

    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| root.join(String::from_utf8_lossy(path).as_ref()))
        .collect()
}

/// Locate the workspace root directory.
fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_owned());
    Path::new(&manifest_dir)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn extracts_inline_and_reference_targets() {
        let markdown = concat!(
            "[inline](docs/README.md)\n",
            "foo](missing.md) \\[escaped](missing.md) [valid](README.md)\n",
            "[outer [inner]](README.md)\n",
            "[parentheses](docs/spec_(draft).md)\n",
            "[escaped](docs/spec_\\(draft\\).md)\n",
            "[reference]: README.md#top\n",
            "[^note]: Explanatory text, not a link target.\n",
        );
        assert_eq!(
            extract_targets(markdown),
            vec![
                (1, "docs/README.md".to_owned()),
                (2, "README.md".to_owned()),
                (3, "README.md".to_owned()),
                (4, "docs/spec_(draft).md".to_owned()),
                (5, "docs/spec_(draft).md".to_owned()),
                (6, "README.md#top".to_owned()),
            ]
        );
    }

    #[test]
    fn skips_code_fences_and_html_comments() {
        let markdown = concat!(
            "    [indented code](missing.md)\n",
            "\t[tab code](missing.md)\n",
            "- item\n    [nested list link](README.md)\n",
            "````md\n```\n[long fence code](missing.md)\n````\n",
            "```md\n[code](missing.md)\n```\n",
            "<!--\n```md\n[comment](missing.md)\n-->\n",
            "`[inline code](missing.md)` [real](README.md)\n",
            "``[multi ` backtick](missing.md)``\n",
            "`\n[multiline code](missing.md)\n`\n",
            "[before](README.md) <!-- [hidden](missing.md) --> [after](docs/README.md)\n",
        );
        assert_eq!(
            extract_targets(markdown),
            vec![
                (4, "README.md".to_owned()),
                (16, "README.md".to_owned()),
                (21, "README.md".to_owned()),
                (21, "docs/README.md".to_owned()),
            ]
        );
    }

    #[test]
    fn classifies_relative_targets() {
        assert_eq!(local_path("docs/README.md#top"), Some("docs/README.md"));
        assert_eq!(local_path("../README.md?plain=1"), Some("../README.md"));
        assert_eq!(local_path("https://example.com"), None);
        assert_eq!(local_path("s3://example-bucket/object"), None);
        assert_eq!(local_path("vscode://file/example"), None);
        assert_eq!(local_path("urn:isbn:9780141036144"), None);
        assert_eq!(local_path("#section"), None);
    }
}
