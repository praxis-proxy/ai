// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask lint-separators` — enforce that separator comments total
//! exactly 80 columns (indent + `// ` + dashes).

use clap::Parser;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Expected total width of separator lines (indent + `// ` + dashes).
const TARGET_WIDTH: usize = 80;

/// Minimum dash count before a line is treated as a separator.
const MIN_DASHES: usize = 20;

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask lint-separators`.
#[derive(Parser)]
pub(crate) struct Args;

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Scan all `.rs` files in the workspace for separator comments that do not
/// total exactly [`TARGET_WIDTH`] columns.
pub(crate) fn run(_args: Args) {
    let root = workspace_root();
    let violations = collect_violations(&root);

    if violations.is_empty() {
        println!("all separator comments are {TARGET_WIDTH} columns wide");
    } else {
        report_violations(&violations, &root);
        std::process::exit(1);
    }
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// A separator comment whose total width does not match [`TARGET_WIDTH`].
struct Violation {
    /// Path to the file containing the violation.
    path: std::path::PathBuf,

    /// 1-based line number of the separator.
    line: usize,

    /// Actual column width of the separator line.
    actual_width: usize,
}

/// Collect all separator-width violations across the workspace.
fn collect_violations(root: &std::path::Path) -> Vec<Violation> {
    let mut files = Vec::new();
    walk_rs_files(root, &mut files);
    files.sort();

    let mut violations = Vec::new();
    for path in &files {
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        for (line_num, line) in content.lines().enumerate() {
            if let Some(v) = check_separator(line, line_num + 1, path) {
                violations.push(v);
            }
        }
    }
    violations
}

/// Check whether a line is a separator comment with the wrong width.
fn check_separator(line: &str, line_num: usize, path: &std::path::Path) -> Option<Violation> {
    let trimmed = line.trim_start();
    let after_prefix = trimmed.strip_prefix("// ")?;

    if after_prefix.len() < MIN_DASHES {
        return None;
    }
    if !after_prefix.bytes().all(|b| b == b'-') {
        return None;
    }

    let actual_width = line.len();
    if actual_width != TARGET_WIDTH {
        return Some(Violation {
            path: path.to_owned(),
            line: line_num,
            actual_width,
        });
    }

    None
}

/// Print violation details to stderr.
fn report_violations(violations: &[Violation], root: &std::path::Path) {
    eprintln!(
        "{count} separator comment(s) with wrong width:",
        count = violations.len()
    );
    for v in violations {
        let rel = v.path.strip_prefix(root).unwrap_or(&v.path).display();
        eprintln!(
            "  {rel}:{line}: {actual} columns (expected {TARGET_WIDTH})",
            line = v.line,
            actual = v.actual_width,
        );
    }
    eprintln!(
        "\nseparator comments must be exactly {TARGET_WIDTH} columns: \
         indent + \"// \" + dashes. Adjust the dash count for the \
         indentation level."
    );
}

// -----------------------------------------------------------------------------
// File Collection
// -----------------------------------------------------------------------------

/// Recursively collect all `.rs` files under `dir`.
fn walk_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            walk_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Locate the workspace root directory.
fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_owned());
    std::path::Path::new(&manifest_dir)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    /// Dummy path for unit tests.
    fn dummy_path() -> std::path::PathBuf {
        std::path::PathBuf::from("test.rs")
    }

    #[test]
    fn top_level_77_dashes_passes() {
        let line = format!("// {}", "-".repeat(77));
        assert_eq!(line.len(), TARGET_WIDTH);
        assert!(check_separator(&line, 1, &dummy_path()).is_none());
    }

    #[test]
    fn indented_73_dashes_passes() {
        let line = format!("    // {}", "-".repeat(73));
        assert_eq!(line.len(), TARGET_WIDTH);
        assert!(check_separator(&line, 1, &dummy_path()).is_none());
    }

    #[test]
    fn indented_77_dashes_fails() {
        let line = format!("    // {}", "-".repeat(77));
        assert_eq!(line.len(), 84);
        let v = check_separator(&line, 1, &dummy_path());
        assert!(v.is_some(), "84-col separator should fail");
        assert_eq!(v.unwrap().actual_width, 84);
    }

    #[test]
    fn top_level_75_dashes_fails() {
        let line = format!("// {}", "-".repeat(75));
        assert_eq!(line.len(), 78);
        let v = check_separator(&line, 1, &dummy_path());
        assert!(v.is_some(), "78-col separator should fail");
        assert_eq!(v.unwrap().actual_width, 78);
    }

    #[test]
    fn non_separator_comment_ignored() {
        let line = "// this is a regular comment";
        assert!(check_separator(line, 1, &dummy_path()).is_none());
    }

    #[test]
    fn short_dash_line_ignored() {
        let line = "// ----------";
        assert!(
            check_separator(line, 1, &dummy_path()).is_none(),
            "lines with fewer than {MIN_DASHES} dashes should be ignored"
        );
    }

    #[test]
    fn mixed_content_ignored() {
        let line = "// ----------- Section Title -----------";
        assert!(
            check_separator(line, 1, &dummy_path()).is_none(),
            "lines with non-dash content after dashes should be ignored"
        );
    }

    #[test]
    fn code_line_ignored() {
        let line = "    let x = 42;";
        assert!(check_separator(line, 1, &dummy_path()).is_none());
    }

    #[test]
    fn eight_space_indent_passes() {
        let line = format!("        // {}", "-".repeat(69));
        assert_eq!(line.len(), TARGET_WIDTH);
        assert!(check_separator(&line, 1, &dummy_path()).is_none());
    }

    #[test]
    fn real_files_pass() {
        let root = workspace_root();
        let violations = collect_violations(&root);

        assert!(
            violations.is_empty(),
            "{count} separator violations found",
            count = violations.len()
        );
    }
}
