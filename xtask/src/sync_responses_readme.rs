// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask sync-responses-readme` generates a pipeline-overview
//! table in `apis/src/openai/responses/README.md` from the
//! `HttpFilter` implementations found in that directory.

use std::{
    collections::HashMap,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use quote::ToTokens as _;

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask sync-responses-readme`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Write the generated README instead of checking for drift.
    #[arg(long)]
    fix: bool,
}

// -----------------------------------------------------------------------------
// Data Types
// -----------------------------------------------------------------------------

/// Which pipeline hooks a filter implements.
#[expect(
    clippy::struct_excessive_bools,
    reason = "one bool per pipeline hook is the natural representation"
)]
struct PipelineHooks {
    /// `on_request` does real work (ctx parameter is used).
    on_request: bool,
    /// Filter implements `on_request_body`.
    on_request_body: bool,
    /// `on_response` does real work (ctx parameter is used).
    on_response: bool,
    /// Filter implements `on_response_body`.
    on_response_body: bool,
}

/// Extracted pipeline metadata for one filter.
struct FilterMeta {
    /// Filter name from `fn name()`.
    name: String,
    /// First sentence of the filter struct's doc comment.
    description: String,
    /// Active pipeline hooks.
    hooks: PipelineHooks,
    /// Request-side body access (`None`, `ReadOnly`, `ReadWrite`).
    request_body_access: String,
    /// Request-side body mode (`Stream`, `StreamBuffer`).
    request_body_mode: String,
    /// Response-side body access.
    response_body_access: String,
    /// Response-side body mode.
    response_body_mode: String,
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Verify or regenerate the pipeline-overview table.
pub(crate) fn run(args: &Args) {
    let root = workspace_root();
    let responses_dir = root.join("apis/src/openai/responses");
    let readme_path = responses_dir.join("README.md");

    let mut filters = discover_filters(&responses_dir);
    filters.sort_by(|a, b| a.name.cmp(&b.name));

    let content = render_readme(&filters);

    if args.fix {
        fs::write(&readme_path, &content).unwrap();
        println!(
            "wrote {}",
            readme_path.strip_prefix(&root).unwrap_or(&readme_path).display()
        );
    } else {
        let current = fs::read_to_string(&readme_path).unwrap_or_default();
        if current == content {
            println!("apis/src/openai/responses/README.md is up to date");
        } else {
            eprintln!("apis/src/openai/responses/README.md is stale");
            eprintln!("\nrun: cargo xtask sync-responses-readme --fix");
            std::process::exit(1);
        }
    }
}

// -----------------------------------------------------------------------------
// Discovery
// -----------------------------------------------------------------------------

/// Discover all filters under the responses directory.
fn discover_filters(responses_dir: &Path) -> Vec<FilterMeta> {
    let mut filters = Vec::new();

    let top_mod = responses_dir.join("mod.rs");
    if top_mod.is_file() {
        filters.extend(extract_filters_from_file(&top_mod));
    }

    let Ok(entries) = fs::read_dir(responses_dir) else {
        return filters;
    };
    let mut subdirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subdirs.sort();

    for subdir in &subdirs {
        for rs_file in collect_rs_files(subdir) {
            filters.extend(extract_filters_from_file(&rs_file));
        }
    }

    filters
}

/// Collect all `.rs` files under a directory (non-recursive, skips tests).
///
/// Only scans `dir` itself, not nested subdirectories.  This is
/// intentional: every `responses/` submodule keeps its filter in the
/// immediate directory (e.g. `store/mod.rs`), so a recursive walk
/// would only add noise from test helpers or internal modules.
fn collect_rs_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "rs")
                && !p.file_name().is_some_and(|n| n == "tests.rs" || n == "test.rs")
        })
        .collect();
    files.sort();
    files
}

/// Extract filter metadata from `impl HttpFilter` blocks in one file.
fn extract_filters_from_file(path: &Path) -> Vec<FilterMeta> {
    let Ok(source) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(file) = syn::parse_file(&source) else {
        return Vec::new();
    };

    let struct_docs = collect_struct_docs(&file);
    let mut filters = Vec::new();

    for item in &file.items {
        let syn::Item::Impl(imp) = item else {
            continue;
        };
        if let Some(meta) = try_extract_filter(imp, &struct_docs) {
            filters.push(meta);
        }
    }

    filters
}

/// Try to extract a [`FilterMeta`] from a single `impl` block.
fn try_extract_filter(imp: &syn::ItemImpl, struct_docs: &HashMap<String, String>) -> Option<FilterMeta> {
    if !is_http_filter_impl(imp) {
        return None;
    }

    let name = extract_filter_name(imp)?;
    let struct_name = impl_self_type_name(imp).unwrap_or_default();
    let description = struct_docs
        .get(&struct_name)
        .map(|doc| first_sentence(doc))
        .unwrap_or_default();

    let mut meta = new_filter_meta(name, description);
    for impl_item in &imp.items {
        let syn::ImplItem::Fn(method) = impl_item else {
            continue;
        };
        classify_method(method, &mut meta);
    }
    Some(meta)
}

/// Check if an impl block is `impl HttpFilter for ...`.
fn is_http_filter_impl(imp: &syn::ItemImpl) -> bool {
    imp.trait_
        .as_ref()
        .and_then(|(path, _)| path.segments.last())
        .is_some_and(|seg| seg.ident == "HttpFilter")
}

/// Create a [`FilterMeta`] with default (inactive) hook state.
fn new_filter_meta(name: String, description: String) -> FilterMeta {
    FilterMeta {
        name,
        description,
        hooks: PipelineHooks {
            on_request: false,
            on_request_body: false,
            on_response: false,
            on_response_body: false,
        },
        request_body_access: "None".to_owned(),
        request_body_mode: "Stream".to_owned(),
        response_body_access: "None".to_owned(),
        response_body_mode: "Stream".to_owned(),
    }
}

/// Classify one method from an `impl HttpFilter` block into the
/// appropriate [`FilterMeta`] field.
fn classify_method(method: &syn::ImplItemFn, meta: &mut FilterMeta) {
    match method.sig.ident.to_string().as_str() {
        "on_request" => meta.hooks.on_request = !has_unused_ctx_param(&method.sig),
        "on_request_body" => meta.hooks.on_request_body = true,
        "on_response" => meta.hooks.on_response = !has_unused_ctx_param(&method.sig),
        "on_response_body" => meta.hooks.on_response_body = true,
        "request_body_access" => meta.request_body_access = extract_body_access_value(&method.block),
        "request_body_mode" => meta.request_body_mode = extract_body_mode_value(&method.block),
        "response_body_access" => meta.response_body_access = extract_body_access_value(&method.block),
        "response_body_mode" => meta.response_body_mode = extract_body_mode_value(&method.block),
        _ => {},
    }
}

// -----------------------------------------------------------------------------
// AST Extraction Helpers
// -----------------------------------------------------------------------------

/// Collect doc comments for all structs in a file, keyed by struct name.
fn collect_struct_docs(file: &syn::File) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for item in &file.items {
        if let syn::Item::Struct(s) = item {
            let doc = extract_doc_comment(&s.attrs);
            if !doc.is_empty() {
                map.insert(s.ident.to_string(), doc);
            }
        }
    }
    map
}

/// Extract the self type name from an impl block (e.g. `Foo` from
/// `impl HttpFilter for Foo`).
fn impl_self_type_name(imp: &syn::ItemImpl) -> Option<String> {
    if let syn::Type::Path(type_path) = imp.self_ty.as_ref() {
        type_path.path.segments.last().map(|s| s.ident.to_string())
    } else {
        None
    }
}

/// Join `#[doc = "..."]` attributes into a single string.
fn extract_doc_comment(attrs: &[syn::Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s), ..
            }) = &nv.value
        {
            lines.push(s.value());
        }
    }
    lines
        .iter()
        .map(|l| l.strip_prefix(' ').unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

/// Extract the first sentence from a doc comment.
///
/// Stops at headings (`# ...`), code fences, or blank lines before
/// looking for sentence boundaries. Sentence end is a period
/// followed by end-of-string, or a period-space-uppercase sequence
/// (to avoid splitting on abbreviations like `e.g.`).
fn first_sentence(doc: &str) -> String {
    let prose: String = doc
        .trim_start()
        .lines()
        .map(str::trim)
        .take_while(|line| {
            !line.is_empty() && !line.starts_with('#') && !line.starts_with("```") && !line.starts_with("[`")
        })
        .collect::<Vec<_>>()
        .join(" ");

    let chars: Vec<char> = prose.chars().collect();
    for (i, &ch) in chars.iter().enumerate() {
        if ch != '.' {
            continue;
        }
        let is_end = match chars.get(i + 1) {
            None => true,
            Some(&' ') => chars.get(i + 2).is_some_and(char::is_ascii_uppercase),
            Some(_) => false,
        };
        if is_end {
            return chars.get(..=i).map(|s| s.iter().collect()).unwrap_or(prose);
        }
    }
    prose
}

/// Extract the filter name from `fn name(&self) -> &'static str { "..." }`.
fn extract_filter_name(imp: &syn::ItemImpl) -> Option<String> {
    imp.items.iter().find_map(|item| {
        let syn::ImplItem::Fn(method) = item else {
            return None;
        };
        if method.sig.ident != "name" {
            return None;
        }
        method.block.stmts.iter().find_map(|stmt| {
            if let syn::Stmt::Expr(expr, _) = stmt {
                extract_str_literal(expr)
            } else {
                None
            }
        })
    })
}

/// Extract a string literal value from an expression.
fn extract_str_literal(expr: &syn::Expr) -> Option<String> {
    if let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(lit_str),
        ..
    }) = expr
    {
        Some(lit_str.value())
    } else {
        None
    }
}

/// Check whether the `ctx` parameter (second after `&self`) is prefixed
/// with `_`, indicating the filter does not use the request context.
fn has_unused_ctx_param(sig: &syn::Signature) -> bool {
    sig.inputs.iter().nth(1).is_some_and(|arg| {
        if let syn::FnArg::Typed(pat_type) = arg {
            let pat_str = pat_type.pat.to_token_stream().to_string();
            pat_str.starts_with('_')
        } else {
            false
        }
    })
}

/// Extract a `BodyAccess` variant name from a method body.
fn extract_body_access_value(block: &syn::Block) -> String {
    let tokens = block.to_token_stream().to_string();
    if tokens.contains("ReadWrite") {
        "ReadWrite".to_owned()
    } else if tokens.contains("ReadOnly") {
        "ReadOnly".to_owned()
    } else {
        "None".to_owned()
    }
}

/// Extract a `BodyMode` variant name from a method body.
fn extract_body_mode_value(block: &syn::Block) -> String {
    let tokens = block.to_token_stream().to_string();
    if tokens.contains("StreamBuffer") {
        "StreamBuffer".to_owned()
    } else if tokens.contains("SizeLimit") {
        "SizeLimit".to_owned()
    } else {
        "Stream".to_owned()
    }
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------

/// Render the full README content.
fn render_readme(filters: &[FilterMeta]) -> String {
    let mut out = String::new();
    write_preamble(&mut out);
    write_description_list(&mut out, filters);
    write_pipeline_table(&mut out, filters);
    out
}

/// Write the markdown preamble.
fn write_preamble(out: &mut String) {
    writeln!(out, "# OpenAI Responses Filters").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Pipeline overview for filters under `apis/src/openai/responses/`.").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "<!-- This file is auto-generated by `cargo xtask sync-responses-readme`."
    )
    .unwrap();
    writeln!(out, "     Do not edit by hand. -->").unwrap();
}

/// Write the filter description list.
fn write_description_list(out: &mut String, filters: &[FilterMeta]) {
    writeln!(out).unwrap();
    for f in filters {
        if f.description.is_empty() {
            writeln!(out, "- **`{}`**", f.name).unwrap();
        } else {
            writeln!(out, "- **`{}`** — {}", f.name, f.description).unwrap();
        }
    }
}

/// Write the pipeline hooks table.
fn write_pipeline_table(out: &mut String, filters: &[FilterMeta]) {
    writeln!(out).unwrap();
    writeln!(out, "## Pipeline Hooks").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Body-phase columns show `Access / Mode` when the hook is implemented."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Filter | `on_request` | `on_request_body` | `on_response` | `on_response_body` |"
    )
    .unwrap();
    writeln!(
        out,
        "|--------|:------------:|:-----------------:|:--------------:|:------------------:|"
    )
    .unwrap();

    for f in filters {
        write_pipeline_row(out, f);
    }
}

/// Write one pipeline table row.
fn write_pipeline_row(out: &mut String, f: &FilterMeta) {
    let on_req = format_header_hook(f.hooks.on_request);
    let on_req_body = format_body_hook(f.hooks.on_request_body, &f.request_body_access, &f.request_body_mode);
    let on_resp = format_header_hook(f.hooks.on_response);
    let on_resp_body = format_body_hook(f.hooks.on_response_body, &f.response_body_access, &f.response_body_mode);
    writeln!(
        out,
        "| `{name}` | {on_req} | {on_req_body} | {on_resp} | {on_resp_body} |",
        name = f.name
    )
    .unwrap();
}

/// Format a header-phase hook cell (`on_request` / `on_response`).
fn format_header_hook(active: bool) -> &'static str {
    if active { "\u{2713}" } else { "\u{2014}" }
}

/// Format a body-phase hook cell showing access and mode when active.
fn format_body_hook(active: bool, access: &str, mode: &str) -> String {
    if !active {
        return "\u{2014}".to_owned();
    }
    format!("{access} / {mode}")
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Find the workspace root by looking for the top-level `Cargo.toml`.
fn workspace_root() -> PathBuf {
    let output = std::process::Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format=plain"])
        .output()
        .expect("failed to run cargo locate-project");
    let path = String::from_utf8(output.stdout).expect("non-utf8 path");
    PathBuf::from(path.trim())
        .parent()
        .expect("Cargo.toml has no parent")
        .to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_hook_active() {
        assert_eq!(format_header_hook(true), "\u{2713}");
    }

    #[test]
    fn header_hook_inactive() {
        assert_eq!(format_header_hook(false), "\u{2014}");
    }

    #[test]
    fn body_hook_inactive() {
        assert_eq!(format_body_hook(false, "ReadOnly", "StreamBuffer"), "\u{2014}");
    }

    #[test]
    fn body_hook_active() {
        assert_eq!(
            format_body_hook(true, "ReadOnly", "StreamBuffer"),
            "ReadOnly / StreamBuffer"
        );
    }

    #[test]
    fn first_sentence_period_then_uppercase() {
        assert_eq!(
            first_sentence("Rewrites the model field. More details here."),
            "Rewrites the model field."
        );
    }

    #[test]
    fn first_sentence_period_at_end() {
        assert_eq!(first_sentence("Rewrites the model field."), "Rewrites the model field.");
    }

    #[test]
    fn first_sentence_no_period() {
        assert_eq!(first_sentence("Rewrites the model field"), "Rewrites the model field");
    }

    #[test]
    fn first_sentence_skips_abbreviation() {
        assert_eq!(
            first_sentence("Converts e.g. vLLM input. Rest."),
            "Converts e.g. vLLM input."
        );
    }

    #[test]
    fn first_sentence_multiline() {
        assert_eq!(
            first_sentence("Converts parts\nfor backends. Next sentence."),
            "Converts parts for backends."
        );
    }

    #[test]
    fn first_sentence_stops_at_heading() {
        assert_eq!(
            first_sentence("Does stuff.\n\n# YAML\n\n```yaml\nfilter: foo\n```"),
            "Does stuff."
        );
    }

    fn parse_single_filter(source: &str) -> FilterMeta {
        let file = syn::parse_file(source).expect("synthetic source must parse");
        let struct_docs = collect_struct_docs(&file);
        file.items
            .iter()
            .filter_map(|item| {
                if let syn::Item::Impl(imp) = item {
                    Some(imp)
                } else {
                    None
                }
            })
            .find_map(|imp| try_extract_filter(imp, &struct_docs))
            .expect("should extract exactly one filter")
    }

    #[test]
    fn extract_filter_from_synthetic_impl() {
        let meta = parse_single_filter(
            r#"
            /// Does something cool.
            struct FakeFilter;
            impl HttpFilter for FakeFilter {
                fn name(&self) -> &'static str { "fake_filter" }
                async fn on_request(&self, ctx: &mut Ctx) -> Result<()> { Ok(()) }
                async fn on_request_body(&self, ctx: &mut BodyCtx) -> Result<()> { Ok(()) }
                fn request_body_access(&self) -> BodyAccess { BodyAccess::ReadWrite }
                fn request_body_mode(&self) -> BodyMode { BodyMode::StreamBuffer { max_bytes: 1024 } }
            }
        "#,
        );

        assert_eq!(meta.name, "fake_filter");
        assert_eq!(meta.description, "Does something cool.");
        assert!(meta.hooks.on_request);
        assert!(meta.hooks.on_request_body);
        assert!(!meta.hooks.on_response);
        assert!(!meta.hooks.on_response_body);
        assert_eq!(meta.request_body_access, "ReadWrite");
        assert_eq!(meta.request_body_mode, "StreamBuffer");
        assert_eq!(meta.response_body_access, "None");
        assert_eq!(meta.response_body_mode, "Stream");
    }

    #[test]
    fn extract_filter_unused_ctx_marks_inactive() {
        let meta = parse_single_filter(
            r#"
            struct InactiveFilter;
            impl HttpFilter for InactiveFilter {
                fn name(&self) -> &'static str { "inactive_filter" }
                async fn on_request(&self, _ctx: &mut Ctx) -> Result<()> { Ok(()) }
                async fn on_response(&self, _ctx: &mut Ctx) -> Result<()> { Ok(()) }
            }
        "#,
        );
        assert!(!meta.hooks.on_request, "underscore-prefixed ctx should be inactive");
        assert!(!meta.hooks.on_response, "underscore-prefixed ctx should be inactive");
    }

    #[test]
    fn body_access_read_write_matched_before_read_only() {
        let rw: syn::Block = syn::parse_str("{ BodyAccess::ReadWrite }").unwrap();
        assert_eq!(extract_body_access_value(&rw), "ReadWrite");

        let ro: syn::Block = syn::parse_str("{ BodyAccess::ReadOnly }").unwrap();
        assert_eq!(extract_body_access_value(&ro), "ReadOnly");

        let none: syn::Block = syn::parse_str("{ BodyAccess::None }").unwrap();
        assert_eq!(extract_body_access_value(&none), "None");
    }

    #[test]
    fn body_mode_stream_buffer_matched_before_stream() {
        let sb: syn::Block = syn::parse_str("{ BodyMode::StreamBuffer { max_bytes: 1024 } }").unwrap();
        assert_eq!(extract_body_mode_value(&sb), "StreamBuffer");

        let sl: syn::Block = syn::parse_str("{ BodyMode::SizeLimit { max_bytes: 512 } }").unwrap();
        assert_eq!(extract_body_mode_value(&sl), "SizeLimit");

        let stream: syn::Block = syn::parse_str("{ BodyMode::Stream }").unwrap();
        assert_eq!(extract_body_mode_value(&stream), "Stream");
    }
}
