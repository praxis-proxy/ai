// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Generate and lint per-filter documentation files.
//!
//! Parses Rust source files under `filters/src/` and `apis/src/` with
//! [`syn`] to extract config structs, field metadata, YAML examples,
//! and filter descriptions. Produces one markdown file per filter at
//! `docs/filters/{filter_name}.md` and a reference index at
//! `docs/filters/reference.md`.

use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use quote::ToTokens as _;

// ---------------------------------------------------------------------------
// CLI Arguments
// ---------------------------------------------------------------------------

/// CLI arguments for `cargo xtask generate-filter-docs`.
#[derive(Parser)]
pub(crate) struct GenerateArgs;

/// CLI arguments for `cargo xtask lint-filter-docs`.
#[derive(Parser)]
pub(crate) struct LintArgs;

// ---------------------------------------------------------------------------
// Entry Points
// ---------------------------------------------------------------------------

/// Generate all per-filter doc files and the reference index.
pub(crate) fn generate(_args: GenerateArgs) {
    let root = workspace_root();
    let shared_items = parse_shared_config_items(&root);
    let all_filters = discover_all_filters(&root, &shared_items);
    let docs_dir = root.join("docs/filters");
    let mut wrote = 0;

    create_dir_all_or_exit(&docs_dir);

    for entry in &all_filters {
        let path = docs_dir.join(format!("{}.md", entry.filter.name));
        let content = render_filter_doc(entry);
        write_or_exit(&path, &content);
        print_relative(&root, &path, "wrote");
        wrote += 1;
    }

    let index = render_reference_index(&all_filters);
    let index_path = docs_dir.join("reference.md");
    write_or_exit(&index_path, &index);
    print_relative(&root, &index_path, "wrote");
    wrote += 1;

    remove_stale_docs(&root, &docs_dir, &all_filters);
    println!("{wrote} filter doc(s) generated");
}

/// Check that all per-filter doc files and the reference index are up to date.
pub(crate) fn lint(_args: LintArgs) {
    let root = workspace_root();
    let shared_items = parse_shared_config_items(&root);
    let all_filters = discover_all_filters(&root, &shared_items);
    let docs_dir = root.join("docs/filters");
    let stale = collect_stale_doc_paths(&root, &docs_dir, &all_filters);

    if stale.is_empty() {
        println!("all filter doc files are up to date");
    } else {
        eprintln!("filter doc files are stale:");
        for path in &stale {
            eprintln!("  {}", path.display());
        }
        eprintln!("\nrun: cargo xtask generate-filter-docs");
        std::process::exit(1);
    }
}

/// Return generated doc paths that differ from current source metadata.
fn collect_stale_doc_paths(root: &Path, docs_dir: &Path, all_filters: &[FilterEntry]) -> Vec<PathBuf> {
    let mut stale = Vec::new();

    for entry in all_filters {
        let path = docs_dir.join(format!("{}.md", entry.filter.name));
        let expected = render_filter_doc(entry);
        if !file_matches(&path, &expected) {
            stale.push(relative_path(root, &path));
        }
    }

    let index_path = docs_dir.join("reference.md");
    let expected_index = render_reference_index(all_filters);
    if !file_matches(&index_path, &expected_index) {
        stale.push(relative_path(root, &index_path));
    }

    check_for_stale_files(root, docs_dir, all_filters, &mut stale);
    stale
}

// ---------------------------------------------------------------------------
// Data Types
// ---------------------------------------------------------------------------

/// A filter with its location metadata for output path construction.
struct FilterEntry {
    /// Crate kind: `"apis"` for provider-specific, `"filters"` for
    /// cross-cutting.
    crate_kind: String,
    /// Category slug (e.g. `anthropic`, `agentic`, `guardrails`).
    category: String,
    /// Extracted filter information.
    filter: FilterInfo,
}

/// Information extracted for one filter.
#[derive(Clone)]
struct FilterInfo {
    /// Filter name as returned by `fn name()` (e.g. `"a2a"`).
    name: String,
    /// First paragraph of the filter struct doc comment.
    description: String,
    /// First paragraphs from same-name filter variants.
    extra_descriptions: Vec<String>,
    /// Additional notes extracted from config struct docs.
    config_notes: Vec<String>,
    /// Config fields in declaration order.
    fields: Vec<FieldInfo>,
    /// YAML configuration example from doc comments.
    yaml_examples: Vec<String>,
}

/// Information extracted for one config field.
#[derive(Clone)]
struct FieldInfo {
    /// Field name.
    name: String,
    /// Human-readable type string.
    type_str: String,
    /// Doc comment text.
    doc: String,
    /// Field presence requirement.
    required: RequiredKind,
}

/// How a field must appear in YAML.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequiredKind {
    /// Field must be present.
    Yes,
    /// Field may be omitted.
    No,
    /// Field is one of several mutually exclusive alternatives.
    OneOf,
}

impl RequiredKind {
    /// Render the table-cell label.
    fn as_str(self) -> &'static str {
        match self {
            Self::Yes => "yes",
            Self::No => "no",
            Self::OneOf => "one of",
        }
    }
}

/// A filter anchor: a source file containing both `fn name()` and `fn from_config()`.
struct FilterAnchor {
    /// Path to the anchor file.
    file: PathBuf,
    /// Filter name from `fn name()`.
    name: String,
    /// Config struct type name from `parse_filter_config::<T>()`.
    config_type_name: Option<String>,
}

/// A parsed config struct with its name and fields.
#[derive(Clone)]
struct ConfigStruct {
    /// Struct identifier (e.g. `StaticResponseConfig`).
    name: String,
    /// Config struct doc comment.
    doc: String,
    /// Fields in declaration order.
    fields: Vec<RawField>,
}

/// Parsed items accumulated from source files belonging to one filter module.
struct ModuleItems {
    /// Module-level doc comments from files in the filter scope.
    module_docs: Vec<String>,
    /// Local config structs found (with `Deserialize` + `deny_unknown_fields`).
    configs: Vec<ConfigStruct>,
    /// Doc comments on public structs and filter implementation structs.
    struct_docs: Vec<String>,
    /// Struct definitions available for nested field rendering.
    structs: BTreeMap<String, ConfigStruct>,
    /// Enum definitions with `Deserialize` for variant rendering.
    enums: BTreeMap<String, EnumInfo>,
}

impl ModuleItems {
    /// Create an empty item collection.
    fn new() -> Self {
        Self {
            module_docs: Vec::new(),
            configs: Vec::new(),
            struct_docs: Vec::new(),
            structs: BTreeMap::new(),
            enums: BTreeMap::new(),
        }
    }

    /// Create per-filter items seeded with shared struct and enum metadata.
    fn clone_for_filter(&self) -> Self {
        Self {
            module_docs: Vec::new(),
            configs: Vec::new(),
            struct_docs: Vec::new(),
            structs: self.structs.clone(),
            enums: self.enums.clone(),
        }
    }
}

/// Parsed enum metadata.
#[derive(Clone)]
struct EnumInfo {
    /// YAML variant names.
    variants: Vec<String>,
    /// Whether serde tries variants by shape instead of by variant tag.
    untagged: bool,
    /// Source shape for each variant.
    variant_shapes: Vec<EnumVariantShape>,
    /// Named fields from struct-like variants.
    fields: Vec<RawField>,
}

/// Source shape for one enum variant.
#[derive(Clone)]
enum EnumVariantShape {
    /// Unit variant, usually rendered as a scalar YAML value.
    Unit,
    /// Tuple variant with one wrapped type.
    Unnamed(Box<syn::Type>),
    /// Struct-like variant, rendered as a YAML object when untagged.
    Named,
}

/// A raw field before type rendering.
#[derive(Clone)]
struct RawField {
    /// Field name.
    name: String,
    /// Raw type from syn.
    ty: syn::Type,
    /// Doc comment lines joined.
    doc: String,
    /// Has `#[serde(default)]` or `#[serde(default = "...")]`.
    has_default: bool,
    /// Custom serde deserializer from `#[serde(deserialize_with = "...")]`.
    deserialize_with: Option<String>,
    /// Has `#[serde(flatten)]`.
    flatten: bool,
    /// Additional requiredness hint from surrounding syntax.
    requirement_hint: RequirementHint,
}

/// Requiredness hint from the surrounding source shape.
#[derive(Clone, Copy)]
enum RequirementHint {
    /// Use the field type and serde defaults to infer requiredness.
    Normal,
    /// Field came from one of several struct-like enum variants.
    OneOf,
}

impl FilterInfo {
    /// Merge another same-name filter variant into this doc entry.
    fn merge(&mut self, other: Self) {
        if self.description.is_empty() {
            self.description = other.description;
        } else if !other.description.is_empty()
            && other.description != self.description
            && !self.extra_descriptions.contains(&other.description)
        {
            self.extra_descriptions.push(other.description);
        }
        append_unique(&mut self.extra_descriptions, other.extra_descriptions);
        append_unique(&mut self.config_notes, other.config_notes);
        append_unique_fields(&mut self.fields, other.fields);
        append_unique(&mut self.yaml_examples, other.yaml_examples);
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Parse shared config types that filters may reference from praxis core.
fn parse_shared_config_items(root: &Path) -> ModuleItems {
    let mut items = ModuleItems::new();
    let praxis_root = root.join("../praxis");
    for dir in &[
        praxis_root.join("core/src/config"),
        praxis_root.join("tls/src/config"),
        praxis_root.join("filter/src/builtins/http/payload_processing"),
    ] {
        for path in collect_rs_files(dir) {
            let Ok(source) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(file) = syn::parse_file(&source) else {
                continue;
            };
            parse_file_items(&file, &mut items);
        }
    }

    items.configs.clear();
    items.module_docs.clear();
    items.struct_docs.clear();
    items
}

/// Discover all filters across `apis/src/` and `filters/src/`.
fn discover_all_filters(root: &Path, shared_items: &ModuleItems) -> Vec<FilterEntry> {
    let mut entries = Vec::new();

    discover_crate_filters(root, &root.join("apis/src"), "apis", shared_items, &mut entries);
    discover_crate_filters(root, &root.join("filters/src"), "filters", shared_items, &mut entries);

    entries.sort_by(|a, b| a.filter.name.cmp(&b.filter.name));
    entries
}

/// Discover filters within one crate source directory.
///
/// Scans top-level subdirectories as categories. Standalone `.rs`
/// files directly under `src_dir` are assigned to a `"general"`
/// category.
fn discover_crate_filters(
    _root: &Path,
    src_dir: &Path,
    crate_kind: &str,
    shared_items: &ModuleItems,
    entries: &mut Vec<FilterEntry>,
) {
    let (category_dirs, standalone_anchors) = classify_src_entries(src_dir);

    for category_dir in &category_dirs {
        let category = dir_file_name(category_dir);
        let filters = extract_filters(category_dir, shared_items);
        for filter in filters {
            entries.push(FilterEntry {
                crate_kind: crate_kind.to_owned(),
                category: category.clone(),
                filter,
            });
        }
    }

    for anchor_path in &standalone_anchors {
        if let Some(entry) = extract_standalone_filter(anchor_path, crate_kind, shared_items) {
            entries.push(entry);
        }
    }
}

/// Partition a crate `src/` directory into category subdirs and
/// standalone anchor files.
fn classify_src_entries(src_dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let Ok(dir_entries) = fs::read_dir(src_dir) else {
        return (Vec::new(), Vec::new());
    };

    let mut standalone_anchor_files: Vec<PathBuf> = Vec::new();
    let mut category_dirs: Vec<PathBuf> = Vec::new();

    for entry in dir_entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = dir_file_name(&path);
            if name != "store" && name != "classifier" && name != "token_usage" {
                category_dirs.push(path);
            }
        } else if path.extension().is_some_and(|e| e == "rs")
            && path
                .file_name()
                .is_some_and(|n| n != "lib.rs" && n != "mod.rs" && n != "tests.rs")
            && parse_anchor_file(&path).is_some()
        {
            standalone_anchor_files.push(path);
        }
    }

    category_dirs.sort();
    (category_dirs, standalone_anchor_files)
}

/// Extract a filter entry from a standalone anchor file.
fn extract_standalone_filter(anchor_path: &Path, crate_kind: &str, shared_items: &ModuleItems) -> Option<FilterEntry> {
    let source = fs::read_to_string(anchor_path).ok()?;
    let file = syn::parse_file(&source).ok()?;
    let mut items = shared_items.clone_for_filter();
    parse_file_items(&file, &mut items);

    let anchor = parse_anchor_file(anchor_path)?;
    let filter = build_filter(&items, &anchor.name, anchor.config_type_name.as_deref());
    let category = infer_standalone_category(&anchor.name);
    Some(FilterEntry {
        crate_kind: crate_kind.to_owned(),
        category,
        filter,
    })
}

/// Infer a category for standalone filter files from the filter name.
fn infer_standalone_category(name: &str) -> String {
    if name.contains("token") {
        "token_usage".to_owned()
    } else {
        "general".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Filter Extraction
// ---------------------------------------------------------------------------

/// Extract all filters from a category directory using anchor-based discovery.
fn extract_filters(category_dir: &Path, shared_items: &ModuleItems) -> Vec<FilterInfo> {
    let anchors = discover_filter_anchors(category_dir);

    let mut category_shared = shared_items.clone_for_filter();
    parse_category_shared_types(category_dir, &anchors, &mut category_shared);

    let filters: Vec<FilterInfo> = anchors
        .iter()
        .map(|anchor| {
            let files = scope_files_for_anchor(anchor, category_dir, &anchors);
            let mut items = category_shared.clone_for_filter();
            for path in &files {
                let Ok(source) = fs::read_to_string(path) else {
                    continue;
                };
                let Ok(file) = syn::parse_file(&source) else {
                    continue;
                };
                parse_file_items(&file, &mut items);
            }
            build_filter(&items, &anchor.name, anchor.config_type_name.as_deref())
        })
        .collect();
    merge_filter_variants(filters)
}

/// Parse non-anchor `.rs` files directly in the category root for shared
/// enum/struct types. Only struct field info and enums survive
/// `clone_for_filter`, so module docs and config structs from these
/// files do not leak into individual filter docs.
fn parse_category_shared_types(category_dir: &Path, anchors: &[FilterAnchor], out: &mut ModuleItems) {
    let Ok(entries) = fs::read_dir(category_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        if path.file_name().is_some_and(|n| n == "mod.rs" || n == "tests.rs") {
            continue;
        }
        if anchors.iter().any(|a| a.file == path) {
            continue;
        }
        let Ok(source) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(file) = syn::parse_file(&source) else {
            continue;
        };
        parse_file_items(&file, out);
    }
    out.configs.clear();
    out.module_docs.clear();
    out.struct_docs.clear();
}

/// Discover filter anchor files under a directory tree.
///
/// An anchor is a `.rs` file containing both `fn name()` and
/// `fn from_config()`. Duplicate names are preserved so variant
/// configs, such as MCP broker mode, can be merged into one doc.
fn discover_filter_anchors(dir: &Path) -> Vec<FilterAnchor> {
    let rs_files = collect_rs_files(dir);
    let mut anchors: Vec<FilterAnchor> = rs_files.iter().filter_map(|path| parse_anchor_file(path)).collect();
    anchors.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.file.components().count().cmp(&b.file.components().count()))
            .then_with(|| a.file.cmp(&b.file))
    });

    anchors
}

/// Merge same-name filter variants into a single rendered doc entry.
fn merge_filter_variants(filters: Vec<FilterInfo>) -> Vec<FilterInfo> {
    let mut by_name = BTreeMap::<String, FilterInfo>::new();
    for filter in filters {
        by_name
            .entry(filter.name.clone())
            .and_modify(|existing| existing.merge(filter.clone()))
            .or_insert(filter);
    }
    by_name.into_values().collect()
}

/// Parse a single file to check if it is a filter anchor.
fn parse_anchor_file(path: &Path) -> Option<FilterAnchor> {
    let source = fs::read_to_string(path).ok()?;
    let file = syn::parse_file(&source).ok()?;

    let mut filter_name = None;
    let mut config_type = None;
    let mut has_factory = false;

    for item in &file.items {
        if let syn::Item::Impl(imp) = item {
            if let Some(name) = extract_filter_name(imp) {
                filter_name = Some(name);
            }
            if has_from_config_method(imp) {
                has_factory = true;
                config_type = extract_config_type_name(imp);
            }
        }
    }

    let name = filter_name?;
    if !has_factory {
        return None;
    }

    Some(FilterAnchor {
        file: path.to_owned(),
        name,
        config_type_name: config_type,
    })
}

/// Determine the `.rs` files belonging to a filter anchor's scope.
///
/// Standalone files (sharing a directory with other anchors) own only
/// themselves. Module-directory anchors own their directory tree,
/// excluding subdirectories that contain other anchors. Nested anchors also
/// include parent support modules so imported config enums remain resolvable.
fn scope_files_for_anchor(anchor: &FilterAnchor, category_dir: &Path, all_anchors: &[FilterAnchor]) -> Vec<PathBuf> {
    let Some(anchor_dir) = anchor.file.parent() else {
        return vec![anchor.file.clone()];
    };

    if is_root_anchor_with_nested_anchors(anchor, category_dir, all_anchors) {
        return root_anchor_scope_files(anchor, category_dir, all_anchors);
    }

    let has_sibling = all_anchors
        .iter()
        .any(|a| a.file != anchor.file && a.file.parent() == Some(anchor_dir));

    if has_sibling {
        let mut files = vec![anchor.file.clone()];
        append_ancestor_support_files(category_dir, anchor_dir, all_anchors, &mut files);
        files.sort();
        files.dedup();
        return files;
    }

    let excluded: HashSet<&Path> = all_anchors
        .iter()
        .filter(|a| a.file != anchor.file && a.file.starts_with(anchor_dir) && a.file.parent() != Some(anchor_dir))
        .filter_map(|a| a.file.parent())
        .collect();

    let mut files = Vec::new();
    scope_files_recursive(anchor_dir, &excluded, &mut files);
    append_ancestor_support_files(category_dir, anchor_dir, all_anchors, &mut files);
    files.sort();
    files.dedup();
    files
}

/// Return whether this category-root anchor should avoid nested modules.
fn is_root_anchor_with_nested_anchors(
    anchor: &FilterAnchor,
    category_dir: &Path,
    all_anchors: &[FilterAnchor],
) -> bool {
    anchor.file.parent() == Some(category_dir)
        && all_anchors
            .iter()
            .any(|a| a.file != anchor.file && a.file.starts_with(category_dir))
}

/// Collect files for a filter anchored directly in the category root.
fn root_anchor_scope_files(anchor: &FilterAnchor, category_dir: &Path, all_anchors: &[FilterAnchor]) -> Vec<PathBuf> {
    let mut files = vec![anchor.file.clone()];
    append_direct_support_files(category_dir, all_anchors, &mut files);
    files.sort();
    files.dedup();
    files
}

/// Include non-anchor `.rs` files from ancestor module directories.
fn append_ancestor_support_files(
    category_dir: &Path,
    anchor_dir: &Path,
    all_anchors: &[FilterAnchor],
    out: &mut Vec<PathBuf>,
) {
    let mut current = anchor_dir.parent();
    while let Some(dir) = current {
        if dir == category_dir || !dir.starts_with(category_dir) {
            break;
        }
        append_direct_support_files(dir, all_anchors, out);
        current = dir.parent();
    }
}

/// Append direct child `.rs` support files from one directory.
fn append_direct_support_files(dir: &Path, all_anchors: &[FilterAnchor], out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rs")
            && path.file_name().is_some_and(|n| n != "tests.rs")
            && !all_anchors.iter().any(|anchor| anchor.file == path)
        {
            out.push(path);
        }
    }
}

/// Recursively collect `.rs` files, skipping excluded sub-anchor directories.
fn scope_files_recursive(dir: &Path, excluded: &HashSet<&Path>, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if !excluded.contains(path.as_path()) {
                scope_files_recursive(&path, excluded, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") && path.file_name().is_some_and(|n| n != "tests.rs") {
            out.push(path);
        }
    }
}

/// Recursively collect `.rs` files, skipping test files.
fn collect_rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_rs_files_recursive(dir, &mut files);
    files.sort();
    files
}

/// Walk directory tree collecting `.rs` files.
fn collect_rs_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files_recursive(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") && path.file_name().is_some_and(|n| n != "tests.rs") {
            out.push(path);
        }
    }
}

// ---------------------------------------------------------------------------
// syn Parsing
// ---------------------------------------------------------------------------

/// Parse a syn file and accumulate config structs and enums into `out`.
fn parse_file_items(file: &syn::File, out: &mut ModuleItems) {
    let module_docs = extract_doc_comment(&file.attrs);
    if !module_docs.is_empty() {
        out.module_docs.push(module_docs);
    }

    for item in &file.items {
        match item {
            syn::Item::Struct(s) => parse_struct(s, out),
            syn::Item::Enum(e) if derives_deserialize(&e.attrs) => {
                let info = extract_enum_info(e);
                if !info.variants.is_empty() {
                    out.enums.insert(e.ident.to_string(), info);
                }
            },
            _ => {},
        }
    }
}

/// Handle a struct item: check for config struct and filter doc comments.
fn parse_struct(s: &syn::ItemStruct, out: &mut ModuleItems) {
    let docs = extract_doc_comment(&s.attrs);
    if let Some(fields) = parse_config_fields(s) {
        let config = ConfigStruct {
            name: s.ident.to_string(),
            doc: docs.clone(),
            fields,
        };
        if is_nested_config_struct(s) {
            out.structs.insert(config.name.clone(), config.clone());
            if config.name == "ClusterTlsRaw" {
                out.structs.insert(
                    "ClusterTls".to_owned(),
                    ConfigStruct {
                        name: "ClusterTls".to_owned(),
                        doc: config.doc.clone(),
                        fields: config.fields.clone(),
                    },
                );
            }
        }
        if is_config_struct(s) {
            out.configs.push(config);
        }
    }
    if !docs.is_empty() && is_filter_doc_candidate(s) {
        out.struct_docs.push(docs);
    }
}

/// Return whether a struct's doc comment should contribute filter prose.
fn is_filter_doc_candidate(s: &syn::ItemStruct) -> bool {
    matches!(s.vis, syn::Visibility::Public(_)) || s.ident.to_string().ends_with("Filter")
}

/// Build a [`FilterInfo`] from parsed items, using the anchor's name and config type.
fn build_filter(items: &ModuleItems, name: &str, config_type: Option<&str>) -> FilterInfo {
    let description_doc = items
        .struct_docs
        .iter()
        .find(|doc| !doc.is_empty())
        .or_else(|| items.module_docs.iter().find(|doc| !doc.is_empty()))
        .cloned()
        .unwrap_or_default();

    let description = first_paragraph(&description_doc);
    let yaml_examples = collect_yaml_examples(items);

    let config = select_config(items, config_type);
    let mut all_notes = filter_notes(&description_doc);
    let cfg_notes = config.map_or_else(Vec::new, |c| config_notes(&c.doc));
    append_unique(&mut all_notes, cfg_notes);
    let fields = config.map_or_else(Vec::new, |c| build_fields(c, items));

    FilterInfo {
        name: name.to_owned(),
        description,
        extra_descriptions: Vec::new(),
        config_notes: all_notes,
        fields,
        yaml_examples,
    }
}

/// Collect all unique YAML examples from module and filter struct docs.
fn collect_yaml_examples(items: &ModuleItems) -> Vec<String> {
    let mut examples = Vec::new();
    for doc in items.module_docs.iter().chain(items.struct_docs.iter()) {
        append_unique(&mut examples, extract_yaml_examples(doc));
    }
    examples
}

/// Build rendered field metadata for one config struct.
fn build_fields(config: &ConfigStruct, items: &ModuleItems) -> Vec<FieldInfo> {
    let mut fields = Vec::new();
    let mut stack = Vec::new();
    append_rendered_fields("", &config.fields, items, &mut stack, &mut fields);
    fields
}

/// Append rendered field rows, including nested config fields.
fn append_rendered_fields(
    prefix: &str,
    raw_fields: &[RawField],
    items: &ModuleItems,
    stack: &mut Vec<String>,
    out: &mut Vec<FieldInfo>,
) {
    for field in raw_fields {
        let path = field_path(prefix, &field.name);
        if !field.flatten {
            out.push(FieldInfo {
                name: path.clone(),
                type_str: render_field_type(field, &items.enums),
                doc: field.doc.clone(),
                required: required_kind(field),
            });
        }

        let nested_prefix = if field.flatten {
            prefix.to_owned()
        } else {
            collection_field_path(&path, &field.ty)
        };
        append_nested_fields(&nested_prefix, &field.ty, items, stack, out);
    }
}

/// Append nested rows for a field type when its shape is known.
fn append_nested_fields(
    prefix: &str,
    ty: &syn::Type,
    items: &ModuleItems,
    stack: &mut Vec<String>,
    out: &mut Vec<FieldInfo>,
) {
    let Some(type_name) = nested_type_name(ty) else {
        return;
    };
    if stack.iter().any(|name| name == &type_name) {
        return;
    }

    if let Some(config) = items.structs.get(&type_name) {
        stack.push(type_name);
        append_rendered_fields(prefix, &config.fields, items, stack, out);
        stack.pop();
    } else if let Some(info) = items.enums.get(&type_name)
        && !info.fields.is_empty()
    {
        stack.push(type_name);
        append_rendered_fields(prefix, &info.fields, items, stack, out);
        stack.pop();
    }
}

/// Return the rendered requirement kind for a raw field.
fn required_kind(field: &RawField) -> RequiredKind {
    if matches!(field.requirement_hint, RequirementHint::OneOf) {
        RequiredKind::OneOf
    } else if !field.has_default && !is_option_type(&field.ty) {
        RequiredKind::Yes
    } else {
        RequiredKind::No
    }
}

/// Build a dotted field path.
fn field_path(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else if name.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Add collection notation to field paths for nested sequence types.
fn collection_field_path(path: &str, ty: &syn::Type) -> String {
    if is_sequence_type(ty) {
        format!("{path}[]")
    } else {
        path.to_owned()
    }
}

/// Select the config struct matching the type name from `parse_filter_config`.
fn select_config<'a>(items: &'a ModuleItems, config_type: Option<&str>) -> Option<&'a ConfigStruct> {
    let type_name = config_type?;
    items.configs.iter().find(|c| c.name == type_name)
}

// ---------------------------------------------------------------------------
// Attribute Helpers
// ---------------------------------------------------------------------------

/// Check if attributes include `#[derive(..., Deserialize)]`.
fn derives_deserialize(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("derive") {
            return false;
        }
        let Ok(meta) = attr.parse_args_with(syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated)
        else {
            return false;
        };
        meta.iter()
            .any(|p| p.segments.last().is_some_and(|s| s.ident == "Deserialize"))
    })
}

/// Check if a struct derives `Deserialize` and has `#[serde(deny_unknown_fields)]`.
fn is_config_struct(s: &syn::ItemStruct) -> bool {
    derives_deserialize(&s.attrs) && has_serde_attr(&s.attrs, "deny_unknown_fields")
}

/// Check if a struct can be used for nested config rendering.
fn is_nested_config_struct(s: &syn::ItemStruct) -> bool {
    derives_deserialize(&s.attrs) || matches!(s.vis, syn::Visibility::Public(_))
}

/// Check if attributes contain `#[serde(<ident>)]`.
fn has_serde_attr(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| serde_attr_contains(attr, name))
}

/// Parse fields from a config struct.
fn parse_config_fields(s: &syn::ItemStruct) -> Option<Vec<RawField>> {
    let syn::Fields::Named(fields) = &s.fields else {
        return None;
    };

    Some(
        fields
            .named
            .iter()
            .map(|f| RawField {
                name: serde_field_name(f),
                doc: extract_doc_comment(&f.attrs),
                has_default: has_serde_default(&f.attrs),
                deserialize_with: serde_deserialize_with(&f.attrs),
                flatten: has_serde_attr(&f.attrs, "flatten"),
                requirement_hint: RequirementHint::Normal,
                ty: f.ty.clone(),
            })
            .collect(),
    )
}

/// Extract the concatenated doc comment from attributes.
fn extract_doc_comment(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if let syn::Meta::NameValue(nv) = &attr.meta
                && attr.path().is_ident("doc")
                && let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s), ..
                }) = &nv.value
            {
                return Some(s.value());
            }
            None
        })
        .collect();

    let trimmed: Vec<&str> = lines
        .iter()
        .map(|l| l.strip_prefix(' ').unwrap_or(l.as_str()))
        .collect();
    trimmed.join("\n").trim().to_owned()
}

/// Check if a field has `#[serde(default)]` or `#[serde(default = "...")]`.
fn has_serde_default(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| serde_attr_contains(attr, "default"))
}

/// Return the YAML field name for a struct field.
fn serde_field_name(field: &syn::Field) -> String {
    field
        .attrs
        .iter()
        .find_map(serde_rename)
        .or_else(|| field.ident.as_ref().map(ToString::to_string))
        .unwrap_or_default()
}

/// Return whether a serde attribute contains a given nested key.
fn serde_attr_contains(attr: &syn::Attribute, name: &str) -> bool {
    if !attr.path().is_ident("serde") {
        return false;
    }

    let mut found = false;
    drop(attr.parse_nested_meta(|meta| {
        if meta.path.is_ident(name) {
            found = true;
        }
        Ok(())
    }));
    found
}

/// Extract `#[serde(rename = "...")]` from a field or variant.
fn serde_rename(attr: &syn::Attribute) -> Option<String> {
    serde_lit_value(attr, "rename")
}

/// Extract `#[serde(deserialize_with = "...")]` from a field.
fn serde_deserialize_with(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|attr| serde_lit_value(attr, "deserialize_with"))
}

/// Extract a string-literal serde attribute value.
fn serde_lit_value(attr: &syn::Attribute, name: &str) -> Option<String> {
    if !attr.path().is_ident("serde") {
        return None;
    }

    let mut value = None;
    drop(attr.parse_nested_meta(|meta| {
        if meta.path.is_ident(name) {
            let meta_value = meta.value()?;
            let lit: syn::LitStr = meta_value.parse()?;
            value = Some(lit.value());
        } else if meta.input.peek(syn::Token![=]) {
            let meta_value = meta.value()?;
            let _: syn::Expr = meta_value.parse()?;
        }
        Ok(())
    }));
    value
}

/// Extract enum metadata, applying serde rename rules where present.
fn extract_enum_info(e: &syn::ItemEnum) -> EnumInfo {
    let rename_all = detect_rename_all(&e.attrs);
    let untagged = has_serde_attr(&e.attrs, "untagged");
    let variants = e
        .variants
        .iter()
        .map(|v| {
            v.attrs
                .iter()
                .find_map(serde_rename)
                .unwrap_or_else(|| apply_rename(&v.ident.to_string(), rename_all))
        })
        .collect();
    let variant_shapes = e.variants.iter().map(enum_variant_shape).collect();
    let mut variant_fields: Vec<Vec<RawField>> = e.variants.iter().map(parse_variant_fields).collect();
    let named_variant_count = variant_fields.iter().filter(|fields| !fields.is_empty()).count();
    if named_variant_count > 1 {
        for fields in &mut variant_fields {
            for field in fields {
                field.requirement_hint = RequirementHint::OneOf;
            }
        }
    }
    let fields = variant_fields.into_iter().flatten().collect();

    EnumInfo {
        variants,
        untagged,
        variant_shapes,
        fields,
    }
}

/// Return the source shape for an enum variant.
fn enum_variant_shape(variant: &syn::Variant) -> EnumVariantShape {
    match &variant.fields {
        syn::Fields::Unit => EnumVariantShape::Unit,
        syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
            fields.unnamed.first().map_or(EnumVariantShape::Named, |field| {
                EnumVariantShape::Unnamed(Box::new(field.ty.clone()))
            })
        },
        syn::Fields::Unnamed(_) | syn::Fields::Named(_) => EnumVariantShape::Named,
    }
}

/// Detect `#[serde(rename_all = "...")]` on attributes.
fn detect_rename_all(attrs: &[syn::Attribute]) -> Option<&'static str> {
    let value = attrs.iter().find_map(|attr| serde_lit_value(attr, "rename_all"))?;
    match value.as_str() {
        "snake_case" => Some("snake_case"),
        "lowercase" => Some("lowercase"),
        "UPPERCASE" => Some("UPPERCASE"),
        "camelCase" => Some("camelCase"),
        "PascalCase" => Some("PascalCase"),
        "kebab-case" => Some("kebab-case"),
        "SCREAMING_SNAKE_CASE" => Some("SCREAMING_SNAKE_CASE"),
        "SCREAMING-KEBAB-CASE" => Some("SCREAMING-KEBAB-CASE"),
        _ => None,
    }
}

/// Apply a rename rule to a variant name.
fn apply_rename(name: &str, rule: Option<&str>) -> String {
    match rule {
        Some("snake_case") => to_snake_case(name),
        Some("lowercase") => name.to_lowercase(),
        Some("UPPERCASE") => name.to_uppercase(),
        Some("camelCase") => to_camel_case(name),
        Some("kebab-case") => to_snake_case(name).replace('_', "-"),
        Some("SCREAMING_SNAKE_CASE") => to_snake_case(name).to_uppercase(),
        Some("SCREAMING-KEBAB-CASE") => to_snake_case(name).to_uppercase().replace('_', "-"),
        _ => name.to_owned(),
    }
}

/// Parse fields from struct-like enum variants.
fn parse_variant_fields(variant: &syn::Variant) -> Vec<RawField> {
    let syn::Fields::Named(fields) = &variant.fields else {
        return Vec::new();
    };
    fields
        .named
        .iter()
        .map(|f| RawField {
            name: serde_field_name(f),
            doc: extract_doc_comment(&f.attrs),
            has_default: has_serde_default(&f.attrs),
            deserialize_with: serde_deserialize_with(&f.attrs),
            flatten: has_serde_attr(&f.attrs, "flatten"),
            requirement_hint: RequirementHint::Normal,
            ty: f.ty.clone(),
        })
        .collect()
}

/// Extract filter name from `fn name(&self) -> &'static str { "..." }`.
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

/// Check if an impl block contains a `from_config` method.
fn has_from_config_method(imp: &syn::ItemImpl) -> bool {
    imp.items
        .iter()
        .any(|item| matches!(item, syn::ImplItem::Fn(method) if method.sig.ident == "from_config"))
}

/// Extract the config type name from `let cfg: T = parse_filter_config(...)`.
fn extract_config_type_name(imp: &syn::ItemImpl) -> Option<String> {
    let method = imp.items.iter().find_map(|item| {
        if let syn::ImplItem::Fn(m) = item
            && m.sig.ident == "from_config"
        {
            return Some(m);
        }
        None
    })?;

    for stmt in &method.block.stmts {
        if let syn::Stmt::Local(local) = stmt {
            let Some(init) = local.init.as_ref() else {
                continue;
            };
            let init_str = init.expr.to_token_stream().to_string();
            if !init_str.contains("parse_filter_config") {
                continue;
            }
            if let syn::Pat::Type(pat_type) = &local.pat {
                return Some(pat_type.ty.to_token_stream().to_string());
            }
        }
    }
    None
}

/// Extract a string literal value from an expression.
fn extract_str_literal(expr: &syn::Expr) -> Option<String> {
    if let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(s), ..
    }) = expr
    {
        Some(s.value())
    } else {
        None
    }
}

/// Convert `PascalCase` to `snake_case`.
fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// Convert `PascalCase` to `camelCase`.
fn to_camel_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let mut out = first.to_lowercase().collect::<String>();
            out.push_str(chars.as_str());
            out
        },
    }
}

// ---------------------------------------------------------------------------
// YAML Example Extraction
// ---------------------------------------------------------------------------

/// Extract YAML example blocks from a full doc comment.
///
/// Looks for Markdown headings that include `YAML`, such as `# YAML`,
/// `# YAML configuration`, or `# Single-field YAML`, followed by a fenced
/// ` ```yaml ... ``` ` block.
fn extract_yaml_examples(doc: &str) -> Vec<String> {
    let mut examples = Vec::new();
    let mut lines = doc.lines().peekable();
    while let Some(line) = lines.next() {
        if is_yaml_heading(line)
            && let Some(example) = extract_yaml_fence_after_heading(&mut lines)
        {
            examples.push(example);
        }
    }
    examples
}

/// Extract the first YAML fenced block after the current heading.
fn extract_yaml_fence_after_heading(lines: &mut std::iter::Peekable<std::str::Lines<'_>>) -> Option<String> {
    while let Some(line) = lines.peek().copied() {
        let trimmed = line.trim();
        if is_yaml_fence_start(trimmed) {
            lines.next();
            return collect_yaml_fence(lines);
        }
        if is_markdown_heading(trimmed) {
            return None;
        }
        lines.next();
    }
    None
}

/// Collect lines until the closing fence.
fn collect_yaml_fence(lines: &mut std::iter::Peekable<std::str::Lines<'_>>) -> Option<String> {
    let mut yaml = Vec::new();
    for line in lines.by_ref() {
        if line.trim() == "```" {
            break;
        }
        yaml.push(line);
    }
    (!yaml.is_empty()).then(|| yaml.join("\n"))
}

/// Return whether a doc line is a Markdown heading whose text mentions YAML.
fn is_yaml_heading(line: &str) -> bool {
    let trimmed = line.trim();
    if !is_markdown_heading(trimmed) {
        return false;
    }
    let heading = trimmed.trim_start_matches('#').trim();
    heading
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|word| word.eq_ignore_ascii_case("yaml"))
}

/// Return whether a doc line starts a Markdown heading.
fn is_markdown_heading(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

/// Return whether a doc line starts a YAML fenced block.
fn is_yaml_fence_start(line: &str) -> bool {
    line.starts_with("```yaml") || line.starts_with("```yml")
}

// ---------------------------------------------------------------------------
// Type Rendering
// ---------------------------------------------------------------------------

/// Return whether a type path is `Option<T>`.
fn is_option_type(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Path(tp) if tp.path.segments.last().is_some_and(|s| s.ident == "Option"))
}

/// Render a [`syn::Type`] as a human-readable string for documentation.
fn render_type(ty: &syn::Type, enums: &BTreeMap<String, EnumInfo>) -> String {
    if let syn::Type::Path(tp) = ty {
        render_type_path(tp, enums)
    } else {
        quote::quote!(#ty).to_string()
    }
}

/// Render a field type, accounting for custom serde scalar deserializers.
fn render_field_type(field: &RawField, enums: &BTreeMap<String, EnumInfo>) -> String {
    custom_deserializer_type(field).map_or_else(|| render_type(&field.ty, enums), ToOwned::to_owned)
}

/// Return the YAML-facing type for known custom deserializers.
fn custom_deserializer_type(field: &RawField) -> Option<&'static str> {
    match field.deserialize_with.as_deref() {
        Some("deserialize_redirect_status") => Some("301 \\| 302 \\| 307 \\| 308"),
        _ => None,
    }
}

/// Render a type path, resolving known wrappers and enum types.
fn render_type_path(tp: &syn::TypePath, enums: &BTreeMap<String, EnumInfo>) -> String {
    let last = tp.path.segments.last().expect("TypePath has at least one segment");
    let ident = last.ident.to_string();

    match ident.as_str() {
        "Vec" => render_vec_type(last, enums),
        "Option" => render_inner_or(last, enums, "any"),
        "Arc" => render_arc_type(last, enums),
        "BTreeMap" | "HashMap" => render_map_type(last, enums),
        "String" => "string".to_owned(),
        "SecretString" => "string (secret)".to_owned(),
        "Value" => "any".to_owned(),
        "bool" | "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" | "f32" | "f64" => {
            ident
        },
        other => enums
            .get(other)
            .map_or_else(|| other.to_owned(), |info| render_enum_type(info, enums)),
    }
}

/// Render `Vec<T>` as a YAML array shape.
fn render_vec_type(segment: &syn::PathSegment, enums: &BTreeMap<String, EnumInfo>) -> String {
    let inner = render_inner_or(segment, enums, "any");
    if inner.contains(" \\| ") {
        format!("({inner})[]")
    } else {
        format!("{inner}[]")
    }
}

/// Render an enum as its YAML-facing alternatives.
fn render_enum_type(info: &EnumInfo, enums: &BTreeMap<String, EnumInfo>) -> String {
    if info.untagged {
        return render_union(
            info.variants
                .iter()
                .zip(info.variant_shapes.iter())
                .map(|(name, shape)| render_untagged_variant_type(name, shape, enums)),
        );
    }

    render_union(info.variants.iter().map(|name| format!("`{name}`")))
}

/// Render one untagged enum variant by the YAML shape serde accepts.
fn render_untagged_variant_type(name: &str, shape: &EnumVariantShape, enums: &BTreeMap<String, EnumInfo>) -> String {
    match shape {
        EnumVariantShape::Unit => format!("`{name}`"),
        EnumVariantShape::Unnamed(ty) => render_type(ty, enums),
        EnumVariantShape::Named => "object".to_owned(),
    }
}

/// Join unique rendered alternatives in first-seen order.
fn render_union<I>(items: I) -> String
where
    I: IntoIterator<Item = String>,
{
    let mut labels = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        if seen.insert(item.clone()) {
            labels.push(item);
        }
    }
    labels.join(" \\| ")
}

/// Render `BTreeMap<K, V>` and `HashMap<K, V>` as YAML object shapes.
fn render_map_type(segment: &syn::PathSegment, enums: &BTreeMap<String, EnumInfo>) -> String {
    let args = extract_angle_bracket_args(segment);
    let key = args
        .first()
        .map_or_else(|| "string".to_owned(), |t| render_type(t, enums));
    let value = args.get(1).map_or_else(|| "any".to_owned(), |t| render_type(t, enums));
    format!("object<{key}, {value}>")
}

/// Render the inner type argument or a fallback.
fn render_inner_or(segment: &syn::PathSegment, enums: &BTreeMap<String, EnumInfo>, fallback: &str) -> String {
    extract_angle_bracket_arg(segment).map_or_else(|| fallback.to_owned(), |t| render_type(&t, enums))
}

/// Render `Arc<str>` as `"string"`, other `Arc<T>` by inner type.
fn render_arc_type(segment: &syn::PathSegment, enums: &BTreeMap<String, EnumInfo>) -> String {
    match extract_angle_bracket_arg(segment) {
        Some(syn::Type::Path(p)) if p.path.segments.last().is_some_and(|s| s.ident == "str") => "string".to_owned(),
        Some(t) => render_type(&t, enums),
        None => "any".to_owned(),
    }
}

/// Return whether a type contains a sequence wrapper.
fn is_sequence_type(ty: &syn::Type) -> bool {
    let syn::Type::Path(tp) = ty else {
        return false;
    };
    let Some(segment) = tp.path.segments.last() else {
        return false;
    };
    match segment.ident.to_string().as_str() {
        "Vec" => true,
        "Option" | "Arc" | "Box" => extract_angle_bracket_arg(segment).is_some_and(|inner| is_sequence_type(&inner)),
        _ => false,
    }
}

/// Return the innermost named config type for nested field rendering.
fn nested_type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(tp) = ty else {
        return None;
    };
    let segment = tp.path.segments.last()?;
    let ident = segment.ident.to_string();

    match ident.as_str() {
        "Vec" | "Option" | "Arc" | "Box" => {
            extract_angle_bracket_arg(segment).and_then(|inner| nested_type_name(&inner))
        },
        "BTreeMap" | "HashMap" => extract_angle_bracket_args(segment).get(1).and_then(nested_type_name),
        "String" | "SecretString" | "Value" | "str" | "bool" | "u8" | "u16" | "u32" | "u64" | "usize" | "i8"
        | "i16" | "i32" | "i64" | "isize" | "f32" | "f64" => None,
        _ => Some(ident),
    }
}

/// Extract the first type argument from angle brackets.
fn extract_angle_bracket_arg(segment: &syn::PathSegment) -> Option<syn::Type> {
    extract_angle_bracket_args(segment).into_iter().next()
}

/// Extract type arguments from angle brackets.
fn extract_angle_bracket_args(segment: &syn::PathSegment) -> Vec<syn::Type> {
    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
        return args
            .args
            .iter()
            .filter_map(|arg| match arg {
                syn::GenericArgument::Type(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// Markdown Rendering
// ---------------------------------------------------------------------------

/// Extract the first paragraph from a doc comment.
fn first_paragraph(doc: &str) -> String {
    let mut lines = Vec::new();
    for line in doc.lines() {
        if line.is_empty() || line.starts_with('#') {
            break;
        }
        lines.push(line);
    }
    lines.join(" ").trim().to_owned()
}

/// Extract behavioral notes from a filter struct doc comment.
///
/// Collects prose paragraphs that appear before any heading. Headings
/// and everything after them (YAML examples, Rust doctests) are
/// handled by other extractors or intentionally dropped.
fn filter_notes(doc: &str) -> Vec<String> {
    let mut notes = Vec::new();

    for paragraph in doc.split("\n\n").skip(1) {
        let trimmed = paragraph.trim();
        if trimmed.starts_with('#') {
            break;
        }
        if let Some(note) = normalize_config_note(trimmed) {
            notes.push(note);
        }
    }
    notes
}

/// Extract useful configuration notes from a config struct doc comment.
fn config_notes(doc: &str) -> Vec<String> {
    doc.split("\n\n").skip(1).filter_map(normalize_config_note).collect()
}

/// Normalize one config doc paragraph into prose, dropping fenced code.
fn normalize_config_note(paragraph: &str) -> Option<String> {
    normalize_doc_prose(paragraph)
}

/// Normalize one field doc comment into safe table-cell prose.
fn normalize_field_doc(doc: &str) -> String {
    normalize_doc_prose(doc).unwrap_or_default().replace('|', "\\|")
}

/// Normalize doc comment prose, dropping fenced code and reference definitions.
fn normalize_doc_prose(doc: &str) -> Option<String> {
    let mut in_fence = false;
    let mut lines = Vec::new();
    for line in doc.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || trimmed.is_empty() || is_markdown_reference_definition(trimmed) || trimmed.starts_with('#') {
            continue;
        }
        lines.push(trimmed);
    }

    let normalized = lines.join(" ").trim().to_owned();
    (!normalized.is_empty()).then_some(normalized)
}

/// Return whether a line is a Markdown reference definition.
fn is_markdown_reference_definition(line: &str) -> bool {
    line.starts_with('[') && line.contains("]:")
}

/// Render the markdown content for a single filter doc file.
fn render_filter_doc(entry: &FilterEntry) -> String {
    let mut out = String::new();
    writeln!(out, "<!-- Generated by: cargo xtask generate-filter-docs -->").unwrap();
    writeln!(out, "<!-- Do not edit manually -->").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "# `{}`", entry.filter.name).unwrap();
    writeln!(out).unwrap();

    if !entry.filter.description.is_empty() {
        writeln!(out, "{}", entry.filter.description).unwrap();
    }
    for description in &entry.filter.extra_descriptions {
        writeln!(out).unwrap();
        writeln!(out, "{description}").unwrap();
    }

    render_config_notes(&mut out, &entry.filter.config_notes);
    render_config_table(&mut out, &entry.filter.fields);
    render_yaml_examples(&mut out, &entry.filter.yaml_examples);
    out
}

/// Render configuration notes if present.
fn render_config_notes(out: &mut String, notes: &[String]) {
    if notes.is_empty() {
        return;
    }
    writeln!(out).unwrap();
    writeln!(out, "## Configuration Notes").unwrap();
    for note in notes {
        writeln!(out).unwrap();
        writeln!(out, "{note}").unwrap();
    }
}

/// Render the configuration table if fields are present.
fn render_config_table(out: &mut String, fields: &[FieldInfo]) {
    if fields.is_empty() {
        return;
    }
    writeln!(out).unwrap();
    writeln!(out, "## Configuration").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Field | Type | Required | Description |").unwrap();
    writeln!(out, "|-------|------|---------|-------------|").unwrap();
    for field in fields {
        let doc = normalize_field_doc(&field.doc);
        writeln!(
            out,
            "| `{}` | {} | {} | {} |",
            field.name,
            field.type_str,
            field.required.as_str(),
            doc
        )
        .unwrap();
    }
}

/// Render YAML example sections if present.
fn render_yaml_examples(out: &mut String, examples: &[String]) {
    if examples.is_empty() {
        return;
    }
    writeln!(out).unwrap();
    writeln!(out, "## {}", if examples.len() == 1 { "Example" } else { "Examples" }).unwrap();
    for (index, yaml) in examples.iter().enumerate() {
        if examples.len() > 1 {
            writeln!(out).unwrap();
            writeln!(out, "### Example {}", index + 1).unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "```yaml").unwrap();
        writeln!(out, "{yaml}").unwrap();
        writeln!(out, "```").unwrap();
    }
}

/// Render the reference index linking to all per-filter docs.
fn render_reference_index(entries: &[FilterEntry]) -> String {
    let mut out = render_reference_header();
    let grouped = group_by_crate_category(entries);

    let mut current_crate = "";
    for ((crate_kind, category), filters) in &grouped {
        if *crate_kind != current_crate {
            current_crate = crate_kind;
            writeln!(out).unwrap();
            let crate_title = crate_heading(crate_kind);
            writeln!(out, "## {crate_title}").unwrap();
        }
        render_category_section(&mut out, category, filters);
    }

    out
}

/// Render the static header for the reference index.
fn render_reference_header() -> String {
    let mut out = String::new();
    writeln!(out, "<!-- Generated by: cargo xtask generate-filter-docs -->").unwrap();
    writeln!(out, "<!-- Do not edit manually -->").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "# Filter Reference").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "AI filters provided by Praxis AI. For base proxy").unwrap();
    writeln!(out, "filters (router, load balancer, headers, CORS, etc.),").unwrap();
    writeln!(out, "see the [Praxis core filter reference][core-ref].").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "[core-ref]: https://github.com/praxis-proxy/praxis/blob/main/docs/filters/reference.md"
    )
    .unwrap();
    out
}

/// Return the section heading for a crate kind.
fn crate_heading(crate_kind: &str) -> &str {
    match crate_kind {
        "apis" => "Provider APIs (praxis-ai-apis)",
        "filters" => "Cross-Provider Filters (praxis-ai-filters)",
        other => other,
    }
}

/// Group filter entries by `(crate_kind, category)`.
fn group_by_crate_category(entries: &[FilterEntry]) -> BTreeMap<(&str, &str), Vec<&FilterEntry>> {
    let mut grouped: BTreeMap<(&str, &str), Vec<&FilterEntry>> = BTreeMap::new();
    for entry in entries {
        grouped
            .entry((&entry.crate_kind, &entry.category))
            .or_default()
            .push(entry);
    }
    grouped
}

/// Render one category section in the reference index.
fn render_category_section(out: &mut String, category: &str, filters: &[&FilterEntry]) {
    let title = format_title(category);
    writeln!(out).unwrap();
    writeln!(out, "### {title}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Filter | Description |").unwrap();
    writeln!(out, "|--------|-------------|").unwrap();
    for f in filters {
        let link = format!("{}.md", f.filter.name);
        writeln!(out, "| [`{}`]({link}) | {} |", f.filter.name, f.filter.description).unwrap();
    }
}

/// Format a category slug into a human-readable title.
fn format_title(category_name: &str) -> String {
    category_name
        .split('_')
        .map(|w| match w {
            "ai" => "AI".to_owned(),
            "tcp" => "TCP".to_owned(),
            "ip" => "IP".to_owned(),
            "http" => "HTTP".to_owned(),
            "openai" => "OpenAI".to_owned(),
            "mcp" => "MCP".to_owned(),
            "a2a" => "A2A".to_owned(),
            other => capitalize(other),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Capitalize the first letter of a word.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let mut result = c.to_uppercase().collect::<String>();
            result.push_str(chars.as_str());
            result
        },
    }
}

// ---------------------------------------------------------------------------
// Stale File Management
// ---------------------------------------------------------------------------

/// Remove generated doc files that no longer correspond to a filter.
fn remove_stale_docs(root: &Path, docs_dir: &Path, entries: &[FilterEntry]) {
    let expected = build_expected_paths(docs_dir, entries);

    let Ok(dir_entries) = fs::read_dir(docs_dir) else {
        return;
    };
    for entry in dir_entries.flatten() {
        let path = entry.path();
        if is_generated_md(&path) && !expected.contains(&path) && fs::remove_file(&path).is_ok() {
            print_relative(root, &path, "removed stale");
        }
    }
}

/// Check for stale files during lint and append to the stale list.
fn check_for_stale_files(root: &Path, docs_dir: &Path, entries: &[FilterEntry], stale: &mut Vec<PathBuf>) {
    let expected = build_expected_paths(docs_dir, entries);

    let Ok(dir_entries) = fs::read_dir(docs_dir) else {
        return;
    };
    for entry in dir_entries.flatten() {
        let path = entry.path();
        if is_generated_md(&path) && !expected.contains(&path) {
            stale.push(relative_path(root, &path));
        }
    }
}

/// Build the set of expected doc file paths.
fn build_expected_paths(docs_dir: &Path, entries: &[FilterEntry]) -> HashSet<PathBuf> {
    let mut paths: HashSet<PathBuf> = entries
        .iter()
        .map(|e| docs_dir.join(format!("{}.md", e.filter.name)))
        .collect();
    paths.insert(docs_dir.join("reference.md"));
    paths
}

/// Check if a file is a generated markdown doc (has the generation comment).
fn is_generated_md(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "md")
        && fs::read_to_string(path)
            .is_ok_and(|c| c.starts_with("<!-- Generated by: cargo xtask generate-filter-docs -->"))
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Append strings from `items` to `target`, preserving first occurrence order.
fn append_unique(target: &mut Vec<String>, items: Vec<String>) {
    let mut seen: HashSet<String> = target.iter().cloned().collect();
    for item in items {
        if seen.insert(item.clone()) {
            target.push(item);
        }
    }
}

/// Append fields from `items` to `target`, preserving first occurrence by name.
fn append_unique_fields(target: &mut Vec<FieldInfo>, items: Vec<FieldInfo>) {
    let mut seen: HashSet<String> = target.iter().map(|field| field.name.clone()).collect();
    for item in items {
        if seen.insert(item.name.clone()) {
            target.push(item);
        }
    }
}

/// Locate the workspace root directory.
fn workspace_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_owned());
    Path::new(&manifest_dir)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_owned()
}

/// Extract the final component of a directory path as a string.
fn dir_file_name(dir: &Path) -> String {
    dir.file_name()
        .expect("directory has a file name")
        .to_string_lossy()
        .into_owned()
}

/// Create directories recursively, exiting on failure.
fn create_dir_all_or_exit(dir: &Path) {
    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("failed to create {}: {e}", dir.display());
        std::process::exit(1);
    }
}

/// Write content to a file, exiting on failure.
fn write_or_exit(path: &Path, content: &str) {
    if let Err(e) = fs::write(path, content) {
        eprintln!("failed to write {}: {e}", path.display());
        std::process::exit(1);
    }
}

/// Check if a file's content matches expected content.
fn file_matches(path: &Path, expected: &str) -> bool {
    fs::read_to_string(path).is_ok_and(|actual| actual == expected)
}

/// Print a path relative to root with an action prefix.
fn print_relative(root: &Path, path: &Path, action: &str) {
    println!("  {action} {}", path.strip_prefix(root).unwrap_or(path).display());
}

/// Get a path relative to root.
fn relative_path(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_owned()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use super::*;

    #[test]
    fn to_snake_case_basic() {
        assert_eq!(to_snake_case("Global"), "global", "single word");
        assert_eq!(to_snake_case("PerIp"), "per_ip", "two words");
        assert_eq!(to_snake_case("SomeHTTPMode"), "some_h_t_t_p_mode", "acronym");
    }

    #[test]
    fn capitalize_basic() {
        assert_eq!(capitalize("traffic"), "Traffic", "basic word");
        assert_eq!(capitalize(""), "", "empty string");
    }

    #[test]
    fn first_paragraph_extracts_before_blank_line() {
        let doc = "First line.\nSecond line.\n\nSecond paragraph.";
        assert_eq!(
            first_paragraph(doc),
            "First line. Second line.",
            "should stop at blank line"
        );
    }

    #[test]
    fn first_paragraph_extracts_before_heading() {
        let doc = "Description here.\n# YAML configuration\nstuff";
        assert_eq!(first_paragraph(doc), "Description here.", "should stop at heading");
    }

    #[test]
    fn extract_yaml_examples_basic() {
        let doc = "Some filter.\n\n# YAML configuration\n\n```yaml\nfilter: test\nfoo: bar\n```\n\n# Example\nignored";
        assert_eq!(
            extract_yaml_examples(doc),
            vec!["filter: test\nfoo: bar".to_owned()],
            "should extract yaml block"
        );
    }

    #[test]
    fn extract_yaml_examples_accepts_short_heading() {
        let doc = "Some filter.\n\n# YAML\n\n```yaml\nfilter: test\n```\n";
        assert_eq!(
            extract_yaml_examples(doc),
            vec!["filter: test".to_owned()],
            "should extract short yaml heading"
        );
    }

    #[test]
    fn extract_yaml_examples_accepts_specific_headings() {
        let doc = "Some filter.\n\n# Single-field YAML\n\n```yaml\nfilter: test\nfield: model\n```\n\n# Multi-field YAML\n\n```yaml\nfilter: test\nfields: []\n```\n";
        assert_eq!(
            extract_yaml_examples(doc),
            vec![
                "filter: test\nfield: model".to_owned(),
                "filter: test\nfields: []".to_owned()
            ],
            "specific YAML headings should be extracted in order"
        );
    }

    #[test]
    fn extract_yaml_examples_missing() {
        let doc = "Some filter without yaml.";
        assert_eq!(
            extract_yaml_examples(doc),
            Vec::<String>::new(),
            "should return no examples when no yaml section exists"
        );
    }

    #[test]
    fn config_notes_skip_fenced_code_blocks() {
        let doc = "YAML configuration for a filter.\n\n```rust\nlet yaml = r#\"field: value\"#;\nassert!(true);\n```\n\nAccepts either single-field syntax or multi-field syntax.";
        assert_eq!(
            config_notes(doc),
            vec!["Accepts either single-field syntax or multi-field syntax.".to_owned()],
            "fenced doctests should not render as prose notes"
        );
    }

    #[test]
    fn field_docs_preserve_inline_link_continuation_lines() {
        let doc = "Protocol versions accepted during negotiation.\nEvery entry must be implemented by this build (present in\n[`protocol::SUPPORTED_VERSIONS`]). Defaults to the versions\nthis build implements.";
        assert_eq!(
            normalize_field_doc(doc),
            "Protocol versions accepted during negotiation. Every entry must be implemented by this build (present in [`protocol::SUPPORTED_VERSIONS`]). Defaults to the versions this build implements."
        );
    }

    #[test]
    fn field_docs_skip_reference_definition_lines() {
        let doc = "Uses [`Thing`] for validation.\n\n[`Thing`]: crate::Thing";
        assert_eq!(
            normalize_field_doc(doc),
            "Uses [`Thing`] for validation.",
            "reference definitions should not render inside table cells"
        );
    }

    #[test]
    fn is_config_struct_detects_config() {
        let source = "
            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct MyConfig {
                field: u64,
            }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        if let syn::Item::Struct(s) = &file.items[0] {
            assert!(is_config_struct(s), "should detect config struct");
        } else {
            panic!("expected struct");
        }
    }

    #[test]
    fn is_config_struct_rejects_non_config() {
        let source = "
            #[derive(Debug)]
            struct NotConfig {
                field: u64,
            }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        if let syn::Item::Struct(s) = &file.items[0] {
            assert!(!is_config_struct(s), "should reject non-config struct");
        } else {
            panic!("expected struct");
        }
    }

    #[test]
    fn extract_filter_name_finds_name() {
        let source = r#"
            impl HttpFilter for MyFilter {
                fn name(&self) -> &'static str {
                    "my_filter"
                }
            }
        "#;
        let file: syn::File = syn::parse_str(source).unwrap();
        if let syn::Item::Impl(imp) = &file.items[0] {
            assert_eq!(
                extract_filter_name(imp),
                Some("my_filter".to_owned()),
                "should extract filter name"
            );
        } else {
            panic!("expected impl");
        }
    }

    #[test]
    fn render_filter_doc_has_sections() {
        let result = render_filter_doc(&sample_filter_entry());
        assert!(
            result.starts_with("<!-- Generated by: cargo xtask generate-filter-docs -->"),
            "should have generation comment"
        );
        assert!(result.contains("# `timeout`"), "should have title");
        assert!(result.contains("## Configuration"), "should have config");
        assert!(result.contains("## Example"), "should have example");
    }

    #[test]
    fn render_filter_doc_has_field_row() {
        let result = render_filter_doc(&sample_filter_entry());
        assert!(
            result.contains("| `timeout_ms` | u64 | yes | Max time in milliseconds. |"),
            "should have field row"
        );
        assert!(
            result.contains("| Field | Type | Required | Description |"),
            "configuration table should describe requiredness, not fake defaults"
        );
        assert!(result.contains("filter: timeout"), "should have yaml");
    }

    #[test]
    fn render_filter_doc_strips_fenced_field_docs() {
        let mut entry = sample_filter_entry();
        entry.filter.fields[0].doc =
            "Maximum allowed time.\n\n```rust\nlet yaml = \"timeout_ms: 5000\";\n```\n\nUse `0 | 1` only in tests."
                .to_owned();

        let result = render_filter_doc(&entry);

        assert!(
            result.contains("| `timeout_ms` | u64 | yes | Maximum allowed time. Use `0 \\| 1` only in tests. |"),
            "field table rows should render prose without fenced doctests"
        );
        assert!(
            !result.contains("let yaml"),
            "field table rows should not include doctest body text"
        );
    }

    #[test]
    fn option_fields_render_optional() {
        let source = "
            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct MyConfig {
                /// Optional field.
                field: Option<String>,
            }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        let mut items = ModuleItems::new();
        parse_file_items(&file, &mut items);
        let filter = build_filter(&items, "test", Some("MyConfig"));
        assert_eq!(
            filter.fields[0].required,
            RequiredKind::No,
            "Option fields are optional"
        );
    }

    #[test]
    fn map_types_render_as_objects() {
        let ty: syn::Type = syn::parse_str("BTreeMap<String, String>").unwrap();
        assert_eq!(
            render_type(&ty, &BTreeMap::new()),
            "object<string, string>",
            "maps should render as YAML object shapes"
        );
    }

    #[test]
    fn untagged_wrapper_enum_types_render_wrapped_yaml_shapes() {
        let source = concat!(
            "#[derive(Deserialize)] #[serde(untagged)] enum LoadBalancerStrategy ",
            "{ Simple(SimpleStrategy), Parameterised(ParameterisedStrategy) }",
            "#[derive(Deserialize)] #[serde(rename_all = \"snake_case\")] enum SimpleStrategy ",
            "{ RoundRobin, LeastConnections, #[serde(rename = \"p2c\")] PowerOfTwoChoices }",
            "#[derive(Deserialize)] enum ParameterisedStrategy ",
            "{ #[serde(rename = \"consistent_hash\")] ConsistentHash(ConsistentHashOpts) }",
            "#[derive(Deserialize)] struct ConsistentHashOpts { header: Option<String> }",
            "#[derive(Deserialize)] #[serde(untagged)] enum Endpoint ",
            "{ Simple(String), Weighted { address: String, weight: u32 } }",
        );
        let file: syn::File = syn::parse_str(source).unwrap();
        let mut items = ModuleItems::new();
        parse_file_items(&file, &mut items);

        let strategy: syn::Type = syn::parse_str("LoadBalancerStrategy").unwrap();
        assert_eq!(
            render_type(&strategy, &items.enums),
            "`round_robin` \\| `least_connections` \\| `p2c` \\| `consistent_hash`"
        );
        let endpoints: syn::Type = syn::parse_str("Vec<Endpoint>").unwrap();
        assert_eq!(render_type(&endpoints, &items.enums), "(string \\| object)[]");
    }

    #[test]
    fn nested_config_fields_render_dotted_paths() {
        let source = "
            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct OuterConfig {
                /// Header settings.
                #[serde(default)]
                headers: HeaderConfig,
                /// Cluster entries.
                clusters: Vec<ClusterConfig>,
            }

            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct HeaderConfig {
                /// Method header.
                method: Option<String>,
            }

            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct ClusterConfig {
                /// Cluster name.
                name: String,
            }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        let mut items = ModuleItems::new();
        parse_file_items(&file, &mut items);

        let filter = build_filter(&items, "test", Some("OuterConfig"));
        let names: Vec<&str> = filter.fields.iter().map(|field| field.name.as_str()).collect();
        assert!(names.contains(&"headers.method"), "nested object field should render");
        assert!(names.contains(&"clusters[].name"), "nested list field should render");
    }

    #[test]
    fn flattened_fields_render_at_parent_path() {
        let source = "
            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct OuterConfig { routes: Vec<RouteConfig> }
            #[derive(Debug, Deserialize)]
            struct RouteConfig { #[serde(flatten)] path: PathMatch, cluster: String }
            #[derive(Debug, Deserialize)]
            #[serde(untagged)]
            enum PathMatch { Exact { path: String }, Prefix { path_prefix: String } }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        let mut items = ModuleItems::new();
        parse_file_items(&file, &mut items);

        let filter = build_filter(&items, "test", Some("OuterConfig"));
        let names: Vec<&str> = filter.fields.iter().map(|field| field.name.as_str()).collect();
        assert!(names.contains(&"routes[].path"), "flattened exact path should render");
        assert!(
            names.contains(&"routes[].path_prefix"),
            "flattened prefix path should render"
        );
        assert!(
            !names.contains(&"routes[].path.path"),
            "flattened field should not add an extra segment"
        );
    }

    #[test]
    fn flattened_enum_variant_fields_render_as_one_of() {
        let source = "
            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct OuterConfig { routes: Vec<RouteConfig> }
            #[derive(Debug, Deserialize)]
            struct RouteConfig { #[serde(flatten)] path: PathMatch, cluster: String }
            #[derive(Debug, Deserialize)]
            #[serde(untagged)]
            enum PathMatch { Exact { path: String }, Prefix { path_prefix: String } }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        let mut items = ModuleItems::new();
        parse_file_items(&file, &mut items);

        let filter = build_filter(&items, "test", Some("OuterConfig"));
        let path_field = filter.fields.iter().find(|field| field.name == "routes[].path");
        let prefix_field = filter.fields.iter().find(|field| field.name == "routes[].path_prefix");

        assert_eq!(
            path_field.map(|field| field.required),
            Some(RequiredKind::OneOf),
            "flattened exact path should be marked as one-of"
        );
        assert_eq!(
            prefix_field.map(|field| field.required),
            Some(RequiredKind::OneOf),
            "flattened prefix path should be marked as one-of"
        );
    }

    #[test]
    fn module_level_yaml_examples_are_included() {
        let source = "
            //! Module-level description.
            //!
            //! # YAML configuration
            //!
            //! ```yaml
            //! filter: module_filter
            //! answer: 42
            //! ```

            /// Public filter description.
            pub struct ModuleFilter;

            #[derive(Debug, Deserialize)]
            #[serde(deny_unknown_fields)]
            struct ModuleConfig {
                /// Answer value.
                answer: u64,
            }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        let mut items = ModuleItems::new();
        parse_file_items(&file, &mut items);

        let filter = build_filter(&items, "module_filter", Some("ModuleConfig"));

        assert_eq!(
            filter.yaml_examples,
            vec!["filter: module_filter\nanswer: 42".to_owned()],
            "module-level YAML examples should be rendered"
        );
    }

    #[test]
    fn render_reference_index_format() {
        let entries = vec![FilterEntry {
            crate_kind: "filters".to_owned(),
            category: "traffic_management".to_owned(),
            filter: FilterInfo {
                name: "timeout".to_owned(),
                description: "Enforces maximum latency.".to_owned(),
                extra_descriptions: vec![],
                config_notes: vec![],
                fields: vec![],
                yaml_examples: vec![],
            },
        }];
        let result = render_reference_index(&entries);
        assert!(result.contains("# Filter Reference"), "should have title");
        assert!(
            result.contains("### Traffic Management"),
            "should have category heading"
        );
        assert!(result.contains("[`timeout`](timeout.md)"), "should have filter link");
    }

    #[test]
    fn format_title_category() {
        assert_eq!(
            format_title("traffic_management"),
            "Traffic Management",
            "basic category"
        );
        assert_eq!(format_title("openai"), "OpenAI", "special case");
        assert_eq!(format_title("token_usage"), "Token Usage", "multi-word");
    }

    #[test]
    fn has_from_config_detects_factory() {
        let source = "
            impl MyFilter {
                fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
                    Ok(Box::new(Self))
                }
            }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        if let syn::Item::Impl(imp) = &file.items[0] {
            assert!(has_from_config_method(imp), "should detect from_config method");
        } else {
            panic!("expected impl");
        }
    }

    #[test]
    fn extract_config_type_from_impl() {
        let source = r#"
            impl MyFilter {
                fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
                    let cfg: MyFilterConfig = parse_filter_config("my_filter", config)?;
                    Ok(Box::new(Self { timeout: cfg.timeout }))
                }
            }
        "#;
        let file: syn::File = syn::parse_str(source).unwrap();
        if let syn::Item::Impl(imp) = &file.items[0] {
            assert_eq!(
                extract_config_type_name(imp),
                Some("MyFilterConfig".to_owned()),
                "should extract config type name"
            );
        } else {
            panic!("expected impl");
        }
    }

    #[test]
    fn extract_config_type_none_without_parse() {
        let source = "
            impl MyFilter {
                fn from_config(_config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
                    Ok(Box::new(Self))
                }
            }
        ";
        let file: syn::File = syn::parse_str(source).unwrap();
        if let syn::Item::Impl(imp) = &file.items[0] {
            assert_eq!(
                extract_config_type_name(imp),
                None,
                "should return None for config-less filters"
            );
        } else {
            panic!("expected impl");
        }
    }

    #[test]
    fn extract_filters_from_agentic() {
        let root = workspace_root();
        let agentic_dir = root.join("filters/src/agentic");
        if !agentic_dir.is_dir() {
            return;
        }
        let filters = extract_filters(&agentic_dir, &ModuleItems::new());
        assert!(!filters.is_empty(), "should extract at least one filter from agentic");
        let a2a = filters.iter().find(|f| f.name == "a2a");
        assert!(a2a.is_some(), "should find a2a filter");
        let a2a = a2a.unwrap();
        assert!(!a2a.description.is_empty(), "a2a should have a description");
    }

    #[test]
    fn discover_anchors_finds_nested_filters() {
        let root = workspace_root();
        let agentic_dir = root.join("filters/src/agentic");
        if !agentic_dir.is_dir() {
            return;
        }
        let anchors = discover_filter_anchors(&agentic_dir);
        let names: Vec<&str> = anchors.iter().map(|a| a.name.as_str()).collect();

        assert!(names.contains(&"a2a"), "should find a2a");
        assert!(names.contains(&"mcp"), "should find mcp");
    }

    #[test]
    fn mcp_docs_include_broker_fields() {
        let root = workspace_root();
        let agentic_dir = root.join("filters/src/agentic");
        if !agentic_dir.is_dir() {
            return;
        }
        let filters = extract_filters(&agentic_dir, &ModuleItems::new());
        let mcp = filters.iter().find(|f| f.name == "mcp").expect("mcp filter");
        let field_names: Vec<&str> = mcp.fields.iter().map(|f| f.name.as_str()).collect();
        assert!(
            field_names.contains(&"servers"),
            "mcp docs should include broker-mode servers field"
        );
    }

    /// Build a sample [`FilterEntry`] for rendering tests.
    fn sample_filter_entry() -> FilterEntry {
        FilterEntry {
            crate_kind: "filters".to_owned(),
            category: "traffic_management".to_owned(),
            filter: FilterInfo {
                name: "timeout".to_owned(),
                description: "Enforces maximum latency.".to_owned(),
                extra_descriptions: vec![],
                config_notes: vec![],
                fields: vec![FieldInfo {
                    name: "timeout_ms".to_owned(),
                    type_str: "u64".to_owned(),
                    doc: "Max time in milliseconds.".to_owned(),
                    required: RequiredKind::Yes,
                }],
                yaml_examples: vec!["filter: timeout\ntimeout_ms: 5000".to_owned()],
            },
        }
    }
}
