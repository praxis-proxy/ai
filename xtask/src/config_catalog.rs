// SPDX-License-Identifier: Apache-2.0
//! Generate and lint the machine-readable Praxis AI catalog fragment.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use praxis_config_catalog::SchemaId;

use super::filter_docs::*;

/// Generator-owned unions for factories that intentionally dispatch between
/// multiple serde shapes under one registered filter name. MCP's public
/// `mcp` factory selects broker mode when `servers` is present; this is a
/// source-level dispatch, not two registry entries. The emitted schema is a
/// typed `one_of`, preserving both configuration contracts without changing
/// runtime registration.
const MCP_FACTORY_UNIONS: &[(&str, &str, &str)] = &[("mcp", "McpConfig", "McpBrokerConfig")];

/// Arguments for catalog generation.
#[derive(Parser)]
pub(crate) struct GenerateArgs {}

/// Arguments for catalog linting.
#[derive(Parser)]
pub(crate) struct LintArgs {}

pub(crate) fn generate(_args: GenerateArgs) {
    let root = workspace_root();
    let path = root.join("docs/catalog/config-catalog.json");
    let bytes = render_catalog_impl(&root);
    fs::create_dir_all(path.parent().expect("catalog parent")).expect("create catalog directory");
    fs::write(&path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    println!("wrote {}", path.display());
}

pub(crate) fn lint(_args: LintArgs) {
    let root = workspace_root();
    let path = root.join("docs/catalog/config-catalog.json");
    let expected = render_catalog_impl(&root);
    match fs::read(&path) {
        Ok(actual) if actual == expected => println!("configuration catalog is up to date"),
        _ => {
            eprintln!("configuration catalog is stale or missing: {}", path.display());
            std::process::exit(1);
        },
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has workspace parent")
        .to_path_buf()
}

// -----------------------------------------------------------------------------
// Machine-readable catalog
// -----------------------------------------------------------------------------

/// Render the AI-owned catalog fragment from the same parsed source metadata
/// used by the filter documentation generator. Keeping this seam here makes
/// the human and machine-readable descriptions fail together when a config
/// shape changes.
pub(crate) fn render_catalog_impl(root: &Path) -> Vec<u8> {
    use praxis_config_catalog::{
        CatalogCompatibility, CatalogFormatVersion, CatalogFragment, ConfigExample, ConfigSchema, FilterDescriptor,
        ProducerComponent, ProducerInfo, Protocol, SchemaNode, VersionRequirement,
    };

    let shared = parse_shared_config_items(root);
    validate_factory_variants(root, MCP_FACTORY_UNIONS).unwrap_or_else(|error| panic!("{error}"));
    let entries = discover_all_filters(root, &shared);
    let registry = praxis_ai_filters::build_ai_registry();
    validate_yaml_examples(&entries).unwrap_or_else(|error| panic!("{error}"));
    let mut feature_profile = ai_feature_profile(root).expect("valid filters Cargo.toml");
    feature_profile.available.extend(
        ["apis", "experimental", "opentelemetry", "praxis-main"]
            .into_iter()
            .map(str::to_owned),
    );
    for entry in &entries {
        feature_profile
            .available
            .extend(required_features_for(root, &entry.filter));
    }
    feature_profile.default_enabled.insert("apis".to_owned());
    validate_registry_parity(root, &entries, &registry, &active_catalog_features());
    let mut fragment = CatalogFragment {
        format_version: CatalogFormatVersion { major: 1, minor: 0 },
        producer: ProducerInfo {
            component: ProducerComponent::Ai,
            package: "praxis-ai".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            source_revision: std::env::var("PRAXIS_SOURCE_REVISION")
                .ok()
                .filter(|revision| !revision.trim().is_empty()),
        },
        compatibility: CatalogCompatibility {
            requires_format_major: 1,
            requires_core: Some(VersionRequirement::from("^0.5.4")),
        },
        feature_profile,
        schemas: BTreeMap::new(),
        roots: Vec::new(),
        filters: Vec::new(),
        diagnostics: Vec::new(),
    };

    for entry in entries {
        let schema_id = SchemaId::from(format!("ai.filter.http.{}.{}", entry.category, entry.filter.name));
        let node = if entry.filter.name == "mcp" && entry.filter.variants.len() > 1 {
            let mut variants = Vec::new();
            for variant in &entry.filter.variants {
                let fields = CatalogSchemaBuilder::new(
                    &mut fragment.schemas,
                    &mut fragment.diagnostics,
                    &variant.source_items,
                    root,
                    &entry.filter.name,
                    Some(&variant.config_type_name),
                    entry.filter.name == "mcp",
                )
                .fields(&variant.raw_fields)
                .unwrap_or_else(|error| panic!("cannot represent AI filter {}: {error}", entry.filter.name));
                variants.push(SchemaNode::object(fields));
            }
            SchemaNode::one_of(variants)
        } else {
            let fields = CatalogSchemaBuilder::new(
                &mut fragment.schemas,
                &mut fragment.diagnostics,
                &entry.filter.source_items,
                root,
                &entry.filter.name,
                entry.filter.config_type_name.as_deref(),
                false,
            )
            .fields(&entry.filter.raw_fields)
            .unwrap_or_else(|error| panic!("cannot represent AI filter {}: {error}", entry.filter.name));
            SchemaNode::object(fields)
        };
        fragment.schemas.insert(
            schema_id.clone(),
            ConfigSchema {
                id: schema_id.clone(),
                title: entry.filter.name.clone(),
                description: entry.filter.description.clone(),
                shared: false,
                node: SchemaNode {
                    kind: node.kind,
                    title: None,
                    description: String::new(),
                    default: None,
                    examples: Vec::new(),
                    rules: Vec::new(),
                    sensitive: false,
                },
                producer: Some(fragment.producer.clone()),
            },
        );
        let name = entry.filter.name.clone();
        let required_features = required_features_for(root, &entry.filter);
        let capabilities = filter_capabilities(&entry.filter, registry.is_security_filter(&name));
        let source = source_location(root, &entry.filter);
        fragment.filters.push(FilterDescriptor {
            name: name.clone(),
            protocol: Protocol::Http,
            category: entry.category,
            description: entry.filter.description,
            config_schema: schema_id,
            required_features,
            capabilities,
            examples: entry
                .filter
                .yaml_examples
                .into_iter()
                .map(|yaml| ConfigExample { yaml })
                .collect(),
            source,
            producer: Some(fragment.producer.clone()),
        });
    }
    praxis_config_catalog::generator::render(fragment)
        .unwrap_or_else(|error| panic!("invalid generated AI catalog: {error}"))
}

/// A recursive converter from serde-facing Rust types to catalog schemas.
struct CatalogSchemaBuilder<'a> {
    schemas: &'a mut BTreeMap<SchemaId, praxis_config_catalog::ConfigSchema>,
    diagnostics: &'a mut Vec<praxis_config_catalog::CatalogDiagnostic>,
    source: &'a ModuleItems,
    root: &'a Path,
    owner: String,
    config_type_name: Option<String>,
    allow_json_value: bool,
    visiting: HashSet<String>,
}

impl<'a> CatalogSchemaBuilder<'a> {
    fn new(
        schemas: &'a mut BTreeMap<SchemaId, praxis_config_catalog::ConfigSchema>,
        diagnostics: &'a mut Vec<praxis_config_catalog::CatalogDiagnostic>,
        source: &'a ModuleItems,
        root: &'a Path,
        owner: &str,
        config_type_name: Option<&str>,
        allow_json_value: bool,
    ) -> Self {
        Self {
            schemas,
            diagnostics,
            source,
            root,
            owner: owner.to_owned(),
            config_type_name: config_type_name.map(str::to_owned),
            allow_json_value,
            visiting: HashSet::new(),
        }
    }

    fn fields(&mut self, fields: &[RawField]) -> Result<Vec<praxis_config_catalog::ObjectField>, String> {
        if let Some(config) = self
            .config_type_name
            .as_deref()
            .and_then(|name| self.source.configs.iter().find(|config| config.name == name))
            && let Some(try_from) = config.try_from.as_deref()
        {
            return Err(format!(
                "catalog_error=unsupported_try_from target={} type={try_from}",
                self.owner
            ));
        }
        fields
            .iter()
            .map(|field| {
                Ok(praxis_config_catalog::ObjectField {
                    serialized_name: field.name.clone(),
                    aliases: field.aliases.clone(),
                    schema: self.field_node(field)?,
                    required: matches!(field.requirement_hint, RequirementHint::Normal)
                        && !field.has_default
                        && !is_option_type(&field.ty),
                    flattened: field.flatten,
                })
            })
            .collect()
    }

    fn field_node(&mut self, field: &RawField) -> Result<praxis_config_catalog::SchemaNode, String> {
        let mut node = if let Some(kind) = custom_deserializer_kind(field)? {
            praxis_config_catalog::SchemaNode::simple(kind)
        } else {
            self.type_node(&field.ty)?
        };
        node.description = field.doc.clone();
        let ty = &field.ty;
        node.sensitive = quote::quote!(#ty).to_string().to_ascii_lowercase().contains("secret");
        if let Some(default_path) = field.default_path.as_deref() {
            match evaluate_default(self.root, default_path) {
                Some(value) => node.default = Some(value),
                None => self.diagnostics.push(praxis_config_catalog::CatalogDiagnostic {
                    severity: praxis_config_catalog::DiagnosticSeverity::Warning,
                    code: "unevaluable_default".to_owned(),
                    target: format!("{}.{}", self.owner, field.name),
                    message: format!("default function `{default_path}` could not be evaluated safely"),
                }),
            }
        }
        Ok(node)
    }

    fn type_node(&mut self, ty: &syn::Type) -> Result<praxis_config_catalog::SchemaNode, String> {
        use praxis_config_catalog::{SchemaKind, SchemaNode};
        match ty {
            syn::Type::Reference(reference) => self.type_node(&reference.elem),
            syn::Type::Array(array) => Ok(SchemaNode::array(self.type_node(&array.elem)?)),
            syn::Type::Slice(slice) => Ok(SchemaNode::array(self.type_node(&slice.elem)?)),
            syn::Type::Tuple(tuple) if tuple.elems.is_empty() => Ok(SchemaNode::simple(SchemaKind::Null)),
            syn::Type::Tuple(_) => Err(format!("unsupported tuple type `{}`", quote::quote!(#ty))),
            syn::Type::Path(path) => self.path_node(path),
            _ => Err(format!("unsupported Rust type `{}`", quote::quote!(#ty))),
        }
    }

    fn path_node(&mut self, path: &syn::TypePath) -> Result<praxis_config_catalog::SchemaNode, String> {
        use praxis_config_catalog::{SchemaKind, SchemaNode};
        let segment = path.path.segments.last().ok_or_else(|| "empty type path".to_owned())?;
        let name = segment.ident.to_string();
        let args = type_args(&segment.arguments);
        match name.as_str() {
            "Option" | "Box" | "Arc" | "Rc" | "Cow" | "Zeroizing" => args
                .last()
                .ok_or_else(|| format!("missing inner type for `{name}`"))
                .and_then(|ty| self.type_node(ty)),
            "Vec" | "VecDeque" | "SmallVec" => args
                .last()
                .ok_or_else(|| format!("missing item type for `{name}`"))
                .and_then(|ty| self.type_node(ty))
                .map(SchemaNode::array),
            "BTreeMap" | "HashMap" | "IndexMap" => {
                let key = args.first().ok_or_else(|| format!("missing map key for `{name}`"))?;
                let key_name = quote::quote!(#key).to_string();
                if key_name != "String" && key_name != "str" {
                    return Err(format!("non-string map key `{key_name}`"));
                }
                args.get(1)
                    .ok_or_else(|| format!("missing map value for `{name}`"))
                    .and_then(|ty| self.type_node(ty))
                    .map(SchemaNode::map)
            },
            "bool" => Ok(SchemaNode::simple(SchemaKind::Boolean)),
            "f32" | "f64" => Ok(SchemaNode::simple(SchemaKind::Number)),
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => {
                Ok(SchemaNode::simple(SchemaKind::Integer))
            },
            "String" | "PathBuf" | "Url" | "Uri" | "HeaderName" | "HeaderValue" | "IpAddr" | "Ipv4Addr"
            | "Ipv6Addr" | "SocketAddr" | "SocketAddrV4" | "SocketAddrV6" | "Duration" | "Regex" | "SecretString" => {
                Ok(SchemaNode::simple(SchemaKind::String))
            },
            // This Core enum is re-exported through `praxis_filter` and its
            // source module is not part of the filter-local AST scope.
            "OnInvalidBehavior" => Ok(SchemaNode::enum_strings(vec![
                "continue".to_owned(),
                "reject".to_owned(),
                "error".to_owned(),
            ])),
            "Value" if self.allow_json_value => {
                self.ensure_json_value_schema();
                Ok(SchemaNode::reference("ai.type.json_value".to_owned()))
            },
            "Value" | "Condition" | "ResponseCondition" => Err(format!(
                "opaque serde value `{name}` has no portable catalog representation"
            )),
            _ if self.source.structs.contains_key(&name) => self.struct_node(&name),
            _ if self.source.enums.contains_key(&name) => self.enum_node(&name),
            _ => Err(format!("unresolved configuration type `{name}`")),
        }
    }

    fn ensure_json_value_schema(&mut self) {
        use praxis_config_catalog::{SchemaKind, SchemaNode};
        let id = SchemaId::from("ai.type.json_value");
        if self.schemas.contains_key(&id) {
            return;
        }
        let reference = || SchemaNode::reference(id.0.clone());
        let node = SchemaNode::one_of(vec![
            SchemaNode::simple(SchemaKind::Null),
            SchemaNode::simple(SchemaKind::Boolean),
            SchemaNode::simple(SchemaKind::Number),
            SchemaNode::simple(SchemaKind::String),
            SchemaNode::array(reference()),
            SchemaNode::map(reference()),
        ]);
        self.schemas.insert(
            id.clone(),
            praxis_config_catalog::ConfigSchema {
                id,
                title: "JSON value".to_owned(),
                description: "Any JSON value accepted by the MCP tool schema fields.".to_owned(),
                shared: false,
                node,
                producer: None,
            },
        );
    }

    fn struct_node(&mut self, name: &str) -> Result<praxis_config_catalog::SchemaNode, String> {
        if let Some(config) = self.source.structs.get(name)
            && let Some(try_from) = config.try_from.as_deref()
        {
            return Err(format!(
                "catalog_error=unsupported_try_from target={name} type={try_from}"
            ));
        }
        if name == "ProviderConfig" && self.source.structs.contains_key("NemoConfig") {
            let mut fields = self
                .source
                .structs
                .get("NemoConfig")
                .ok_or_else(|| "missing NemoConfig".to_owned())?
                .fields
                .clone();
            fields.insert(
                0,
                RawField {
                    name: "type".to_owned(),
                    aliases: Vec::new(),
                    ty: syn::parse_str("String").expect("literal type parses"),
                    doc: "Provider discriminator.".to_owned(),
                    has_default: false,
                    default_path: None,
                    deserialize_with: None,
                    flatten: false,
                    requirement_hint: RequirementHint::Normal,
                },
            );
            let mut node = praxis_config_catalog::SchemaNode::object(self.fields(&fields)?);
            if let praxis_config_catalog::SchemaKind::Object { fields, .. } = &mut node.kind
                && let Some(discriminator) = fields.first_mut()
            {
                discriminator.schema =
                    praxis_config_catalog::SchemaNode::literal(serde_json::Value::String("nemo".to_owned()));
            }
            return Ok(node);
        }
        let id = SchemaId::from(format!("ai.type.{name}"));
        if self.visiting.contains(name) {
            return Ok(praxis_config_catalog::SchemaNode::reference(id.0.clone()));
        }
        if !self.schemas.contains_key(&id) {
            let fields = self
                .source
                .structs
                .get(name)
                .ok_or_else(|| format!("missing struct `{name}`"))?
                .fields
                .clone();
            self.visiting.insert(name.to_owned());
            let object = praxis_config_catalog::SchemaNode::object(self.fields(&fields)?);
            self.visiting.remove(name);
            self.schemas.insert(
                id.clone(),
                praxis_config_catalog::ConfigSchema {
                    id: id.clone(),
                    title: name.to_owned(),
                    description: String::new(),
                    shared: false,
                    node: object,
                    producer: None,
                },
            );
        }
        Ok(praxis_config_catalog::SchemaNode::reference(id.0.clone()))
    }

    fn enum_node(&mut self, name: &str) -> Result<praxis_config_catalog::SchemaNode, String> {
        let info = self
            .source
            .enums
            .get(name)
            .ok_or_else(|| format!("missing enum `{name}`"))?;
        if !info.untagged
            && info.tag.is_none()
            && info
                .variant_shapes
                .iter()
                .all(|shape| matches!(shape, EnumVariantShape::Unit))
        {
            return Ok(praxis_config_catalog::SchemaNode::enum_strings(info.variants.clone()));
        }
        let variants = info
            .variant_shapes
            .iter()
            .zip(info.variants.iter())
            .enumerate()
            .map(|(index, (shape, label))| match shape {
                EnumVariantShape::Unit if info.tag.is_none() => Ok(praxis_config_catalog::SchemaNode::literal(
                    serde_json::Value::String(label.clone()),
                )),
                EnumVariantShape::Unit => self.tagged_variant_node(info, index, label, Vec::new()),
                EnumVariantShape::Unnamed(ty) if info.content.is_some() => {
                    let field = praxis_config_catalog::ObjectField {
                        serialized_name: info.content.clone().unwrap_or_default(),
                        aliases: Vec::new(),
                        schema: self.type_node(ty)?,
                        required: true,
                        flattened: false,
                    };
                    self.tagged_variant_node(info, index, label, vec![field])
                },
                EnumVariantShape::Unnamed(ty) => self.type_node(ty),
                EnumVariantShape::Named => {
                    let fields = info.variant_fields.get(index).cloned().unwrap_or_default();
                    let fields = self.fields(&fields)?;
                    self.tagged_variant_node(info, index, label, fields)
                },
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(praxis_config_catalog::SchemaNode {
            kind: praxis_config_catalog::SchemaKind::OneOf {
                variants,
                discriminator: info.tag.clone(),
            },
            title: None,
            description: String::new(),
            default: None,
            examples: Vec::new(),
            rules: Vec::new(),
            sensitive: false,
        })
    }

    fn tagged_variant_node(
        &mut self,
        info: &EnumInfo,
        _index: usize,
        label: &str,
        mut fields: Vec<praxis_config_catalog::ObjectField>,
    ) -> Result<praxis_config_catalog::SchemaNode, String> {
        let Some(tag) = info.tag.as_ref() else {
            return Ok(praxis_config_catalog::SchemaNode::literal(serde_json::Value::String(
                label.to_owned(),
            )));
        };
        fields.insert(
            0,
            praxis_config_catalog::ObjectField {
                serialized_name: tag.clone(),
                aliases: Vec::new(),
                schema: praxis_config_catalog::SchemaNode::literal(serde_json::Value::String(label.to_owned())),
                required: true,
                flattened: false,
            },
        );
        let mut node = praxis_config_catalog::SchemaNode::object(fields);
        if info.content.is_some() {
            node.description = "Adjacent serde-tagged variant.".to_owned();
        }
        Ok(node)
    }
}

/// Convert the small set of AI custom deserializers whose wire shape is known.
/// Unknown custom deserializers fail generation rather than being represented
/// by the Rust field type, which may not describe the YAML contract.
fn custom_deserializer_kind(field: &RawField) -> Result<Option<praxis_config_catalog::SchemaKind>, String> {
    let Some(path) = field.deserialize_with.as_deref() else {
        return Ok(None);
    };
    let name = path.rsplit("::").next().unwrap_or(path);
    match name {
        // Duration is parsed from a human-readable string at the YAML boundary.
        "deserialize_duration" => Ok(Some(praxis_config_catalog::SchemaKind::String)),
        // These preserve the underlying serde shape already represented by the
        // field type; the custom function only broadens accepted input forms.
        "nullable_vec" | "deserialize_metadata_object" => Ok(None),
        _ => Err(format!(
            "catalog_error=unsupported_deserializer field={} deserializer={path}",
            field.name
        )),
    }
}

/// Parse examples during generation so malformed documentation cannot enter
/// the machine-readable catalog as if it were usable configuration.
fn validate_yaml_examples(entries: &[FilterEntry]) -> Result<(), String> {
    for entry in entries {
        for (index, example) in entry.filter.yaml_examples.iter().enumerate() {
            serde_yaml::from_str::<serde_yaml::Value>(example).map_err(|error| {
                format!(
                    "catalog_error=invalid_example filter={} index={} error={error}",
                    entry.filter.name, index
                )
            })?;
        }
    }
    Ok(())
}

/// Safely evaluate only literal defaults. Runtime-dependent defaults produce a
/// warning diagnostic and remain without a literal value.
fn evaluate_default(root: &Path, function: &str) -> Option<serde_json::Value> {
    for directory in [root.join("apis/src"), root.join("filters/src")] {
        for path in collect_rs_files(&directory) {
            let Ok(source) = fs::read_to_string(path) else { continue };
            let marker = format!("fn {function}");
            let Some(start) = source.find(&marker) else { continue };
            let tail = &source[start..];
            let body = &tail[..tail.find('}').unwrap_or(tail.len())];
            if body.contains("true") {
                return Some(serde_json::Value::Bool(true));
            }
            if body.contains("false") {
                return Some(serde_json::Value::Bool(false));
            }
            if let Some(value) = body.split('"').nth(1) {
                return Some(serde_json::Value::String(value.to_owned()));
            }
            if let Some(number) = body
                .split(|character: char| !character.is_ascii_digit() && character != '-')
                .find(|value| !value.is_empty() && *value != "-")
                && let Ok(value) = number.parse::<i64>()
            {
                return Some(serde_json::Value::from(value));
            }
        }
    }
    None
}

fn ai_feature_profile(root: &Path) -> Result<praxis_config_catalog::FeatureProfile, String> {
    let path = root.join("filters/Cargo.toml");
    let Ok(source) = fs::read_to_string(&path) else {
        return Err(format!("missing {}", path.display()));
    };
    feature_profile_from_manifest(&source)
}

fn feature_profile_from_manifest(source: &str) -> Result<praxis_config_catalog::FeatureProfile, String> {
    let mut profile = praxis_config_catalog::FeatureProfile::default();
    let Ok(document) = toml::from_str::<toml::Table>(source) else {
        return Err("invalid TOML feature manifest".to_owned());
    };
    let Some(features) = document.get("features").and_then(toml::Value::as_table) else {
        return Err("missing [features] table".to_owned());
    };
    if let Some(values) = features.get("default").and_then(toml::Value::as_array) {
        profile
            .default_enabled
            .extend(values.iter().filter_map(toml::Value::as_str).map(str::to_owned));
    }
    for (name, _value) in features {
        if name != "default" {
            profile.available.insert(name.clone());
        }
    }
    Ok(profile)
}

fn required_features_for(root: &Path, filter: &FilterInfo) -> BTreeSet<String> {
    let registrations = fs::read_to_string(root.join("filters/src/register.rs")).unwrap_or_default();
    required_features_for_source(&registrations, &filter.name)
}

fn required_features_for_source(registrations: &str, filter_name: &str) -> BTreeSet<String> {
    let mut features = BTreeSet::new();
    let registration_lines: Vec<&str> = registrations.lines().collect();
    let mut function_feature = None;
    let mut depth = 0usize;
    for line in registration_lines {
        let trimmed = line.trim();
        if let Some(feature) = trimmed
            .strip_prefix("#[cfg(feature = \"")
            .and_then(|line| line.strip_suffix("\")]"))
        {
            function_feature = Some(feature.to_owned());
        }
        if trimmed.starts_with("fn ") {
            depth = 0;
        }
        if trimmed.contains(&format!("\"{filter_name}\""))
            && let Some(feature) = &function_feature
        {
            features.insert(feature.clone());
        }
        depth = depth.saturating_add(line.matches('{').count());
        depth = depth.saturating_sub(line.matches('}').count());
        if function_feature.is_some() && (trimmed == ")" || trimmed == ");") {
            function_feature = None;
        }
        if depth == 0 && trimmed.contains('}') {
            function_feature = None;
        }
    }
    features
}

fn source_location(root: &Path, filter: &FilterInfo) -> Option<praxis_config_catalog::SourceLocation> {
    let path = filter.source_path.as_ref()?;
    let source = fs::read_to_string(path).ok()?;
    let line = source.lines().position(|line| line.contains("HttpFilter for"));
    Some(praxis_config_catalog::SourceLocation {
        path: relative_path(root, path).to_string_lossy().replace('\\', "/"),
        line: line.map(|line| line as u32 + 1),
    })
}

fn filter_capabilities(filter: &FilterInfo, is_security: bool) -> praxis_config_catalog::FilterCapabilities {
    let source = filter
        .source_path
        .as_ref()
        .and_then(|path| fs::read_to_string(path).ok())
        .unwrap_or_default();
    capabilities_from_source(&source, is_security)
}

fn capabilities_from_source(source: &str, is_security: bool) -> praxis_config_catalog::FilterCapabilities {
    praxis_config_catalog::FilterCapabilities {
        security_class: if is_security {
            praxis_config_catalog::SecurityClass::Security
        } else {
            praxis_config_catalog::SecurityClass::Standard
        },
        request_headers: source.contains("fn on_request("),
        request_body: source.contains("fn on_request_body("),
        response_headers: source.contains("fn on_response("),
        response_body: source.contains("fn on_response_body("),
        terminal: source.contains("TerminalResponse") || source.contains("StreamingTerminalResponse"),
    }
}

fn active_catalog_features() -> BTreeSet<String> {
    let feature_names = [
        "apis",
        #[cfg(feature = "catalog-http-callout")]
        "http-callout-filter",
        #[cfg(feature = "catalog-azure-ad")]
        "azure-ad-filter",
        #[cfg(feature = "catalog-gcp-adc")]
        "gcp-adc-filter",
        #[cfg(feature = "catalog-token-rate-limit")]
        "token-rate-limit-filter",
    ];
    feature_names.into_iter().map(str::to_owned).collect()
}

fn validate_registry_parity(
    root: &Path,
    entries: &[FilterEntry],
    registry: &praxis_filter::FilterRegistry,
    active_features: &BTreeSet<String>,
) {
    let discovered: BTreeSet<String> = entries
        .iter()
        .filter(|entry| required_features_for(root, &entry.filter).is_subset(active_features))
        .map(|entry| entry.filter.name.clone())
        .collect();
    let core: BTreeSet<String> = praxis_filter::FilterRegistry::with_builtins()
        .available_filters()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let registered: BTreeSet<String> = registry
        .available_filters()
        .into_iter()
        .filter(|name| !core.contains(*name))
        .map(str::to_owned)
        .collect();
    assert_eq!(
        registered, discovered,
        "default AI registry and catalog discovery differ"
    );
    for entry in entries {
        for feature in required_features_for(root, &entry.filter) {
            assert!(
                active_features.contains(&feature)
                    || ai_feature_profile(root).is_ok_and(|profile| profile.available.contains(&feature)),
                "unknown required feature {feature}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use praxis_config_catalog::{
        CatalogCompatibility, CatalogFormatVersion, CatalogFragment, ProducerComponent, ProducerInfo, SchemaId,
    };

    use super::*;

    #[test]
    fn capabilities_follow_http_filter_callbacks() {
        let capabilities = capabilities_from_source(
            "impl HttpFilter for Example { fn on_request(&self) {} fn on_response_body(&self) {} FilterAction::TerminalResponse }",
            false,
        );
        assert!(capabilities.request_headers);
        assert!(capabilities.response_body);
        assert!(!capabilities.request_body);
        assert!(!capabilities.response_headers);
        assert!(capabilities.terminal);
    }

    #[test]
    fn custom_deserializers_are_explicitly_allowlisted() {
        let field = RawField {
            name: "timeout".to_owned(),
            aliases: Vec::new(),
            ty: syn::parse_str("Duration").unwrap(),
            doc: String::new(),
            has_default: false,
            default_path: None,
            deserialize_with: Some("deserialize_duration".to_owned()),
            flatten: false,
            requirement_hint: RequirementHint::Normal,
        };
        assert!(matches!(
            custom_deserializer_kind(&field),
            Ok(Some(praxis_config_catalog::SchemaKind::String))
        ));
        let mut unsupported = field;
        unsupported.deserialize_with = Some("custom::unknown".to_owned());
        let error = custom_deserializer_kind(&unsupported).err().unwrap_or_default();
        assert!(error.contains("catalog_error=unsupported_deserializer"));
    }

    #[test]
    fn literal_defaults_are_evaluated_without_running_code() {
        let root = workspace_root();
        assert_eq!(
            evaluate_default(&root, "default_true"),
            Some(serde_json::Value::Bool(true))
        );
        assert_eq!(evaluate_default(&root, "does_not_exist"), None);
    }

    fn fragment(component: ProducerComponent, package: &str, version: &str) -> CatalogFragment {
        CatalogFragment {
            format_version: CatalogFormatVersion { major: 1, minor: 0 },
            producer: ProducerInfo {
                component,
                package: package.to_owned(),
                version: version.to_owned(),
                source_revision: None,
            },
            compatibility: CatalogCompatibility {
                requires_format_major: 1,
                requires_core: None,
            },
            feature_profile: Default::default(),
            schemas: BTreeMap::new(),
            roots: Vec::new(),
            filters: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn registration_features_associate_with_adjacent_functions() {
        let source = r#"
            #[cfg(feature = "gated-a")]
            fn register_a() { register!("a"); }
            fn register_b() { register!("b"); }
            #[cfg(feature = "gated-c")]
            fn register_c() { register!("c"); }
        "#;
        assert_eq!(
            required_features_for_source(source, "a"),
            ["gated-a".to_owned()].into_iter().collect()
        );
        assert!(required_features_for_source(source, "b").is_empty());
        assert_eq!(
            required_features_for_source(source, "c"),
            ["gated-c".to_owned()].into_iter().collect()
        );
    }

    #[test]
    fn non_string_map_keys_fail_closed() {
        let ty: syn::Type = syn::parse_str("BTreeMap<u32, String>").unwrap();
        let mut schemas = BTreeMap::new();
        let mut diagnostics = Vec::new();
        let source = ModuleItems::new();
        let mut builder = CatalogSchemaBuilder::new(
            &mut schemas,
            &mut diagnostics,
            &source,
            Path::new("."),
            "test",
            None,
            false,
        );
        assert!(builder.type_node(&ty).is_err());
    }

    #[test]
    fn generated_ai_artifact_has_nemo_schema_and_provenance() {
        let root = workspace_root();
        let tracked = fs::read(root.join("docs/catalog/config-catalog.json")).expect("tracked catalog");
        let rendered = render_catalog_impl(&root);
        if active_catalog_features().len() == 1 {
            assert_eq!(tracked, rendered, "tracked catalog must equal deterministic render");
        }
        let fragment: CatalogFragment = serde_json::from_slice(&rendered).expect("catalog JSON");
        fragment.validate().expect("catalog validates");
        let schema = fragment
            .schemas
            .get(&SchemaId::from("ai.filter.http.guardrails.ai_guardrails"))
            .expect("guardrails schema");
        let praxis_config_catalog::SchemaKind::Object { fields, .. } = &schema.node.kind else {
            panic!("guardrails object")
        };
        let names: BTreeSet<_> = fields.iter().map(|field| field.serialized_name.as_str()).collect();
        assert_eq!(names, ["provider", "phase"].into_iter().collect());
        let provider = &fields
            .iter()
            .find(|field| field.serialized_name == "provider")
            .unwrap()
            .schema;
        let praxis_config_catalog::SchemaKind::Object { fields, .. } = &provider.kind else {
            panic!("provider object")
        };
        let provider_names: BTreeSet<_> = fields.iter().map(|field| field.serialized_name.as_str()).collect();
        assert_eq!(
            provider_names,
            ["type", "endpoint", "allow_private_endpoint", "model", "timeout_ms"]
                .into_iter()
                .collect()
        );
        let discriminator = &fields
            .iter()
            .find(|field| field.serialized_name == "type")
            .unwrap()
            .schema
            .kind;
        assert!(matches!(discriminator, praxis_config_catalog::SchemaKind::Literal { value } if value == "nemo"));
        assert!(fragment.filters.iter().all(|filter| filter.producer.is_some()));
        assert!(fragment.schemas.values().all(|schema| schema.producer.is_some()));
        assert!(
            fragment
                .filters
                .iter()
                .all(|filter| filter.source.as_ref().is_some_and(|source| source.line.is_some()))
        );
    }

    #[test]
    fn mcp_factory_union_is_typed_and_provenanced() {
        let root = workspace_root();
        let bytes = render_catalog_impl(&root);
        let fragment: CatalogFragment = serde_json::from_slice(&bytes).expect("catalog JSON");
        let schema = fragment
            .schemas
            .get(&SchemaId::from("ai.filter.http.agentic.mcp"))
            .expect("MCP schema");
        let praxis_config_catalog::SchemaKind::OneOf { variants, .. } = &schema.node.kind else {
            panic!("MCP must expose its two factory shapes as one_of")
        };
        assert_eq!(variants.len(), 2);
        let names: BTreeSet<&str> = variants
            .iter()
            .filter_map(|variant| match &variant.kind {
                praxis_config_catalog::SchemaKind::Object { fields, .. } => {
                    fields.first().map(|field| field.serialized_name.as_str())
                },
                _ => None,
            })
            .collect();
        assert!(names.contains("header_validation"));
        assert!(names.contains("cache_scope"));
        assert_eq!(
            schema.producer.as_ref().map(|p| &p.component),
            Some(&ProducerComponent::Ai)
        );
        assert!(
            fragment
                .schemas
                .get(&SchemaId::from("ai.type.json_value"))
                .is_some_and(|schema| schema.producer.is_some())
        );
    }

    #[test]
    fn feature_profile_fixture_is_strict() {
        let profile = feature_profile_from_manifest("[features]\ndefault=[\"apis\"]\napis=[]\nexperimental=[]\n")
            .expect("valid TOML");
        assert!(profile.available.contains("apis"));
        assert!(profile.default_enabled.contains("apis"));
        assert!(feature_profile_from_manifest("[features\n").is_err());
    }

    #[test]
    fn core_artifact_and_ai_fragment_merge() {
        let root = workspace_root();
        let core_bytes = fs::read(root.join("../praxis/docs/catalog/config-catalog.json")).expect("Core catalog");
        let core: CatalogFragment = serde_json::from_slice(&core_bytes).expect("Core JSON");
        let ai: CatalogFragment = serde_json::from_slice(&render_catalog_impl(&root)).expect("AI JSON");
        let merged = praxis_config_catalog::merge::merge(core, vec![ai]).expect("Core and AI merge");
        assert!(!merged.fragment.filters.is_empty());
        assert!(merged.producers.len() >= 2);
        assert!(!merged.fragment.roots.is_empty(), "Core roots must survive AI merge");
        assert!(merged.fragment.filters.windows(2).all(|pair| {
            (&pair[0].protocol, &pair[0].category, &pair[0].name)
                <= (&pair[1].protocol, &pair[1].category, &pair[1].name)
        }));
        assert!(merged.fragment.filters.iter().all(|filter| filter.producer.is_some()));
        assert!(merged.fragment.schemas.values().all(|schema| schema.producer.is_some()));
    }

    #[test]
    fn incompatible_core_requirement_has_stable_error() {
        let mut ai = fragment(ProducerComponent::Ai, "praxis-ai", "0.3.0");
        ai.compatibility.requires_core = Some("^9.0.0".into());
        let error = praxis_config_catalog::merge::merge(
            fragment(ProducerComponent::Core, "praxis-proxy-core", "0.5.4"),
            vec![ai],
        )
        .expect_err("incompatible Core requirement must fail closed");
        assert!(matches!(
            error,
            praxis_config_catalog::merge::MergeError::CoreRequirement(_)
        ));
    }
}
