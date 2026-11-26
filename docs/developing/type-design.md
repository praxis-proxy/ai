# Type Design

Make invalid states unrepresentable: let the type system and serde enforce
constraints at parse time, not at runtime.

- **Enums over strings for fixed value sets.** Do not use `String` when the
  valid values are known. A `#[serde(rename_all = "snake_case")]` enum gets
  serde validation and removes manual string matching:

  ```rust
  // Bad:
  mode: String, // "per_ip" | "global"

  // Good:
  #[derive(Deserialize)]
  #[serde(rename_all = "snake_case")]
  enum Mode { PerIp, Global }
  ```

- **Structs over maps for known keys.** Do not deserialize config into
  `BTreeMap`/`HashMap` when the key set is known; maps silently absorb unknown
  keys. Use a struct with `#[serde(deny_unknown_fields)]`, and keep maps for
  open key sets such as user-defined header names.

- **Enums over multiple `Option<T>` fields.** When exactly one of N fields
  must be set, use an N-variant enum. Serde's
  `#[serde(rename_all = "snake_case")]` with external tagging maps it to YAML.

- **`#[serde(default)]` over `Option<T>` with `unwrap_or`.** If an
  `Option<T>` is always resolved with `.unwrap_or(DEFAULT)`, use the concrete
  type with `#[serde(default = "fn_name")]`.

- **`#[serde(try_from)]` for constrained numerics.** When a numeric field
  accepts only specific values (for example HTTP redirect status
  301/302/307/308), define an enum with `TryFrom<u16>` and
  `#[serde(try_from = "u16")]` so validation happens at parse time.

- **`#[serde(deny_unknown_fields)]` by default.** Apply it to every config
  struct that does not intentionally accept arbitrary keys (extension points),
  so typos fail at parse time.
