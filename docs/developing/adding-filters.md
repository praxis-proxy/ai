# Adding a Built-in AI Filter

Review the [extensions guide](../filters/extensions.md)
first.

1. Create the filter module:
   - Provider API filters: `apis/src/<provider>/`
   - Cross-provider filters: `filters/src/<category>/`
2. Implement `HttpFilter` (from `praxis-filter`). Add a
   `from_config` factory that deserializes a
   `serde_yaml::Value` into your config struct.
3. Register it in `praxis_ai_filters::register_ai_filters`
   (in `filters/src/register.rs`).
4. Add unit tests and doctests.
5. Add an example config in the appropriate category
   under `examples/configs/`.
6. Add a functional integration test in
   `tests/integration/tests/suite/`.
7. Update `examples/README.md` to list any new or
   renamed example configs.

All testing requirements from
[CONTRIBUTING.md](../../CONTRIBUTING.md#testing) apply. A
feature without tests and an example is not complete.
