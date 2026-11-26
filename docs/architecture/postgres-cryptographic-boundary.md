# PostgreSQL Cryptographic Boundary

The `openai_responses_store` and `openai_conversations` filters can persist
state to PostgreSQL. A PostgreSQL connection performs cryptography in two
places with two providers. This page defines that boundary: what runs inside
the platform TLS provider, what runs application-side, and how the
certificate-authentication compliance profile keeps password cryptography off
the connection path.

## Two call paths, two providers

The two paths are selected at different moments, by different actors.

### 1. TLS transport (inside the boundary)

The TLS handshake (key exchange, certificate chain verification, hostname
verification and, when configured, **client-certificate authentication**) runs
inside the platform TLS library. SQLx selects it through its `tls-native-tls`
feature, which binds [native-tls] to the operating system's provider:
**OpenSSL on Linux** and **Security.framework (Secure Transport) on macOS**.
The PostgreSQL profiles in `store-backends/Cargo.toml` turn it on:

```toml
[features]
_postgres = ["dep:sqlx", "sqlx/postgres", "sqlx/tls-native-tls"]
postgres = ["_postgres", "sqlx/postgres-password-auth"]
postgres-cert-auth = ["_postgres"]
```

With a client certificate, the client proves its identity during the TLS
handshake, so the authentication crypto of a cert-authenticated connection
stays entirely inside the platform TLS boundary and never touches
application-side password code.

The assurance of this path is the assurance of the deployed platform and its
validated configuration. On a RHEL host, OpenSSL follows the system-wide crypto
policy and can operate a FIPS-validated module when the host runs in FIPS mode;
on macOS the provider is Security.framework. Scope any compliance claim to the
platform you ship and its validated configuration.

Call path (client side):

```text
PostgresResponseStore::new
  -> pg_connect_options(database_url, tls)      # store-backends/src/postgres.rs
       -> PgConnectOptions::ssl_mode / ssl_root_cert
          / ssl_client_cert / ssl_client_key
  -> PgPoolOptions::connect_with(options)
       -> sqlx-postgres TLS upgrade
            -> native-tls handshake  (OpenSSL / Security.framework:
               key exchange, AEAD, cert verify)
                 -> [optional] client certificate presented here
```

### 2. Password authentication (outside the boundary)

*After* the handshake, the PostgreSQL **server** decides how to authenticate.
It matches a `pg_hba.conf` rule against `(type, database, user, address)` and
sends an authentication request, which SQLx obeys. The password primitives SQLx
then runs are **RustCrypto**, not the platform TLS provider: they travel inside
the TLS tunnel but are computed outside the TLS boundary. The table under
[SQLx's direct authentication cryptographic operations](#sqlxs-direct-authentication-cryptographic-operations)
lists each one.

Call path (server-driven, after the handshake):

```text
server: match pg_hba.conf rule
  -> AuthenticationSASL     -> sqlx-postgres SCRAM  -> hmac + sha2   (RustCrypto)
  -> AuthenticationMD5      -> sqlx-postgres MD5     -> md-5          (RustCrypto)
  -> AuthenticationCleartext-> sqlx-postgres sends password bytes
  -> AuthenticationOk       -> (cert/trust/peer: no password crypto)
```

### Why the split matters

The server makes the decisive choice, after the handshake, and SQLx obeys it.
A client cannot force the server off the password path. It can only decline to
*supply* a password and refuse to proceed when its own configuration would
leave a password path open. The compliance profile below does that and treats
the server-side half as a hard deployment prerequisite.

## SQLx's direct authentication cryptographic operations

The "outside the boundary" inventory for a cryptographic review:

| Server auth request | SQLx operation | Implementing crate | Family |
| --- | --- | --- | --- |
| `AuthenticationSASL` (SCRAM-SHA-256) | `Hi`/`HMAC`/`H` proof computation | [`hmac`], [`sha2`] | RustCrypto |
| `AuthenticationMD5Password` | `md5(md5(password + user) + salt)` | [`md-5`] | RustCrypto |
| `AuthenticationCleartextPassword` | none (password sent as-is over TLS) | — | — |
| `AuthenticationOk` (`trust` / `cert` / `peer`) | none | — | — |
| TLS handshake + client cert | key exchange, AEAD, chain + hostname verify, client cert | [native-tls] (OpenSSL / Security.framework) | platform TLS |

The general-purpose `store-postgres` profile enables SQLx's password
authentication feature for compatibility. The `store-postgres-cert-auth`
profile compiles no cleartext, MD5, or SCRAM authentication support, and omits
migration checksums and advisory-lock string hashing. The default FIPS feature
set compiles no store. An experimental opt-in FIPS build adds this reduced
profile and also requires client-certificate authentication during startup
validation.

## Selected approach: certificate authentication

The supported compliance approach is **mutual-TLS client-certificate
authentication**, compiled with `store-postgres-cert-auth` and gated by the
`require_certificate_authentication` filter field. A certificate-only build
rejects PostgreSQL configuration that does not enable that field. Under `cert`
authentication the server answers the TLS handshake with `AuthenticationOk` and
**no password primitive executes**.

### Client configuration

Both filters accept the same TLS fields, carried by `PgTlsConfig` in
`store-backends/src/postgres_tls.rs`: `ssl_mode`, `ssl_root_cert`,
`ssl_client_cert`, `ssl_client_key`, and `require_certificate_authentication`.
The generated [`openai_responses_store`] and [`openai_conversations`] pages
describe each field.

`ssl_client_key` is a filesystem path, never inline secret material, so no key
bytes reach the config, logs, or diagnostics. Paths containing `..` traversal
are rejected. The key must be an **unencrypted PKCS#8** PEM, because the
native-tls backend loads client identities as PKCS#8 only (no SEC1/PKCS#1).
Convert a legacy key with
`openssl pkcs8 -topk8 -nocrypt -in client.key -out client.pk8.key`.

### What the compliance profile enforces (client-visible)

When `require_certificate_authentication: true`, validation runs at
filter construction (`PgTlsConfig::validate`, called from each filter's
`from_config`) and **fails closed before any traffic is served** if any
client-detectable password path remains open:

1. `ssl_mode` must be `verify-full` (full chain + hostname verification).
2. Both `ssl_client_cert` and `ssl_client_key` must be present.
3. `database_url` must not contain a password (`user:pass@`).
4. `database_url` must not contain TLS query parameters (`sslmode`,
   `sslrootcert`, …). TLS is configured only through the filter fields, so
   they are authoritative.
5. The `PGPASSWORD` environment variable must not be set.
6. `database_url` must not carry non-addressing connection parameters
   (`application_name`, `options`/`options[...]`, `statement-cache-capacity`).
   The compliance rebuild (below) reconstructs the connection from addressing
   and TLS fields only, so any such parameter would be **silently dropped**.
   Losing `options[search_path]` in particular could route store DDL and
   queries to an unintended schema. Set these defaults on the database role
   instead, e.g. `ALTER ROLE <role> SET search_path = …`.
7. The `ssl_client_key` file must not be group- or world-accessible (Unix
   mode `0600`, enforced on Unix); a readable key would leak the PostgreSQL
   client identity to other local users.

The connection options are also rebuilt from
`PgConnectOptions::new_without_pgpass()` with only addressing fields (host,
port, socket, user, database) plus the filter's TLS settings. A stray
`~/.pgpass` entry cannot inject a password, and no non-addressing URL parameter
(item 6) is silently honored. The effective password is `None`, so SQLx has
nothing to feed a SCRAM/MD5 challenge even if one arrived.

### What the profile CANNOT enforce (server-side prerequisite)

The client cannot see or control `pg_hba.conf`. That the server authenticates
with `cert` and never issues a password challenge is a **deployment
prerequisite the operator must verify out of band.** If a misconfigured server
sends `AuthenticationSASL`, SQLx attempts SCRAM with a `None` password and the
connection fails instead of downgrading. The client still cannot turn a
password-issuing server into a cert-only one.

### Trade-offs

- **Requires a PKI.** Operators issue and rotate client certificates and
  distribute the signing CA. That is heavier than a password, and it removes
  the shared secret.
- **Server coupling.** The guarantee is complete only when the server's
  `pg_hba.conf` uses `cert`. The proxy validates its half; the server half is
  operational.
- **CN → role mapping.** The client certificate's Common Name must equal the
  database role, or be mapped through a `pg_ident.conf` map selected with the
  `cert` rule's `map=<name>` option. A mismatch is a hard connection failure. The CA in `ssl_ca_file`
  must sign the client certificate the proxy presents.
- **`verify-full` on loopback.** Verification checks the hostname/IP against
  the server certificate SANs, so test and dev servers need a certificate with
  the right SAN (e.g. a `127.0.0.1` IP SAN).

## Sample `pg_hba.conf`

This refuses every password path for TCP clients and still lets the
container/image entrypoint bootstrap the database over the local Unix socket:

```conf
# TYPE   DATABASE   USER   ADDRESS        METHOD
# Local socket for bootstrap/administration only.
local    all        all                   trust
# TCP is TLS-only AND certificate-only. No `host` (non-TLS) rules and no
# password methods (md5/scram-sha-256/password) exist, so:
#   * a non-TLS connection is refused (no matching rule), and
#   * a TLS connection without a valid client certificate is refused.
hostssl  all        all    0.0.0.0/0      cert
hostssl  all        all    ::/0           cert
```

Server startup flags that pair with the rules above:

```text
postgres \
  -c ssl=on \
  -c ssl_cert_file=/path/server.crt \
  -c ssl_key_file=/path/server.key \
  -c ssl_ca_file=/path/ca.crt \
  -c hba_file=/path/pg_hba.conf
```

## End-to-end verification

Two ignored (container-gated) integration tests run the full boundary against a
real PostgreSQL server started with the `pg_hba.conf` above:

- `tests/integration/tests/suite/examples/openai_responses_store_postgres_mtls.rs`
- `tests/integration/tests/suite/examples/openai_conversations_postgres_mtls.rs`

Each generates a fresh CA that signs a server certificate (with a `127.0.0.1`
IP SAN so `verify-full` succeeds on loopback) and a client certificate whose
Common Name is the database role. The proxy persists state over the
certificate-authenticated TLS connection, and the test re-reads the row over
the same path, so no password ever traverses the connection. The container
harness is `start_postgres_cert_auth` in `tests/utils/src/net/postgres.rs`.

## Example configurations

- `examples/configs/openai/responses/response-store-postgres-mtls.yaml`
- `examples/configs/openai/conversations/conversations-postgres-mtls.yaml`

## Key files

- `store-backends/src/postgres_tls.rs`: `PgTlsConfig`, compliance-profile
  validation
- `store-backends/src/postgres.rs`: `pg_connect_options`,
  `rebuild_without_password_file`, connection establishment
- `store-backends/src/postgres_url.rs`: URL password / TLS-parameter
  detection
- `apis/src/openai/responses/store/config.rs`: response-store TLS config
  wiring
- `apis/src/openai/conversations/config.rs`: conversations TLS config
  wiring

## Related

- [Response store](response-store.md)
- [Outbound callout security](outbound-callouts.md)
- [Features](../features.md)
- [SQLx password-authentication feature](https://github.com/transact-rs/sqlx/pull/4417)
- [SQLx migration feature](https://github.com/transact-rs/sqlx/pull/4420)
- [SQLx advisory-lock hashing issue](https://github.com/transact-rs/sqlx/issues/4421)

[native-tls]: https://github.com/sfackler/rust-native-tls
[`openai_responses_store`]: ../filters/agents/openai_responses_store.md
[`openai_conversations`]: ../filters/agents/openai_conversations.md
[`hmac`]: https://github.com/RustCrypto/MACs
[`sha2`]: https://github.com/RustCrypto/hashes
[`md-5`]: https://github.com/RustCrypto/hashes
