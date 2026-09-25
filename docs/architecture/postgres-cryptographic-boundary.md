# PostgreSQL Cryptographic Boundary

The `openai_store` and `openai_conversations` filters can
persist state to PostgreSQL. This document defines the cryptographic
boundary of those connections: which cryptographic operations run inside
the platform TLS provider, which run application-side, and how the
certificate-authentication compliance profile keeps password
cryptography off the connection path.

The word *boundary* is deliberate. A connection to PostgreSQL performs
cryptography in two distinct places with two distinct providers. Keeping
them separate — and steering deployments onto the side we control — is
the whole point.

## Two call paths, two providers

There are two independent cryptographic call paths in a PostgreSQL
connection. They are selected at different moments, by different actors.

### 1. TLS transport (inside the boundary)

The TLS handshake — key exchange, certificate chain verification,
hostname verification, and, when configured, **client-certificate
authentication** — runs inside the platform TLS library. SQLx selects it
through its `tls-native-tls` feature, which binds [native-tls] to the
operating system's own provider: **OpenSSL on Linux** and
**Security.framework (Secure Transport) on macOS**. The `store-postgres`
feature in `apis/Cargo.toml` turns it on:

```toml
[features]
store = ["dep:sqlx", "dep:dashmap"]
store-postgres = ["store", "sqlx/postgres", "sqlx/tls-native-tls"]
```

Everything on this path is the host's system TLS stack. Client
certificate authentication is notable: the client proves its identity as
part of the TLS handshake itself, so the *authentication* crypto for a
cert-authenticated connection lives entirely inside the platform TLS
boundary — never in application-side password code.

Because native-tls delegates to the host, the cryptographic assurance of
this path is exactly the assurance of the deployed platform and its
validated configuration. On a RHEL host, OpenSSL follows the system-wide
crypto policy and can operate a FIPS-validated module when the host runs
in FIPS mode; on macOS the provider is Security.framework. Scope any
compliance claim to the platform you actually ship and to its validated
configuration.

Call path (client side):

```text
PostgresResponseStore::new
  -> pg_connect_options(database_url, tls)      # apis/src/store/postgres.rs
       -> PgConnectOptions::ssl_mode / ssl_root_cert
          / ssl_client_cert / ssl_client_key
  -> PgPoolOptions::connect_with(options)
       -> sqlx-postgres TLS upgrade
            -> native-tls handshake  (OpenSSL / Security.framework:
               key exchange, AEAD, cert verify)
                 -> [optional] client certificate presented here
```

### 2. Password authentication (outside the boundary)

*After* the TLS handshake completes, the PostgreSQL **server** decides
how to authenticate the connection. It consults `pg_hba.conf`, matches a
rule against `(type, database, user, address)`, and sends back an
authentication request. SQLx obeys whatever arrives:

- `AuthenticationSASL` (SCRAM-SHA-256) → SQLx computes the SCRAM proof
  using the RustCrypto [`hmac`] + [`sha2`] crates.
- `AuthenticationMD5Password` → SQLx computes the MD5 digest using the
  RustCrypto [`md-5`] crate.
- `AuthenticationCleartextPassword` → SQLx sends the password directly.
- `AuthenticationOk` → no password primitive runs at all (this is the
  `trust`, `cert`, and `peer` case).

These password primitives are **RustCrypto**, not the platform TLS
provider. They sit *outside* the TLS boundary. They are still transported
inside the TLS tunnel, but the cryptographic computation is performed by
a different, general-purpose set of crates.

Call path (server-driven, after the handshake):

```text
server: match pg_hba.conf rule
  -> AuthenticationSASL     -> sqlx-postgres SCRAM  -> hmac + sha2   (RustCrypto)
  -> AuthenticationMD5      -> sqlx-postgres MD5     -> md-5          (RustCrypto)
  -> AuthenticationCleartext-> sqlx-postgres sends password bytes
  -> AuthenticationOk       -> (cert/trust/peer: no password crypto)
```

### Why the split matters

The decisive choice between these two paths is made by the **server**,
delivered *after* the TLS handshake, and obeyed by SQLx. A client cannot
force the server off the password path — it can only decline to *supply*
a password, and refuse to proceed when its own configuration would leave
a password path open. The compliance profile (below) does exactly that,
and documents the server-side half as a hard deployment prerequisite.

## SQLx's direct authentication cryptographic operations

The table below enumerates the authentication cryptographic operations
SQLx performs directly and the crate that implements each. This is the
"outside the boundary" inventory required for a cryptographic review.

| Server auth request | SQLx operation | Implementing crate | Family |
| --- | --- | --- | --- |
| `AuthenticationSASL` (SCRAM-SHA-256) | `Hi`/`HMAC`/`H` proof computation | [`hmac`], [`sha2`] | RustCrypto |
| `AuthenticationMD5Password` | `md5(md5(password + user) + salt)` | [`md-5`] | RustCrypto |
| `AuthenticationCleartextPassword` | none (password sent as-is over TLS) | — | — |
| `AuthenticationOk` (cert / trust / peer) | none | — | — |
| TLS handshake + client cert | key exchange, AEAD, chain + hostname verify, client cert | [native-tls] (OpenSSL / Security.framework) | platform TLS |

The password families (SCRAM/MD5) are pulled in transitively through
`sqlx-postgres`; they are not separately gated. The only way to keep
them off the connection is to ensure the server never issues a password
challenge — i.e. authenticate with a client certificate.

## Selected approach: certificate authentication

The supported compliance approach is **mutual-TLS client-certificate
authentication**, gated by the `require_certificate_authentication`
filter field. Under `cert` authentication the server responds to the TLS
handshake with `AuthenticationOk` and **no password primitive executes**
— authentication happens entirely inside the platform TLS boundary.

### Client configuration

Both filters accept the same TLS fields (carried by `PgTlsConfig` in
`apis/src/store/postgres_tls.rs`):

| Field | Purpose |
| --- | --- |
| `ssl_mode` | TLS mode; must be `verify-full` under the compliance profile |
| `ssl_root_cert` | PEM root CA that signs the server certificate |
| `ssl_client_cert` | PEM client certificate presented for authentication |
| `ssl_client_key` | PEM private key for the client certificate |
| `require_certificate_authentication` | Enable the compliance profile (fail closed) |

The client key is a filesystem path, never inline secret material, so no
key bytes are embedded in the config, logs, or diagnostics. Paths are
rejected if they contain `..` traversal. The key must be an **unencrypted
PKCS#8** PEM: the native-tls backend loads client identities as PKCS#8
only and does not accept SEC1/PKCS#1 keys, so convert a legacy key with
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
   `sslrootcert`, …) — TLS is configured exclusively through the filter
   fields so they are authoritative.
5. The `PGPASSWORD` environment variable must not be set.
6. `database_url` must not carry non-addressing connection parameters
   (`application_name`, `options`/`options[...]`, `statement-cache-capacity`).
   The compliance rebuild (below) reconstructs the connection from addressing
   and TLS fields only, so any such parameter would be **silently dropped** —
   losing `options[search_path]` in particular could route store DDL and
   queries to an unintended schema. Set these defaults on the database role
   instead, e.g. `ALTER ROLE <role> SET search_path = …`.
7. The `ssl_client_key` file must not be group- or world-accessible (Unix
   mode `0600`, enforced on Unix); a readable key would leak the PostgreSQL
   client identity to other local users.

In addition, the connection options are rebuilt from
`PgConnectOptions::new_without_pgpass()` carrying only addressing fields
(host, port, socket, user, database) plus the filter's TLS settings, so a
stray `~/.pgpass` entry cannot inject a password behind the operator's back
and no non-addressing URL parameter (item 6) is silently honored. The
effective password is therefore `None`, and SQLx has nothing to feed a
SCRAM/MD5 challenge even if one arrived.

### What the profile CANNOT enforce (server-side prerequisite)

The client cannot see or control `pg_hba.conf`. The decisive requirement
— that the server authenticates with `cert` and never issues a password
challenge — is a **deployment prerequisite the operator must verify out
of band.** If the server is misconfigured to send `AuthenticationSASL`,
SQLx would attempt SCRAM; because the effective password is `None`, the
connection fails (rather than silently downgrading), but the client
cannot turn a password-issuing server into a cert-only one.

### Trade-offs

- **Requires a PKI.** Operators must issue and rotate client
  certificates and distribute the signing CA. This is heavier than a
  password but is the point: it removes the shared secret.
- **Server coupling.** The guarantee is only complete when the server's
  `pg_hba.conf` uses `cert`. The proxy documents and validates its half;
  the server half is operational.
- **CN → role mapping.** The client certificate's Common Name must equal
  the database role (or be mapped via `pg_ident.conf`). A mismatch is a
  hard connection failure.
- **`verify-full` on loopback.** Verification checks the hostname/IP
  against the server certificate SANs, so test and dev servers need a
  certificate with the right SAN (e.g. a `127.0.0.1` IP SAN).

## Sample `pg_hba.conf`

A complete server-side configuration that refuses every password path
for TCP clients while still letting the container/image entrypoint
bootstrap the database over the local Unix socket:

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

The client certificate's Common Name must match the database role it
authenticates as (map otherwise via `pg_ident.conf` and a `clientcert`
map option). The same CA in `ssl_ca_file` must sign the client
certificate the proxy presents.

## End-to-end verification

Two ignored (container-gated) integration tests exercise the full
boundary against a real PostgreSQL server started with the `pg_hba.conf`
above:

- `tests/integration/tests/suite/examples/openai_store_postgres_mtls.rs`
- `tests/integration/tests/suite/examples/openai_conversations_postgres_mtls.rs`

Each test generates a fresh CA that signs both the server certificate
(with a `127.0.0.1` IP SAN so `verify-full` succeeds on loopback) and a
client certificate whose Common Name is the database role. The proxy
persists state over the certificate-authenticated TLS connection, and
the test re-reads the row over the same verified-TLS + client-cert path
— proving no password ever traverses the connection. The container
harness lives in `tests/utils/src/net/postgres.rs`
(`start_postgres_cert_auth`).

## Example configurations

- `examples/configs/openai/responses/response-store-postgres-mtls.yaml`
- `examples/configs/openai/conversations/conversations-postgres-mtls.yaml`

## Key files

- `apis/src/store/postgres_tls.rs`: `PgTlsConfig`, compliance-profile
  validation
- `apis/src/store/postgres.rs`: `pg_connect_options`,
  `rebuild_without_password_file`, connection establishment
- `apis/src/store/postgres_url.rs`: URL password / TLS-parameter
  detection
- `apis/src/openai/responses/store/config.rs`: response-store TLS config
  wiring
- `apis/src/openai/conversations/config.rs`: conversations TLS config
  wiring

## Related

- [Response store](response-store.md)
- [Outbound callout security](outbound-callouts.md)
- [Features](../features.md)

[native-tls]: https://github.com/sfackler/rust-native-tls
[`hmac`]: https://github.com/RustCrypto/MACs
[`sha2`]: https://github.com/RustCrypto/hashes
[`md-5`]: https://github.com/RustCrypto/hashes
