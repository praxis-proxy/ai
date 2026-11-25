// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Check-mode diagnostics must remain visible without contaminating the YAML dump.

use std::{error::Error, process::Command};

#[test]
fn check_modes_report_each_deprecated_field_on_stderr() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let config = dir.path().join("praxis.yaml");
    std::fs::write(
        &config,
        r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: credential_injection
        clusters:
          - { name: first, header: Authorization, value: example, strip_client_credential: true }
          - { name: second, header: Authorization, value: example, strip_client_credential: false }
      - filter: static_response
        status: 200
"#,
    )?;
    for flag in ["--validate", "--dump"] {
        let output = Command::new(env!("CARGO_BIN_EXE_praxis-ai"))
            .args([flag, "--config"])
            .arg(&config)
            .output()?;
        let stderr = String::from_utf8(output.stderr)?;
        assert!(output.status.success(), "{flag}: {stderr}");
        assert_eq!(
            stderr
                .lines()
                .filter(|line| line.contains("remove strip_client_credential"))
                .count(),
            2,
            "{flag}: {stderr}"
        );
        if flag == "--dump" {
            let dump: serde_yaml::Value = serde_yaml::from_slice(&output.stdout)?;
            assert!(dump.get("configuration").is_some());
        } else {
            assert!(output.stdout.is_empty());
        }
    }
    Ok(())
}
