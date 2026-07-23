// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Live OGX contract test for the file-search callout filter.

use std::time::{Duration, Instant};

use http::header::CONTENT_TYPE;
use praxis_filter::FilterAction;
use reqwest::{Client, RequestBuilder};
use serde_json::{Value, json};

use super::FileSearchCalloutFilter;
use crate::openai::responses::state::ResponsesState;

const DOCUMENT_MARKER: &str = "PRAXIS-OGX-FILE-SEARCH-7319";
const DOCUMENT_NAME: &str = "praxis-file-search-report.txt";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a live OGX starter server"]
async fn live_ogx_file_search_callout() {
    let base_url = std::env::var("OGX_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8321".to_owned());
    let fixture = OgxFixture::provision(base_url).await.expect("provision OGX fixture");

    let result = exercise_filter(&fixture).await;
    fixture.cleanup().await;

    result.expect("file-search filter must interoperate with OGX");
}

async fn exercise_filter(fixture: &OgxFixture) -> Result<(), String> {
    let config: serde_yaml::Value = serde_yaml::from_str(&format!(
        "vector_store_url: {:?}\nallow_private_url: true\nauth_type: none\ntimeout_ms: 30000\n",
        fixture.base_url
    ))
    .map_err(|error| format!("build filter config: {error}"))?;
    let filter = FileSearchCalloutFilter::from_config(&config)
        .map_err(|error| format!("construct file-search filter: {error}"))?;

    let mut matching = make_context(pending_state(&fixture.vector_store_id, "finance"));
    let action = filter
        .on_request(&mut matching)
        .await
        .map_err(|error| format!("execute matching file search: {error}"))?;
    if !matches!(action, FilterAction::Continue) {
        return Err("matching file search did not continue".to_owned());
    }

    let state = matching
        .extensions
        .get::<ResponsesState>()
        .ok_or_else(|| "matching file search removed ResponsesState".to_owned())?;
    let call = state
        .output_items()
        .first()
        .ok_or_else(|| "matching file search produced no output item".to_owned())?;
    if call["status"] != "completed" {
        return Err(format!(
            "matching file search status was {}; call={call}; messages={}",
            call["status"],
            Value::Array(state.messages.clone())
        ));
    }
    let result_text = call["results"]
        .as_array()
        .and_then(|results| results.first())
        .and_then(|result| result["text"].as_str())
        .ok_or_else(|| "matching file search produced no result text".to_owned())?;
    if !result_text.contains(DOCUMENT_MARKER) {
        return Err(format!("matching file search omitted marker: {result_text}"));
    }
    if state.citation_files.get(&fixture.file_id).map(String::as_str) != Some(DOCUMENT_NAME) {
        return Err("matching file search did not retain citation metadata".to_owned());
    }
    let model_output = state
        .messages
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .and_then(|item| item["output"].as_str())
        .ok_or_else(|| "matching file search produced no model context".to_owned())?;
    if !model_output.contains(DOCUMENT_MARKER) || !model_output.contains(&format!("<|{}|>", fixture.file_id)) {
        return Err("matching file search produced incomplete model context".to_owned());
    }

    let mut non_matching = make_context(pending_state(&fixture.vector_store_id, "engineering"));
    let action = filter
        .on_request(&mut non_matching)
        .await
        .map_err(|error| format!("execute non-matching file search: {error}"))?;
    if !matches!(action, FilterAction::Continue) {
        return Err("non-matching file search did not continue".to_owned());
    }
    let state = non_matching
        .extensions
        .get::<ResponsesState>()
        .ok_or_else(|| "non-matching file search removed ResponsesState".to_owned())?;
    let call = state
        .output_items()
        .first()
        .ok_or_else(|| "non-matching file search produced no output item".to_owned())?;
    if call["status"] != "completed" || call["results"].as_array().is_none_or(|results| !results.is_empty()) {
        return Err(format!(
            "non-matching metadata filter returned results: {}",
            call["results"]
        ));
    }

    Ok(())
}

fn pending_state(vector_store_id: &str, department: &str) -> ResponsesState {
    let output = vec![json!({
        "type": "file_search_call",
        "id": "fs-live-ogx",
        "status": "searching",
        "queries": ["What is the Praxis OGX file-search marker?"],
    })];
    ResponsesState {
        include: vec!["file_search_call.results".to_owned()],
        response_object: json!({"id": "resp-live-ogx", "output": output}),
        tools: vec![json!({
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "max_num_results": 5,
            "filters": {"type": "eq", "key": "department", "value": department},
        })],
        ..Default::default()
    }
}

fn make_context(state: ResponsesState) -> praxis_filter::HttpFilterContext<'static> {
    let request = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut context = crate::test_utils::make_filter_context(request);
    context.set_metadata("openai_responses_format.stream", "false");
    context.extensions.insert(state);
    context
}

struct OgxFixture {
    base_url: String,
    client: Client,
    file_id: String,
    vector_store_id: String,
}

impl OgxFixture {
    async fn provision(base_url: String) -> Result<Self, String> {
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|error| format!("build OGX client: {error}"))?;
        let mut fixture = Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            client,
            file_id: String::new(),
            vector_store_id: String::new(),
        };

        if let Err(error) = fixture.create_resources().await {
            fixture.cleanup().await;
            return Err(error);
        }
        Ok(fixture)
    }

    async fn create_resources(&mut self) -> Result<(), String> {
        let embedding_model = std::env::var("OGX_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "sentence-transformers/all-minilm-l6-v2".to_owned());
        let embedding_dimension = std::env::var("OGX_EMBEDDING_DIMENSION")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(384);
        let store = send_json(
            "create vector store",
            self.client
                .post(format!("{}/v1/vector_stores", self.base_url))
                .header(CONTENT_TYPE, "application/json")
                .body(
                    json!({
                        "name": format!("praxis-file-search-{}", std::process::id()),
                        "embedding_model": embedding_model,
                        "embedding_dimension": embedding_dimension,
                        "provider_id": "faiss",
                    })
                    .to_string(),
                ),
        )
        .await?;
        self.vector_store_id = required_string(&store, "id", "vector store")?;

        let boundary = format!("praxis-ogx-{}", std::process::id());
        let upload = send_json(
            "upload file",
            self.client
                .post(format!("{}/v1/files", self.base_url))
                .header(CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
                .body(multipart_body(&boundary)),
        )
        .await?;
        self.file_id = required_string(&upload, "id", "file")?;

        send_json(
            "attach file",
            self.client
                .post(format!(
                    "{}/v1/vector_stores/{}/files",
                    self.base_url, self.vector_store_id
                ))
                .header(CONTENT_TYPE, "application/json")
                .body(
                    json!({
                        "file_id": self.file_id,
                        "attributes": {"department": "finance", "region": "emea"},
                    })
                    .to_string(),
                ),
        )
        .await?;

        self.wait_for_indexing().await
    }

    async fn wait_for_indexing(&self) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            let attachment = send_json(
                "read vector-store file",
                self.client.get(format!(
                    "{}/v1/vector_stores/{}/files/{}",
                    self.base_url, self.vector_store_id, self.file_id
                )),
            )
            .await?;
            match attachment.get("status").and_then(Value::as_str) {
                Some("completed") => return Ok(()),
                Some("failed" | "cancelled") => {
                    return Err(format!("OGX indexing failed: {}", attachment["last_error"]));
                },
                _ if Instant::now() >= deadline => return Err("OGX indexing timed out".to_owned()),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }

    async fn cleanup(&self) {
        if !self.vector_store_id.is_empty() {
            let _response = self
                .client
                .delete(format!("{}/v1/vector_stores/{}", self.base_url, self.vector_store_id))
                .send()
                .await;
        }
        if !self.file_id.is_empty() {
            let _response = self
                .client
                .delete(format!("{}/v1/files/{}", self.base_url, self.file_id))
                .send()
                .await;
        }
    }
}

async fn send_json(operation: &str, request: RequestBuilder) -> Result<Value, String> {
    let response = request.send().await.map_err(|error| format!("{operation}: {error}"))?;
    let status = response.status();
    let body = response
        .bytes()
        .await
        .map_err(|error| format!("{operation} response body: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "{operation} returned {status}: {}",
            String::from_utf8_lossy(&body)
        ));
    }
    serde_json::from_slice(&body).map_err(|error| format!("{operation} response JSON: {error}"))
}

fn required_string(value: &Value, field: &str, object: &str) -> Result<String, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("OGX {object} response omitted {field}"))
}

fn multipart_body(boundary: &str) -> Vec<u8> {
    format!(
        "--{boundary}\r\n\
         Content-Disposition: form-data; name=\"purpose\"\r\n\r\n\
         assistants\r\n\
         --{boundary}\r\n\
         Content-Disposition: form-data; name=\"file\"; filename=\"{DOCUMENT_NAME}\"\r\n\
         Content-Type: text/plain\r\n\r\n\
         Praxis OGX file-search integration report.\n\
         The secret marker is {DOCUMENT_MARKER}.\n\
         Revenue grew 37 percent year over year.\r\n\
         --{boundary}--\r\n"
    )
    .into_bytes()
}
