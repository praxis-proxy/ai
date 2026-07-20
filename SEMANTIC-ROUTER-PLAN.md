# Semantic Router Plugin Implementation Plan

## Objective
Implement an in-stream auto-model-substitution router as an `HttpFilter` in the `praxis-ai` repository. The filter will intercept OpenAI-compatible API traffic, evaluate prompt complexity using a lightweight model, and dynamically rewrite the destination to route the request to either an internal capable model (e.g., for heavy engineering tasks) or a fast external model (e.g., for routine chat).

## Phase 1: Filter Skeleton and Registration
1. **Create the Module:** Create a new module at `filters/src/inference/semantic_router.rs`.
2. **Trait Implementation:** Implement the `HttpFilter` trait and the `from_config` factory.
3. **Registry Wiring:** Register the new filter factory mapping in `server/src/lib.rs` (e.g., inside `register_general_ai_filters`).

## Phase 2: Interception and Extraction
1. **Request Hook:** Hook into the HTTP `POST` request lifecycle for configured AI endpoints (e.g., `/v1/chat/completions`).
2. **Payload Parsing:** Safely buffer and deserialize the JSON payload. 
3. **Extraction:** Extract the user's prompt from the `messages` array while keeping the original payload intact in memory.

## Phase 3: The Lite Model Evaluation
1. **Model Integration:** Integrate a mechanism to evaluate the extracted prompt. The lite model can be an in-memory `ort` model, or an external lite model (e.g., `gemini-3.1-flash-lite` via Vertex API, or a locally served model like `gemma`) using an HTTP client.
2. **Initialization & Configuration:** Configure the evaluation backend (ONNX model paths or HTTP endpoint details) during the factory initialization phase.
3. **Scoring:** Pass the prompt through the selected model (via local inference or HTTP request) to determine its intent category or complexity score.

## Phase 4: Dynamic Target Substitution
1. **Routing Configuration:** Update the YAML configuration to take a list of target models (each with a URL, optional auth, and perhaps a target model name) along with rules mapping the complexity score to the correct target model (e.g. `min_score` and `max_score`).
2. **Target Rewriting:** Based on the model's classification score, evaluate the routing rules to select the appropriate target model. Modify the routing target (e.g., upstream cluster, peer address, or request URI) directly within the filter's request lifecycle hook using `FilterContext`.
3. **Forwarding:** Forward the unaltered original JSON payload to the newly selected endpoint.

## Phase 5: Testing and Validation
1. **Unit Tests:** Add unit tests verifying payload extraction and the evaluation logic.
2. **Integration Test:** Create a functional example configuration in `examples/configs/semantic_router/`.
3. **End-to-End Validation:** Write an integration test in `tests/integration/tests/suite/examples/` proving that different prompts trigger routing to different upstream mock peers seamlessly.
