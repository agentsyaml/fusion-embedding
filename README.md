# fusion-embedding

[简体中文](README.zh-CN.md)

`fusion-embedding` is a small Rust client for Qwen3-VL embedding deployments with fixed wire profiles for vLLM, llama.cpp, and DashScope. A `Client` is constructed with one backend, endpoint, deployment identity, and expected vector dimension.

## Backends and features

| Feature | Backend | Constructor |
| --- | --- | --- |
| `vllm` | vLLM OpenAI-compatible embeddings | `Client::vllm(VllmConfig)` |
| `llama-cpp` | llama.cpp embeddings | `Client::llama_cpp(LlamaCppConfig)` |
| `dashscope` | DashScope native multimodal embeddings | `Client::dashscope(DashScopeConfig)` |

All features are enabled by default. Disable defaults when only one backend is needed:

```toml
[dependencies]
fusion-embedding = { version = "0.1", default-features = false, features = ["vllm"] }
```

## Scope and limits

The public API provides `embed_texts` for a non-empty batch of text and `embed_fusion` for ordered `ContentPart::{Text, ImageDataUrl}` input. It accepts **only canonical PNG or JPEG base64 data URLs** for images. The library does not perform file I/O: read and encode any image in your application before calling it. Remote media URLs, local file paths, video, and other media types are not accepted.

vLLM and llama.cpp fusion requests require exactly one image. DashScope accepts up to five images (including text-only content) and one to twenty content items. A client fixes its model identity, pipeline ID, and expected dimensions; `EmbedOptions` can only override the per-call timeout.

For Qwen text-only deployments, vLLM and llama.cpp apply a fixed Qwen instruction/chat profile to every `embed_texts` item before sending the batch. DashScope text requests remain native.

## Install

```toml
[dependencies]
fusion-embedding = "0.1"
url = "2"
```

Add `secrecy = "0.10"` when constructing a vLLM client with an API key or any DashScope client.

## Minimal calls

Construct a vLLM client and embed text:

```rust,no_run
use std::num::NonZeroUsize;

use fusion_embedding::{Client, EmbedOptions, VllmConfig};
use url::Url;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::vllm(VllmConfig::new(
    Url::parse("http://127.0.0.1:8000/v1/models")?,
    "qwen3-vl-embedding",
    "my-deployment-revision",
    NonZeroUsize::new(2048).unwrap(),
)?)?;
let response = client
    .embed_texts(&["a short sentence".into()], &EmbedOptions::default())
    .await?;
println!("{}", response.embeddings()[0].len());
# Ok(())
# }
```

For image-text fusion, create the data URL outside the library and preserve the desired part order:

```rust,no_run
# use fusion_embedding::{Client, ContentPart, EmbedOptions};
# async fn example(client: Client, image_data_url: String) -> Result<(), fusion_embedding::Error> {
let response = client
    .embed_fusion(
        &[
            ContentPart::ImageDataUrl(image_data_url),
            ContentPart::Text("a red square".into()),
        ],
        &EmbedOptions::default(),
    )
    .await?;
# let _ = response;
# Ok(())
# }
```

The value passed to `ImageDataUrl` has the form `data:image/png;base64,<canonical-base64>` or `data:image/jpeg;base64,<canonical-base64>`. Encode image bytes with your chosen base64 encoder; do not pass a path or URL.

## Backend configuration

| Backend | `new` arguments | Endpoint behavior | Dimensions and authentication |
| --- | --- | --- | --- |
| vLLM | `base_endpoint, model, pipeline_id, expected_dimensions` | Accepts `/` or `/v1/models`, then uses `/v1/embeddings` | Native dimensions by default; `.with_server_dimensions(...)` requests one. API key is optional via `.with_api_key(SecretString)`. |
| llama.cpp | `endpoint, model_identity, pipeline_id, expected_dimensions, media_marker` | Accepts `/` or `/v1/models`, then uses `/v1/embeddings` | Native dimensions only. `model_identity` identifies the embedding space and is not sent to llama.cpp. |
| DashScope | `full_native_post_url, model, pipeline_id, native_dimensions, api_key` | Requires the complete native POST URL | API key is required. `.with_server_dimensions(...)` accepts only 256, 512, 768, 1024, 1536, 2048, or 2560. |

All configurations support `.with_timeouts(connect_timeout, request_timeout)`. Each also has `.allow_remote_http(true)`, which is only for an intentional non-loopback cleartext deployment.

## Responses, errors, and safety

`EmbeddingResponse` exposes ordered `embeddings()`, `space()`, `usage()`, and `diagnostics()`. `EmbeddingSpace` records the provider, configured model, pipeline ID, profile revision, dimension mode, and expected dimension. `Usage` and request/response IDs are optional.

`Error` distinguishes invalid configuration or input, transport failures, provider API errors, invalid responses, and dimension mismatches. Successful responses are checked for complete unique indices, expected dimensions, finite non-zero vectors, then L2-normalized by the library.

HTTPS is accepted by default. Cleartext HTTP is allowed only for loopback endpoints, or after explicit `.allow_remote_http(true)` opt-in for a remote endpoint. Redirects are disabled. API keys and content are redacted in relevant `Debug` output.

## E2E tests

The live tests are ignored and read **process environment variables only**; the library itself does not load `.env`. Use `env.example` as a names-only template, export the applicable values, and do not commit secrets. They also run a three-image COCO image-to-description retrieval sanity check with a fixed cosine threshold and a Top-1 assertion; sample attribution is in [`samples/coco-mini/README.md`](samples/coco-mini/README.md).

| Backend | Required variables | Optional variables |
| --- | --- | --- |
| vLLM | `E2E_IMAGE_PATH`, `VLLM_MODELS_URL` (or `VLLM_BASE_URL`), `VLLM_CONFIGURED_MODEL`, `VLLM_PIPELINE_ID`, `VLLM_DIMENSIONS` | `VLLM_API_KEY`, `VLLM_SEND_DIMENSIONS`, `VLLM_ALLOW_REMOTE_HTTP` |
| llama.cpp | `E2E_IMAGE_PATH`, `LLAMA_CPP_MODELS_URL`, `LLAMA_CPP_CONFIGURED_MODEL`, `LLAMA_CPP_PIPELINE_ID`, `LLAMA_CPP_DIMENSIONS` | `LLAMA_CPP_MEDIA_MARKER`, `LLAMA_CPP_ALLOW_REMOTE_HTTP` |
| DashScope official | `E2E_IMAGE_PATH`, `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, `DASHSCOPE_CONFIGURED_MODEL`, `DASHSCOPE_PIPELINE_ID`, `DASHSCOPE_NATIVE_DIMENSIONS` | `DASHSCOPE_DIMENSIONS` |
| DashScope enterprise | `E2E_IMAGE_PATH`, `DASHSCOPE_EMBEDDING_API_KEY`, `DASHSCOPE_EMBEDDING_BASE_URL`, `DASHSCOPE_ENTERPRISE_CONFIGURED_MODEL`, `DASHSCOPE_ENTERPRISE_PIPELINE_ID`, `DASHSCOPE_ENTERPRISE_NATIVE_DIMENSIONS` | `DASHSCOPE_ENTERPRISE_DIMENSIONS`, `DASHSCOPE_ENTERPRISE_ALLOW_REMOTE_HTTP` |

If `LLAMA_CPP_MEDIA_MARKER` is absent, the E2E test probes the configured llama.cpp endpoint's `/props` route. Run one provider test, or all configured ignored tests:

```sh
cargo test --test e2e vllm_e2e -- --ignored
cargo test --test e2e -- --ignored
```

## Development and release checks

```sh
cargo fmt --check
cargo test
cargo package
```
