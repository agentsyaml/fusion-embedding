# fusion-embedding

[简体中文](README.zh-CN.md)

`fusion-embedding` is a small Rust client for Qwen3-VL embedding deployments with fixed wire profiles for vLLM, llama.cpp, DashScope, and SGLang. A `Client` is constructed with one backend, endpoint, deployment identity, and expected vector dimension.

## Backends and features

| Feature | Backend | Constructor |
| --- | --- | --- |
| `vllm` | vLLM OpenAI-compatible embeddings | `Client::new(VllmConfig)` |
| `llama-cpp` | llama.cpp embeddings | `Client::new(LlamaCppConfig)` |
| `dashscope` | DashScope native multimodal embeddings | `Client::new(DashScopeConfig)` |
| `sglang` | SGLang v0.5.11+ standard embeddings | `Client::new(SglangConfig)` |

The configuration type selects the backend.

vLLM, llama.cpp, and DashScope are enabled by default; SGLang is opt-in. Select only the backend you need:

```toml
[dependencies]
fusion-embedding = { version = "0.1", default-features = false, features = ["sglang"] }
```

## Scope and limits

`embed_texts(&[S])` returns `BatchEmbeddingResponse`; `embed_multimodal(&[MultimodalPart])` fuses all parts into one `EmbeddingResponse`. Parts include `Text`, `ImageDataUrl`, `ImageUrl(Url)`, `VideoUrl(Url)`, and the legacy `VideoSource`, with lowercase constructors such as `image_url(Url)` and `video_url(Url)`. `ImageDataUrl` accepts only validated PNG/JPEG data URLs; that restriction does not apply to remote image URLs.

| Backend | Images per multimodal vector | Limits |
| --- | --- | --- |
| vLLM | One or more, in order | Supports HTTP(S) image URLs; configure a sufficient multimodal image limit and ensure the model supports multiple images. |
| llama.cpp | One or more, in order | Does not support remote image URLs; model and mmproj must support multiple images. |
| DashScope | Up to 10 | 1–20 parts; supports one video and image/video/text fusion. Use `ImageUrl` for remote images and `VideoUrl` for video. |
| SGLang standard endpoint | At most 1 image and 1 video | Supports HTTP(S) image URLs; no multiple images in one item; order is `image → video → text`. `VideoSource` is legacy pass-through. |

Use `embed_texts_with_options` or `embed_multimodal_with_options` to pass `EmbedOptions` by value.

SGLang fusion parts must be in canonical `image → video → text` order. Image, video, and text are each optional but may appear at most once; non-canonical or repeated parts are rejected. `VideoSource` is retained for SGLang's legacy pass-through input; DashScope video is supplied as `VideoUrl`.

For Qwen text-only deployments, vLLM and llama.cpp apply a fixed Qwen instruction/chat profile to every `embed_texts` item before sending the batch. DashScope text requests remain native.

## Install

```toml
[dependencies]
fusion-embedding = "0.1"
url = "2"
```

Add `secrecy = "0.10"` when constructing a vLLM or SGLang client with an API key, or any DashScope client.

## Minimal calls

Construct a vLLM client and embed text:

```rust,no_run
use std::num::NonZeroUsize;

use fusion_embedding::{Client, VllmConfig};
use url::Url;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(VllmConfig::new(
    Url::parse("http://127.0.0.1:8000/v1/models")?,
    "qwen3-vl-embedding",
    "my-deployment-revision",
    NonZeroUsize::new(2048).unwrap(),
)?)?;
let response = client
    .embed_texts(&["a short sentence"])
    .await?;
println!("{}", response.embeddings()[0].len());
# Ok(())
# }
```

For image-text fusion, pass a remote URL directly (the SDK does not download the image):

```rust,no_run
# use fusion_embedding::{Client, MultimodalPart};
# use url::Url;
# async fn example(client: Client) -> Result<(), Box<dyn std::error::Error>> {
let response = client
    .embed_multimodal(
        &[
            MultimodalPart::image_url(Url::parse("https://example.com/image.jpg")?),
            MultimodalPart::text("a red square"),
        ],
    )
    .await?;
# let _ = response;
# Ok(())
# }
```

`ImageDataUrl` requires `data:image/png;base64,<canonical-base64>` or `data:image/jpeg;base64,<canonical-base64>` and validates the media type and base64 strictly. For remote images, use `ImageUrl(Url)` / `image_url(Url)` instead. DashScope video uses `VideoUrl(Url)` / `video_url(Url)`, for example `MultimodalPart::video_url(Url::parse("https://example.com/clip.mp4")?)`; `VideoSource` is legacy SGLang pass-through, not the DashScope video API.

## Backend configuration

| Backend | `new` arguments | Endpoint behavior | Dimensions and authentication |
| --- | --- | --- | --- |
| vLLM | `base_endpoint, model, pipeline_id, expected_dimensions` | Accepts `/`, `/v1/models`, or `/v1/embeddings`; uses `/v1/embeddings` | Native dimensions by default; `.with_server_dimensions(...)` requests one, subject to deployment/model capability. API key is optional via `.with_api_key(SecretString)`. |
| llama.cpp | `endpoint, model_identity, pipeline_id, expected_dimensions, media_marker` | Accepts `/`, `/v1/models`, or `/v1/embeddings`; uses `/v1/embeddings` | Native dimensions only. `model_identity` identifies the embedding space and is not sent to llama.cpp. |
| DashScope | `full_native_post_url, model, pipeline_id, native_dimensions, api_key` | Requires the complete native POST URL | API key is required. `.with_server_dimensions(...)` accepts only 256, 512, 768, 1024, 1536, 2048, or 2560. |
| SGLang | `endpoint, model, pipeline_id, native_dimensions` | Requires SGLang v0.5.11+; accepts `/`, `/v1/models`, or `/v1/embeddings`; uses `/v1/embeddings` | Native dimensions only; API key is optional. Structured inputs are Jinja-templated by SGLang, so the client sends no manual Qwen text wrapper. |

These endpoint paths do not allow arbitrary proxy prefixes. DashScope continues to require its complete native URL.

All configurations support `.with_timeouts(connect_timeout, request_timeout)`. Each also has `.allow_remote_http(true)`, which is only for an intentional non-loopback cleartext deployment.

## Responses, errors, and safety

`BatchEmbeddingResponse` exposes `embeddings()`/`into_embeddings()`; `EmbeddingResponse` exposes `embedding()`/`into_embedding()`. Both expose metadata via `space()`, `usage()`, and `diagnostics()`.

`Error` distinguishes invalid configuration or input, transport failures, provider API errors, invalid responses, and dimension mismatches. Successful responses are checked for complete unique indices, expected dimensions, finite non-zero vectors, then L2-normalized locally by the library. vLLM requests do not send the removed `normalize` parameter; requested dimensions still depend on deployment/model capability.

HTTPS is accepted by default. Cleartext HTTP is allowed only for loopback endpoints, or after explicit `.allow_remote_http(true)` opt-in for a remote endpoint. Remote media URLs are syntax-validated only: the SDK does not fetch media; HTTP(S) URLs may include signed query strings, but userinfo and fragments are rejected. The service fetching remote media must constrain outbound access and redirects. Disabling redirects on SDK endpoint requests does not protect a provider's media fetch. API keys and content are redacted in relevant `Debug` output.

## E2E tests

The live tests are ignored and read **process environment variables only**; the library itself does not load `.env`. Use `env.example` as a names-only template, export the applicable values, and do not commit secrets. They also run a three-image COCO image-to-description retrieval sanity check with a fixed cosine threshold and a Top-1 assertion; sample attribution is in [`samples/coco-mini/README.md`](samples/coco-mini/README.md).

| Backend | Required variables | Optional variables |
| --- | --- | --- |
| vLLM | `E2E_IMAGE_PATH`, `VLLM_MODELS_URL` (or `VLLM_BASE_URL`), `VLLM_CONFIGURED_MODEL`, `VLLM_PIPELINE_ID`, `VLLM_DIMENSIONS` | `VLLM_API_KEY`, `VLLM_SEND_DIMENSIONS`, `VLLM_ALLOW_REMOTE_HTTP` |
| llama.cpp | `E2E_IMAGE_PATH`, `LLAMA_CPP_MODELS_URL`, `LLAMA_CPP_CONFIGURED_MODEL`, `LLAMA_CPP_PIPELINE_ID`, `LLAMA_CPP_DIMENSIONS` | `LLAMA_CPP_MEDIA_MARKER`, `LLAMA_CPP_ALLOW_REMOTE_HTTP` |
| DashScope official | `E2E_IMAGE_PATH`, `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, `DASHSCOPE_CONFIGURED_MODEL`, `DASHSCOPE_PIPELINE_ID`, `DASHSCOPE_NATIVE_DIMENSIONS` | `DASHSCOPE_DIMENSIONS` |
| DashScope enterprise | `E2E_IMAGE_PATH`, `DASHSCOPE_EMBEDDING_API_KEY`, `DASHSCOPE_EMBEDDING_BASE_URL`, `DASHSCOPE_ENTERPRISE_CONFIGURED_MODEL`, `DASHSCOPE_ENTERPRISE_PIPELINE_ID`, `DASHSCOPE_ENTERPRISE_NATIVE_DIMENSIONS` | `DASHSCOPE_ENTERPRISE_DIMENSIONS`, `DASHSCOPE_ENTERPRISE_ALLOW_REMOTE_HTTP` |
| SGLang | `E2E_IMAGE_PATH`, `SGLANG_MODELS_URL` (or `SGLANG_BASE_URL`), `SGLANG_MODEL`, `SGLANG_PIPELINE_ID`, `SGLANG_DIMENSIONS`, `SGLANG_VIDEO_SOURCE` | `SGLANG_API_KEY`, `SGLANG_ALLOW_REMOTE_HTTP` |

If `LLAMA_CPP_MEDIA_MARKER` is absent, the E2E test probes the configured llama.cpp endpoint's `/props` route. Run one provider test, or all configured ignored tests:

```sh
cargo test --test e2e vllm_e2e -- --ignored
cargo test --no-default-features --features sglang --test e2e sglang_e2e -- --ignored
cargo test --test e2e -- --ignored
```

The SGLang client has local HTTP contract coverage only; no live SGLang E2E result has been recorded.

## Development and release checks

CI runs these checks on pushes to `main` and pull requests. For a later release, bump and merge the `Cargo.toml` version first, then push the matching `vX.Y.Z` tag. Configure crates.io Trusted Publishing for owner `agentsyaml`, repository `fusion-embedding`, workflow `release.yml`, and environment `release`. Version `0.1.0` was published manually; do not create a release tag for it.

```sh
cargo fmt --check
cargo test
cargo package
```
