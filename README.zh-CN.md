# fusion-embedding

[English](README.md)

`fusion-embedding` 是一个精简的 Rust 客户端，为 vLLM、llama.cpp、DashScope 与 SGLang 的 Qwen3-VL embedding 部署提供固定的请求协议。一个 `Client` 在创建时绑定一个后端、端点、部署标识和预期向量维度。

## 后端与 feature

| Feature | 后端 | 构造方式 |
| --- | --- | --- |
| `vllm` | vLLM 的 OpenAI 兼容 embedding 接口 | `Client::new(VllmConfig)` |
| `llama-cpp` | llama.cpp embedding 接口 | `Client::new(LlamaCppConfig)` |
| `dashscope` | DashScope 原生多模态 embedding 接口 | `Client::new(DashScopeConfig)` |
| `sglang` | SGLang v0.5.11+ 标准 embedding 接口 | `Client::new(SglangConfig)` |

配置类型决定后端。

默认启用 vLLM、llama.cpp 与 DashScope；SGLang 需要显式启用。只选择 SGLang 时：

```toml
[dependencies]
fusion-embedding = { version = "0.1", default-features = false, features = ["sglang"] }
```

## 范围与限制

`embed_texts(&[S])` 返回 `BatchEmbeddingResponse`；`embed_multimodal(&[MultimodalPart])` 将全部 part 融合为一个 `EmbeddingResponse`。part 包括 `Text`、`ImageDataUrl`、`ImageUrl(Url)`、`VideoUrl(Url)` 和旧版 `VideoSource`，并提供 `image_url(Url)`、`video_url(Url)` 等小写构造方法。`ImageDataUrl` 仅接受严格校验的 PNG/JPEG data URL；此限制不适用于远程图片 URL。

| 后端 | 单个多模态向量图片数 | 限制 |
| --- | --- | --- |
| vLLM | 一张或多张，保序 | 支持 HTTP(S) 图片 URL；部署须配置足够的 multimodal image limit，实际模型须支持多图。 |
| llama.cpp | 一张或多张，保序 | 不支持远程图片 URL；模型和 mmproj 须支持多图。 |
| DashScope | 最多 10 张 | 1–20 个 part；支持一个视频及图片/视频/文本融合。远程图片用 `ImageUrl`，视频用 `VideoUrl`。 |
| SGLang standard endpoint | 最多一张图片和一个视频 | 支持 HTTP(S) 图片 URL；同一 item 不支持多图；顺序为 `image → video → text`。`VideoSource` 为旧版透传。 |

需要按次传递 `EmbedOptions` 时，使用按值的 `embed_texts_with_options` 或 `embed_multimodal_with_options`。

SGLang fusion parts 必须采用 `image → video → text` 的 canonical 顺序。图片、视频、文本均可省略但最多各一个；非 canonical 或重复 part 会被拒绝。`VideoSource` 保留为 SGLang 旧版透传输入；DashScope 视频使用 `VideoUrl`。

面向 Qwen 的纯文本部署时，vLLM 与 llama.cpp 会在发送批量 `embed_texts` 前为每项应用固定的 Qwen instruction/chat profile；DashScope 的文本请求保持原生格式。

## 安装

```toml
[dependencies]
fusion-embedding = "0.1"
url = "2"
```

若需为 vLLM 设置 API key，或使用 DashScope，请额外添加 `secrecy = "0.10"`。

## 最小调用示例

创建 vLLM 客户端并调用文本 embedding：

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
    .embed_texts(&["一段简短文本"])
    .await?;
println!("{}", response.embeddings()[0].len());
# Ok(())
# }
```

图文融合时可直接传入远程 URL（SDK 不会下载图片）：

```rust,no_run
# use fusion_embedding::{Client, MultimodalPart};
# use url::Url;
# async fn example(client: Client) -> Result<(), Box<dyn std::error::Error>> {
let response = client
    .embed_multimodal(
        &[
            MultimodalPart::image_url(Url::parse("https://example.com/image.jpg")?),
            MultimodalPart::text("一个红色方块"),
        ],
    )
    .await?;
# let _ = response;
# Ok(())
# }
```

`ImageDataUrl` 只接受 `data:image/png;base64,<规范-base64>` 或 `data:image/jpeg;base64,<规范-base64>`，并严格校验媒体类型和 base64。远程图片请改用 `ImageUrl(Url)` / `image_url(Url)`。DashScope 视频使用 `VideoUrl(Url)` / `video_url(Url)`，例如 `MultimodalPart::video_url(Url::parse("https://example.com/clip.mp4")?)`；`VideoSource` 是 SGLang 旧版透传，不是 DashScope 视频接口。

## 三种后端的配置差异

| 后端 | `new` 参数 | 端点行为 | 维度与鉴权 |
| --- | --- | --- | --- |
| vLLM | `base_endpoint, model, pipeline_id, expected_dimensions` | 接受 `/`、`/v1/models` 或 `/v1/embeddings`，请求 `/v1/embeddings` | 默认服务端原生维度；`.with_server_dimensions(...)` 会请求指定维度，实际能力取决于部署和模型。可用 `.with_api_key(SecretString)` 设置可选 API key。 |
| llama.cpp | `endpoint, model_identity, pipeline_id, expected_dimensions, media_marker` | 接受 `/`、`/v1/models` 或 `/v1/embeddings`，请求 `/v1/embeddings` | 仅使用原生维度。`model_identity` 用于标识 embedding 空间，不会发送给 llama.cpp。 |
| DashScope | `full_native_post_url, model, pipeline_id, native_dimensions, api_key` | 必须提供完整的原生 POST URL | API key 必填。`.with_server_dimensions(...)` 只接受 256、512、768、1024、1536、2048 或 2560。 |
| SGLang | `endpoint, model, pipeline_id, native_dimensions` | 需要 SGLang v0.5.11+；接受 `/`、`/v1/models` 或 `/v1/embeddings`，请求 `/v1/embeddings` | 仅原生维度；API key 可选。结构化输入由 SGLang Jinja 处理，客户端不手工包装 Qwen 文本。 |

以上端点路径不接受任意 proxy 前缀。DashScope 仍要求完整的原生 URL。

所有配置都支持 `.with_timeouts(connect_timeout, request_timeout)`。每种配置也都有 `.allow_remote_http(true)`，仅在明确需要连接非 loopback 的明文 HTTP 部署时使用。

## 响应、错误与安全默认值

`BatchEmbeddingResponse` 提供 `embeddings()`/`into_embeddings()`；`EmbeddingResponse` 提供 `embedding()`/`into_embedding()`；两者均提供 `space()`、`usage()` 与 `diagnostics()` 元数据。

`Error` 区分配置或输入错误、传输失败、provider API 错误、无效响应与维度不匹配。成功响应会验证索引完整且不重复、维度符合预期、向量有限且非零，随后由库在本地进行 L2 归一化。vLLM 请求不再发送已移除的 `normalize` 参数；请求维度能力仍由部署和模型决定。

默认接受 HTTPS。明文 HTTP 仅对 loopback 端点默认允许；远程端点必须显式调用 `.allow_remote_http(true)`。远程媒体 URL 只做语法校验，SDK 不会获取媒体；HTTP(S) URL 可包含签名 query，但禁止 userinfo 和 fragment。负责获取媒体的服务端必须限制出站访问及重定向。SDK 端点请求禁用重定向，并不能保护服务端的媒体获取。相关 `Debug` 输出会隐藏 API key 和内容。

## E2E 测试

真实 E2E 测试默认忽略，且**只读取进程环境变量**；库本身不会加载 `.env`。可将 `env.example` 作为仅含变量名的模板，导出当前后端需要的变量，且不要提交密钥。测试还会执行三张 COCO 图像到描述的检索 sanity check：使用固定余弦阈值并断言 Top-1；样本归属见 [`samples/coco-mini/README.md`](samples/coco-mini/README.md)。

| 后端 | 必填变量 | 可选变量 |
| --- | --- | --- |
| vLLM | `E2E_IMAGE_PATH`、`VLLM_MODELS_URL`（或 `VLLM_BASE_URL`）、`VLLM_CONFIGURED_MODEL`、`VLLM_PIPELINE_ID`、`VLLM_DIMENSIONS` | `VLLM_API_KEY`、`VLLM_SEND_DIMENSIONS`、`VLLM_ALLOW_REMOTE_HTTP` |
| llama.cpp | `E2E_IMAGE_PATH`、`LLAMA_CPP_MODELS_URL`、`LLAMA_CPP_CONFIGURED_MODEL`、`LLAMA_CPP_PIPELINE_ID`、`LLAMA_CPP_DIMENSIONS` | `LLAMA_CPP_MEDIA_MARKER`、`LLAMA_CPP_ALLOW_REMOTE_HTTP` |
| DashScope 官方端点 | `E2E_IMAGE_PATH`、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`DASHSCOPE_CONFIGURED_MODEL`、`DASHSCOPE_PIPELINE_ID`、`DASHSCOPE_NATIVE_DIMENSIONS` | `DASHSCOPE_DIMENSIONS` |
| DashScope 企业端点 | `E2E_IMAGE_PATH`、`DASHSCOPE_EMBEDDING_API_KEY`、`DASHSCOPE_EMBEDDING_BASE_URL`、`DASHSCOPE_ENTERPRISE_CONFIGURED_MODEL`、`DASHSCOPE_ENTERPRISE_PIPELINE_ID`、`DASHSCOPE_ENTERPRISE_NATIVE_DIMENSIONS` | `DASHSCOPE_ENTERPRISE_DIMENSIONS`、`DASHSCOPE_ENTERPRISE_ALLOW_REMOTE_HTTP` |
| SGLang | `E2E_IMAGE_PATH`、`SGLANG_MODELS_URL`（或 `SGLANG_BASE_URL`）、`SGLANG_MODEL`、`SGLANG_PIPELINE_ID`、`SGLANG_DIMENSIONS`、`SGLANG_VIDEO_SOURCE` | `SGLANG_API_KEY`、`SGLANG_ALLOW_REMOTE_HTTP` |

未设置 `LLAMA_CPP_MEDIA_MARKER` 时，E2E 会探测所配置 llama.cpp 端点的 `/props` 路由。可运行单个 provider，或运行所有已配置的 ignored 测试：

```sh
cargo test --test e2e vllm_e2e -- --ignored
cargo test --no-default-features --features sglang --test e2e sglang_e2e -- --ignored
cargo test --test e2e -- --ignored
```

SGLang 目前只有本地 HTTP contract 测试，尚无真实 SGLang E2E 结果。

## 开发与发布前检查

CI 会在推送到 `main` 和 pull request 时运行这些检查。后续发布应先 bump 并合并 `Cargo.toml` version，再推送匹配的 `vX.Y.Z` tag。crates.io Trusted Publisher 必须配置 owner `agentsyaml`、repo `fusion-embedding`、workflow `release.yml` 与 environment `release`。`0.1.0` 已手工发布，请勿为它创建 release tag。

```sh
cargo fmt --check
cargo test
cargo package
```
