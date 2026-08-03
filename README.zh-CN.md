# fusion-embedding

[English](README.md)

`fusion-embedding` 是一个精简的 Rust 客户端，为 vLLM、llama.cpp 与 DashScope 的 Qwen3-VL embedding 部署提供固定的请求协议。一个 `Client` 在创建时绑定一个后端、端点、部署标识和预期向量维度。

## 后端与 feature

| Feature | 后端 | 构造方式 |
| --- | --- | --- |
| `vllm` | vLLM 的 OpenAI 兼容 embedding 接口 | `Client::vllm(VllmConfig)` |
| `llama-cpp` | llama.cpp embedding 接口 | `Client::llama_cpp(LlamaCppConfig)` |
| `dashscope` | DashScope 原生多模态 embedding 接口 | `Client::dashscope(DashScopeConfig)` |

默认启用全部 feature。只使用一个后端时可关闭默认 feature：

```toml
[dependencies]
fusion-embedding = { version = "0.1", default-features = false, features = ["vllm"] }
```

## 范围与限制

公共 API 提供非空文本批量调用 `embed_texts`，以及接受有序 `ContentPart::{Text, ImageDataUrl}` 的 `embed_fusion`。图片**只能是规范的 PNG 或 JPEG base64 data URL**。库不做文件 I/O；应由应用程序先读取、编码图片再传入。远程媒体 URL、本地文件路径、视频及其他媒体类型都不被接受。

vLLM 与 llama.cpp 的图文调用要求恰好一张图片。DashScope 最多支持五张图片（也允许仅含文本的内容），内容项为 1 到 20 个。客户端会固定模型标识、pipeline ID 与预期维度；`EmbedOptions` 仅能修改单次调用的超时。

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
    .embed_texts(&["一段简短文本".into()], &EmbedOptions::default())
    .await?;
println!("{}", response.embeddings()[0].len());
# Ok(())
# }
```

图文融合时，在库外构造 data URL，并按需要保留内容顺序：

```rust,no_run
# use fusion_embedding::{Client, ContentPart, EmbedOptions};
# async fn example(client: Client, image_data_url: String) -> Result<(), fusion_embedding::Error> {
let response = client
    .embed_fusion(
        &[
            ContentPart::ImageDataUrl(image_data_url),
            ContentPart::Text("一个红色方块".into()),
        ],
        &EmbedOptions::default(),
    )
    .await?;
# let _ = response;
# Ok(())
# }
```

传给 `ImageDataUrl` 的值形如 `data:image/png;base64,<规范-base64>` 或 `data:image/jpeg;base64,<规范-base64>`。请使用应用程序选择的 base64 编码器处理图片字节；不要传入路径或 URL。

## 三种后端的配置差异

| 后端 | `new` 参数 | 端点行为 | 维度与鉴权 |
| --- | --- | --- | --- |
| vLLM | `base_endpoint, model, pipeline_id, expected_dimensions` | 只接受 `/` 或 `/v1/models`，随后请求 `/v1/embeddings` | 默认服务端原生维度；`.with_server_dimensions(...)` 会请求指定维度。可用 `.with_api_key(SecretString)` 设置可选 API key。 |
| llama.cpp | `endpoint, model_identity, pipeline_id, expected_dimensions, media_marker` | 只接受 `/` 或 `/v1/models`，随后请求 `/v1/embeddings` | 仅使用原生维度。`model_identity` 用于标识 embedding 空间，不会发送给 llama.cpp。 |
| DashScope | `full_native_post_url, model, pipeline_id, native_dimensions, api_key` | 必须提供完整的原生 POST URL | API key 必填。`.with_server_dimensions(...)` 只接受 256、512、768、1024、1536、2048 或 2560。 |

所有配置都支持 `.with_timeouts(connect_timeout, request_timeout)`。每种配置也都有 `.allow_remote_http(true)`，仅在明确需要连接非 loopback 的明文 HTTP 部署时使用。

## 响应、错误与安全默认值

`EmbeddingResponse` 提供有序的 `embeddings()`、`space()`、`usage()` 与 `diagnostics()`。`EmbeddingSpace` 记录 provider、配置的模型、pipeline ID、协议版本、维度模式和预期维度。`Usage` 以及请求/响应 ID 都可能不存在。

`Error` 区分配置或输入错误、传输失败、provider API 错误、无效响应与维度不匹配。成功响应会验证索引完整且不重复、维度符合预期、向量有限且非零，随后由库进行 L2 归一化。

默认接受 HTTPS。明文 HTTP 仅对 loopback 端点默认允许；远程端点必须显式调用 `.allow_remote_http(true)`。客户端不跟随重定向。相关 `Debug` 输出会隐藏 API key 和内容。

## E2E 测试

真实 E2E 测试默认忽略，且**只读取进程环境变量**；库本身不会加载 `.env`。可将 `env.example` 作为仅含变量名的模板，导出当前后端需要的变量，且不要提交密钥。测试还会执行三张 COCO 图像到描述的检索 sanity check：使用固定余弦阈值并断言 Top-1；样本归属见 [`samples/coco-mini/README.md`](samples/coco-mini/README.md)。

| 后端 | 必填变量 | 可选变量 |
| --- | --- | --- |
| vLLM | `E2E_IMAGE_PATH`、`VLLM_MODELS_URL`（或 `VLLM_BASE_URL`）、`VLLM_CONFIGURED_MODEL`、`VLLM_PIPELINE_ID`、`VLLM_DIMENSIONS` | `VLLM_API_KEY`、`VLLM_SEND_DIMENSIONS`、`VLLM_ALLOW_REMOTE_HTTP` |
| llama.cpp | `E2E_IMAGE_PATH`、`LLAMA_CPP_MODELS_URL`、`LLAMA_CPP_CONFIGURED_MODEL`、`LLAMA_CPP_PIPELINE_ID`、`LLAMA_CPP_DIMENSIONS` | `LLAMA_CPP_MEDIA_MARKER`、`LLAMA_CPP_ALLOW_REMOTE_HTTP` |
| DashScope 官方端点 | `E2E_IMAGE_PATH`、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`DASHSCOPE_CONFIGURED_MODEL`、`DASHSCOPE_PIPELINE_ID`、`DASHSCOPE_NATIVE_DIMENSIONS` | `DASHSCOPE_DIMENSIONS` |
| DashScope 企业端点 | `E2E_IMAGE_PATH`、`DASHSCOPE_EMBEDDING_API_KEY`、`DASHSCOPE_EMBEDDING_BASE_URL`、`DASHSCOPE_ENTERPRISE_CONFIGURED_MODEL`、`DASHSCOPE_ENTERPRISE_PIPELINE_ID`、`DASHSCOPE_ENTERPRISE_NATIVE_DIMENSIONS` | `DASHSCOPE_ENTERPRISE_DIMENSIONS`、`DASHSCOPE_ENTERPRISE_ALLOW_REMOTE_HTTP` |

未设置 `LLAMA_CPP_MEDIA_MARKER` 时，E2E 会探测所配置 llama.cpp 端点的 `/props` 路由。可运行单个 provider，或运行所有已配置的 ignored 测试：

```sh
cargo test --test e2e vllm_e2e -- --ignored
cargo test --test e2e -- --ignored
```

## 开发与发布前检查

```sh
cargo fmt --check
cargo test
cargo package
```
