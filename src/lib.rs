//! Native vLLM, llama.cpp, DashScope, and SGLang clients with fixed Qwen3-VL wire profiles.

mod error;
mod types;

#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
mod backend;
#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
mod client;
#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
mod transport;
#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
mod validation;

pub use error::Error;
pub use types::{
    BatchEmbeddingResponse, Diagnostics, DimensionMode, EmbedOptions, EmbeddingResponse,
    EmbeddingSpace, MultimodalPart, Provider, Usage,
};

#[cfg(feature = "dashscope")]
pub use backend::dashscope::DashScopeConfig;
#[cfg(feature = "llama-cpp")]
pub use backend::llama_cpp::LlamaCppConfig;
#[cfg(feature = "sglang")]
pub use backend::sglang::SglangConfig;
#[cfg(feature = "vllm")]
pub use backend::vllm::VllmConfig;
#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
pub use client::{BackendConfig, Client};
