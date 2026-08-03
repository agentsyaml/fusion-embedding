//! Native vLLM, llama.cpp, and DashScope clients with fixed Qwen3-VL wire profiles.

mod error;
mod types;

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
mod backend;
#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
mod client;
#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
mod transport;
#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
mod validation;

pub use error::Error;
pub use types::{
    ContentPart, Diagnostics, DimensionMode, EmbedOptions, EmbeddingResponse, EmbeddingSpace,
    Provider, Usage,
};

#[cfg(feature = "dashscope")]
pub use backend::dashscope::DashScopeConfig;
#[cfg(feature = "llama-cpp")]
pub use backend::llama_cpp::LlamaCppConfig;
#[cfg(feature = "vllm")]
pub use backend::vllm::VllmConfig;
#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub use client::Client;
