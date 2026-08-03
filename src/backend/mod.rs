use std::time::Duration;

#[cfg(any(feature = "vllm", feature = "llama-cpp"))]
use serde::Serialize;

pub(crate) const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub(crate) const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
pub(crate) const MAX_PIPELINE_ID_BYTES: usize = 256;
pub(crate) const MAX_MODEL_BYTES: usize = 512;

#[cfg(any(feature = "vllm", feature = "llama-cpp"))]
pub(crate) fn format_qwen_text_inputs(texts: &[String]) -> Vec<String> {
    texts
        .iter()
        .map(|text| {
            format!(
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            )
        })
        .collect()
}

#[cfg(feature = "dashscope")]
pub(crate) mod dashscope;
#[cfg(feature = "llama-cpp")]
pub(crate) mod llama_cpp;
#[cfg(feature = "vllm")]
pub(crate) mod vllm;

#[cfg(any(feature = "vllm", feature = "llama-cpp"))]
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EncodingFormat {
    Float,
}
