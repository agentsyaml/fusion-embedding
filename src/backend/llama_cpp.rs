use std::{fmt, num::NonZeroUsize, time::Duration};

use serde::Serialize;
use url::Url;

#[cfg(test)]
use super::format_qwen_text_inputs;
use super::{
    DEFAULT_CONNECT_TIMEOUT, DEFAULT_REQUEST_TIMEOUT, EncodingFormat, MAX_MODEL_BYTES,
    MAX_PIPELINE_ID_BYTES,
};
use crate::transport::validate_identifier;
use crate::validation::validate_data_url;
use crate::{Error, MultimodalPart};

pub(crate) const PROFILE_REVISION: &str = "qwen3-vl-embedding-llama-cpp-v2";
const EMBEDDINGS_PATH: &str = "/v1/embeddings";
const MAX_MEDIA_MARKER_BYTES: usize = 256;
const SYSTEM_PROMPT: &str = "Represent the user's input.";

/// llama.cpp client configuration with a caller-pinned media marker.
///
/// The endpoint must have path `/` or exactly `/v1/models`; both forms are
/// normalized to the same origin's `/v1/embeddings`. `configured_model` is an
/// embedding-space identity label and is never sent to llama.cpp.
#[cfg(feature = "llama-cpp")]
pub struct LlamaCppConfig {
    pub(crate) endpoint: Url,
    pub(crate) configured_model: String,
    pub(crate) pipeline_id: String,
    pub(crate) expected_dimensions: NonZeroUsize,
    pub(crate) media_marker: String,
    pub(crate) connect_timeout: Duration,
    pub(crate) request_timeout: Duration,
    pub(crate) allow_remote_http: bool,
}

#[cfg(feature = "llama-cpp")]
impl LlamaCppConfig {
    pub fn new(
        endpoint: Url,
        configured_model: impl Into<String>,
        pipeline_id: impl Into<String>,
        expected_dimensions: NonZeroUsize,
        media_marker: impl Into<String>,
    ) -> Result<Self, Error> {
        let configured_model = configured_model.into();
        let pipeline_id = pipeline_id.into();
        let media_marker = media_marker.into();
        validate_identifier("configured model", &configured_model, MAX_MODEL_BYTES)?;
        validate_identifier("pipeline_id", &pipeline_id, MAX_PIPELINE_ID_BYTES)?;
        validate_identifier("media marker", &media_marker, MAX_MEDIA_MARKER_BYTES)?;

        Ok(Self {
            endpoint: normalize_llama_endpoint(endpoint)?,
            configured_model,
            pipeline_id,
            expected_dimensions,
            media_marker,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            allow_remote_http: false,
        })
    }

    pub fn with_timeouts(
        mut self,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<Self, Error> {
        if connect_timeout.is_zero() || request_timeout.is_zero() {
            return Err(Error::Config {
                message: "timeouts must be greater than zero".into(),
            });
        }
        self.connect_timeout = connect_timeout;
        self.request_timeout = request_timeout;
        Ok(self)
    }

    /// Explicitly permits cleartext HTTP to a non-loopback endpoint.
    pub fn allow_remote_http(mut self, allow: bool) -> Self {
        self.allow_remote_http = allow;
        self
    }
}

#[cfg(feature = "llama-cpp")]
impl fmt::Debug for LlamaCppConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlamaCppConfig")
            .field("endpoint", &self.endpoint)
            .field("configured_model", &self.configured_model)
            .field("pipeline_id", &self.pipeline_id)
            .field("expected_dimensions", &self.expected_dimensions)
            .field("media_marker", &"<redacted>")
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .field("allow_remote_http", &self.allow_remote_http)
            .finish()
    }
}

#[cfg(feature = "llama-cpp")]
#[derive(Serialize)]
pub(crate) struct LlamaTextRequest<'a> {
    pub(crate) input: &'a [String],
    pub(crate) encoding_format: EncodingFormat,
    pub(crate) embd_normalize: u8,
}

#[cfg(feature = "llama-cpp")]
#[derive(Serialize)]
pub(crate) struct LlamaFusionRequest<'a> {
    input: LlamaFusionInput<'a>,
    encoding_format: EncodingFormat,
    embd_normalize: u8,
}

#[cfg(feature = "llama-cpp")]
#[derive(Serialize)]
struct LlamaFusionInput<'a> {
    pub(crate) prompt_string: String,
    pub(crate) multimodal_data: Vec<&'a str>,
}

#[cfg(feature = "llama-cpp")]
pub(crate) fn build_llama_fusion_request<'a>(
    media_marker: &str,
    parts: &'a [MultimodalPart],
) -> Result<LlamaFusionRequest<'a>, Error> {
    if parts.is_empty() {
        return Err(Error::InvalidInput {
            message: "fusion input must not be empty".into(),
        });
    }
    let mut image_count = 0;
    let mut image_base64 = Vec::new();
    let mut atoms = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            MultimodalPart::Text(text) => {
                if text.is_empty() {
                    return Err(Error::InvalidInput {
                        message: "llama.cpp fusion text must not be empty".into(),
                    });
                }
                if text.contains(media_marker) {
                    return Err(Error::InvalidInput {
                        message: "llama.cpp fusion text collides with the configured media marker"
                            .into(),
                    });
                }
                atoms.push(text.as_str());
            }
            MultimodalPart::ImageDataUrl(url) => {
                image_count += 1;
                image_base64.push(validate_data_url(url)?);
                atoms.push(media_marker);
            }
            MultimodalPart::VideoSource(_) => {
                return Err(Error::InvalidInput {
                    message: "llama.cpp does not support VideoSource".into(),
                });
            }
        }
    }
    if image_count == 0 {
        return Err(Error::InvalidInput {
            message: "fusion input must contain at least one image".into(),
        });
    }
    let ordered_atoms = atoms.join(" ");
    let prompt_string = format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{ordered_atoms}<|im_end|>\n<|im_start|>assistant\n"
    );
    if prompt_string.match_indices(media_marker).count() != image_count {
        return Err(Error::InvalidInput {
            message: "configured media marker is ambiguous in the llama.cpp prompt".into(),
        });
    }
    Ok(LlamaFusionRequest {
        input: LlamaFusionInput {
            prompt_string,
            multimodal_data: image_base64,
        },
        encoding_format: EncodingFormat::Float,
        embd_normalize: 2,
    })
}

#[cfg(feature = "llama-cpp")]
pub(crate) fn normalize_llama_endpoint(mut endpoint: Url) -> Result<Url, Error> {
    if endpoint.cannot_be_a_base()
        || endpoint.host().is_none()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(Error::Config {
            message: "endpoint must be an absolute URL without userinfo, query, or fragment".into(),
        });
    }
    if !matches!(endpoint.path(), "/" | "/v1/models") {
        return Err(Error::Config {
            message: "llama.cpp endpoint path must be / or /v1/models".into(),
        });
    }
    endpoint.set_path(EMBEDDINGS_PATH);
    Ok(endpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Client, Error, MultimodalPart};
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde_json::json;
    use url::Url;

    fn png_data_url(width: u32, height: u32) -> String {
        let mut bytes = vec![137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13];
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
        bytes.extend_from_slice(&[0; 4]);
        format!("data:image/png;base64,{}", STANDARD.encode(bytes))
    }

    #[test]
    fn llama_text_payload_uses_qwen_chat_profile_and_preserves_batch_order() {
        let input = vec!["first".to_owned(), "second".to_owned()];
        let formatted = format_qwen_text_inputs(&input);
        let request = LlamaTextRequest {
            input: &formatted,
            encoding_format: EncodingFormat::Float,
            embd_normalize: 2,
        };
        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "input": [
                    "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n"
                ],
                "encoding_format": "float",
                "embd_normalize": 2
            })
        );
    }
    #[test]
    fn llama_fusion_payload_preserves_atom_order_and_bare_base64_array() {
        let marker = "<marker>";
        let image = png_data_url(1, 1);
        let bare = image.split_once(',').unwrap().1;
        let cases = [
            (
                vec![
                    MultimodalPart::ImageDataUrl(image.clone()),
                    MultimodalPart::Text("caption".into()),
                ],
                "<marker> caption",
            ),
            (
                vec![
                    MultimodalPart::Text("caption".into()),
                    MultimodalPart::ImageDataUrl(image.clone()),
                ],
                "caption <marker>",
            ),
            (
                vec![
                    MultimodalPart::Text("first".into()),
                    MultimodalPart::Text("second".into()),
                    MultimodalPart::ImageDataUrl(image.clone()),
                ],
                "first second <marker>",
            ),
        ];
        for (parts, atoms) in cases {
            let value =
                serde_json::to_value(build_llama_fusion_request(marker, &parts).unwrap()).unwrap();
            assert_eq!(
                value,
                json!({
                    "input": {
                        "prompt_string": format!(
                            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{atoms}<|im_end|>\n<|im_start|>assistant\n"
                        ),
                        "multimodal_data": [bare]
                    },
                    "encoding_format": "float",
                    "embd_normalize": 2
                })
            );
            assert!(value.get("model").is_none());
            assert!(value.get("dimensions").is_none());
        }
        let second = png_data_url(2, 1);
        let payload = serde_json::to_value(
            build_llama_fusion_request(
                marker,
                &[
                    MultimodalPart::ImageDataUrl(image.clone()),
                    MultimodalPart::Text("between".into()),
                    MultimodalPart::ImageDataUrl(second.clone()),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            payload["input"]["prompt_string"]
                .as_str()
                .unwrap()
                .matches(marker)
                .count(),
            2
        );
        assert_eq!(
            payload["input"]["multimodal_data"],
            json!([bare, second.split_once(',').unwrap().1])
        );
    }
    #[test]
    fn llama_fusion_rejects_empty_text_marker_collision_and_wrong_image_count() {
        let marker = "<marker>";
        let image = png_data_url(1, 1);
        assert!(
            build_llama_fusion_request(
                marker,
                &[
                    MultimodalPart::ImageDataUrl(image.clone()),
                    MultimodalPart::Text(String::new())
                ]
            )
            .is_err()
        );
        assert!(
            build_llama_fusion_request(
                marker,
                &[
                    MultimodalPart::ImageDataUrl(image.clone()),
                    MultimodalPart::Text("contains <marker> collision".into())
                ]
            )
            .is_err()
        );
        for ambiguous_marker in ["user", " "] {
            let error = match build_llama_fusion_request(
                ambiguous_marker,
                &[MultimodalPart::ImageDataUrl(image.clone())],
            ) {
                Err(error) => error,
                Ok(_) => panic!("ambiguous marker was accepted"),
            };
            match error {
                Error::InvalidInput { message } => {
                    assert_eq!(
                        message,
                        "configured media marker is ambiguous in the llama.cpp prompt"
                    );
                }
                other => panic!("expected invalid input, got {other:?}"),
            }
        }
        assert!(
            build_llama_fusion_request(marker, &[MultimodalPart::Text("caption".into())]).is_err()
        );
        assert!(
            build_llama_fusion_request(
                marker,
                &[
                    MultimodalPart::ImageDataUrl(image.clone()),
                    MultimodalPart::ImageDataUrl(image)
                ]
            )
            .is_ok()
        );
    }
    #[test]
    fn llama_endpoint_normalizes_root_and_models_only() {
        for url in ["http://localhost:8080/", "http://localhost:8080/v1/models"] {
            let config = LlamaCppConfig::new(
                Url::parse(url).unwrap(),
                "identity-only-model",
                "pipeline",
                NonZeroUsize::new(2048).unwrap(),
                "<marker>",
            )
            .unwrap();
            assert_eq!(config.endpoint.path(), EMBEDDINGS_PATH);
            assert!(Client::new(config).is_ok());
        }
        for url in [
            "http://localhost:8080/v1/models/",
            "http://localhost:8080/v1/embeddings",
            "http://localhost:8080/props",
        ] {
            assert!(
                LlamaCppConfig::new(
                    Url::parse(url).unwrap(),
                    "m",
                    "p",
                    NonZeroUsize::new(2048).unwrap(),
                    "<marker>"
                )
                .is_err()
            );
        }
        assert!(
            LlamaCppConfig::new(
                Url::parse("http://localhost:8080/").unwrap(),
                "m",
                "p",
                NonZeroUsize::new(2048).unwrap(),
                "\n"
            )
            .is_err()
        );
    }
}
