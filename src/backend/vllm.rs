use std::{fmt, num::NonZeroUsize, time::Duration};

use secrecy::{ExposeSecret, SecretString};
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
use crate::{ContentPart, DimensionMode, Error};

pub(crate) const PROFILE_REVISION: &str = "qwen3-vl-embedding-chat-v2";
const EMBEDDINGS_PATH: &str = "/v1/embeddings";
const SYSTEM_PROMPT: &str = "Represent the user's input.";

/// vLLM client configuration. All deployment identity is fixed at construction.
///
/// The endpoint path must be `/` or exactly `/v1/models`; the client fixes the
/// request path to `/v1/embeddings`.
#[cfg(feature = "vllm")]
pub struct VllmConfig {
    pub(crate) endpoint: Url,
    pub(crate) configured_model: String,
    pub(crate) pipeline_id: String,
    pub(crate) expected_dimensions: NonZeroUsize,
    pub(crate) dimension_mode: DimensionMode,
    pub(crate) requested_dimensions: Option<NonZeroUsize>,
    pub(crate) api_key: Option<SecretString>,
    pub(crate) connect_timeout: Duration,
    pub(crate) request_timeout: Duration,
    pub(crate) allow_remote_http: bool,
}

#[cfg(feature = "vllm")]
impl VllmConfig {
    pub fn new(
        base_endpoint: Url,
        configured_model: impl Into<String>,
        pipeline_id: impl Into<String>,
        expected_dimensions: NonZeroUsize,
    ) -> Result<Self, Error> {
        let configured_model = configured_model.into();
        let pipeline_id = pipeline_id.into();
        validate_identifier("configured model", &configured_model, MAX_MODEL_BYTES)?;
        validate_identifier("pipeline_id", &pipeline_id, MAX_PIPELINE_ID_BYTES)?;

        Ok(Self {
            endpoint: normalize_endpoint(base_endpoint)?,
            configured_model,
            pipeline_id,
            expected_dimensions,
            dimension_mode: DimensionMode::Native,
            requested_dimensions: None,
            api_key: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            allow_remote_http: false,
        })
    }

    /// Requests a deployment-supported server dimension without fallback.
    pub fn with_server_dimensions(mut self, dimensions: NonZeroUsize) -> Self {
        self.expected_dimensions = dimensions;
        self.dimension_mode = DimensionMode::Server;
        self.requested_dimensions = Some(dimensions);
        self
    }

    pub fn with_api_key(mut self, api_key: SecretString) -> Result<Self, Error> {
        if api_key.expose_secret().is_empty() {
            return Err(Error::Config {
                message: "API key must not be empty".into(),
            });
        }
        self.api_key = Some(api_key);
        Ok(self)
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

#[cfg(feature = "vllm")]
impl fmt::Debug for VllmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VllmConfig")
            .field("endpoint", &self.endpoint)
            .field("configured_model", &self.configured_model)
            .field("pipeline_id", &self.pipeline_id)
            .field("expected_dimensions", &self.expected_dimensions)
            .field("dimension_mode", &self.dimension_mode)
            .field("requested_dimensions", &self.requested_dimensions)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .field("allow_remote_http", &self.allow_remote_http)
            .finish()
    }
}

#[cfg(feature = "vllm")]
#[derive(Serialize)]
pub(crate) struct TextRequest<'a> {
    pub(crate) model: &'a str,
    pub(crate) input: &'a [String],
    pub(crate) encoding_format: EncodingFormat,
    pub(crate) normalize: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dimensions: Option<usize>,
}

#[cfg(feature = "vllm")]
#[derive(Serialize)]
pub(crate) struct FusionRequest<'a> {
    model: &'a str,
    messages: [Message<'a>; 3],
    continue_final_message: bool,
    add_special_tokens: bool,
    normalize: bool,
    encoding_format: EncodingFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[cfg(feature = "vllm")]
#[derive(Serialize)]
struct Message<'a> {
    pub(crate) role: Role,
    pub(crate) content: Vec<MessagePart<'a>>,
}

#[cfg(feature = "vllm")]
#[derive(Serialize)]
#[serde(tag = "type")]
enum MessagePart<'a> {
    #[serde(rename = "text")]
    Text { text: &'a str },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl<'a> },
}

#[cfg(feature = "vllm")]
#[derive(Serialize)]
struct ImageUrl<'a> {
    pub(crate) url: &'a str,
}

#[cfg(feature = "vllm")]
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

#[cfg(feature = "vllm")]
pub(crate) fn build_fusion_request<'a>(
    model: &'a str,
    parts: &'a [ContentPart],
    dimensions: Option<NonZeroUsize>,
) -> Result<FusionRequest<'a>, Error> {
    if parts.is_empty() {
        return Err(Error::InvalidInput {
            message: "fusion input must not be empty".into(),
        });
    }
    let mut image_count = 0;
    let mut user_content = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ContentPart::Text(text) => user_content.push(MessagePart::Text { text }),
            ContentPart::ImageDataUrl(url) => {
                image_count += 1;
                validate_data_url(url)?;
                user_content.push(MessagePart::ImageUrl {
                    image_url: ImageUrl { url },
                });
            }
            ContentPart::VideoSource(_) => {
                return Err(Error::InvalidInput {
                    message: "vLLM does not support VideoSource".into(),
                });
            }
        }
    }
    if image_count != 1 {
        return Err(Error::InvalidInput {
            message: "fusion input must contain exactly one image".into(),
        });
    }

    Ok(FusionRequest {
        model,
        messages: [
            Message {
                role: Role::System,
                content: vec![MessagePart::Text {
                    text: SYSTEM_PROMPT,
                }],
            },
            Message {
                role: Role::User,
                content: user_content,
            },
            Message {
                role: Role::Assistant,
                content: vec![MessagePart::Text { text: "" }],
            },
        ],
        continue_final_message: true,
        add_special_tokens: true,
        normalize: true,
        encoding_format: EncodingFormat::Float,
        dimensions: dimensions.map(NonZeroUsize::get),
    })
}

#[cfg(feature = "vllm")]
pub(crate) fn normalize_endpoint(mut endpoint: Url) -> Result<Url, Error> {
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
            message: "vLLM endpoint path must be / or /v1/models".into(),
        });
    }
    endpoint.set_path(EMBEDDINGS_PATH);
    Ok(endpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Client, ContentPart, DimensionMode};
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
    fn text_payload_uses_qwen_chat_profile_and_preserves_batch_order() {
        let input = vec!["first".to_owned(), "second".to_owned()];
        let formatted = format_qwen_text_inputs(&input);
        let request = TextRequest {
            model: "configured",
            input: &formatted,
            encoding_format: EncodingFormat::Float,
            normalize: true,
            dimensions: None,
        };
        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "model": "configured",
                "input": [
                    "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n"
                ],
                "encoding_format": "float",
                "normalize": true
            })
        );
    }
    #[test]
    fn fusion_payload_matches_golden_and_preserves_order() {
        let image = png_data_url(1, 1);
        let parts = vec![
            ContentPart::Text("before".into()),
            ContentPart::ImageDataUrl(image.clone()),
            ContentPart::Text("after".into()),
        ];
        let value =
            serde_json::to_value(build_fusion_request("model", &parts, None).unwrap()).unwrap();
        assert_eq!(
            value,
            json!({
                "model": "model",
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": "after"}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]}
                ],
                "continue_final_message": true,
                "add_special_tokens": true,
                "normalize": true,
                "encoding_format": "float"
            })
        );
        assert!(value.get("dimensions").is_none());
        assert!(value.get("add_generation_prompt").is_none());
    }
    #[test]
    fn vllm_server_dimensions_are_explicit_in_text_fusion_and_space() {
        let dimensions = NonZeroUsize::new(1024).unwrap();
        let input = vec!["first".to_owned()];
        let formatted = format_qwen_text_inputs(&input);
        assert_eq!(
            serde_json::to_value(TextRequest {
                model: "configured",
                input: &formatted,
                encoding_format: EncodingFormat::Float,
                normalize: true,
                dimensions: Some(dimensions.get()),
            })
            .unwrap(),
            json!({
                "model": "configured",
                "input": ["<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n"],
                "encoding_format": "float",
                "normalize": true,
                "dimensions": 1024
            })
        );

        let image = png_data_url(1, 1);
        let parts = vec![
            ContentPart::ImageDataUrl(image.clone()),
            ContentPart::Text("caption".into()),
        ];
        let fusion = serde_json::to_value(
            build_fusion_request("configured", &parts, Some(dimensions)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            fusion,
            json!({
                "model": "configured",
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": "caption"}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]}
                ],
                "continue_final_message": true,
                "add_special_tokens": true,
                "normalize": true,
                "encoding_format": "float",
                "dimensions": 1024
            })
        );

        let config = VllmConfig::new(
            Url::parse("http://localhost:8000/v1/models").unwrap(),
            "configured",
            "pipeline",
            NonZeroUsize::new(2048).unwrap(),
        )
        .unwrap()
        .with_server_dimensions(dimensions);
        assert_eq!(config.dimension_mode, DimensionMode::Server);
        assert_eq!(config.expected_dimensions, dimensions);
        assert_eq!(config.requested_dimensions, Some(dimensions));
        let client = Client::vllm(config).unwrap();
        assert_eq!(client.space.dimension_mode(), DimensionMode::Server);
        assert_eq!(client.space.expected_dimensions(), dimensions);
    }
    #[test]
    fn fusion_requires_exactly_one_image() {
        assert!(build_fusion_request("m", &[], None).is_err());
        assert!(build_fusion_request("m", &[ContentPart::Text("x".into())], None).is_err());
        let image = png_data_url(1, 1);
        assert!(
            build_fusion_request(
                "m",
                &[
                    ContentPart::ImageDataUrl(image.clone()),
                    ContentPart::ImageDataUrl(image)
                ],
                None
            )
            .is_err()
        );
    }
    #[test]
    fn endpoint_security_and_fixed_path() {
        for endpoint in [
            Url::parse("http://localhost:8000/").unwrap(),
            Url::parse("http://localhost:8000/v1/models").unwrap(),
        ] {
            let config =
                VllmConfig::new(endpoint, "model", "pipeline", NonZeroUsize::new(3).unwrap())
                    .unwrap();
            assert_eq!(config.endpoint.path(), EMBEDDINGS_PATH);
            assert_eq!(config.dimension_mode, DimensionMode::Native);
            assert_eq!(config.expected_dimensions.get(), 3);
            assert_eq!(config.requested_dimensions, None);
            assert!(Client::vllm(config).is_ok());
        }
        assert!(
            VllmConfig::new(
                Url::parse("https://example.com/prefix").unwrap(),
                "m",
                "p",
                NonZeroUsize::new(1).unwrap()
            )
            .is_err()
        );

        for url in [
            "https://user@example.com",
            "https://example.com?secret=x",
            "https://example.com#fragment",
        ] {
            assert!(
                VllmConfig::new(
                    Url::parse(url).unwrap(),
                    "m",
                    "p",
                    NonZeroUsize::new(1).unwrap()
                )
                .is_err()
            );
        }
        let remote = VllmConfig::new(
            Url::parse("http://example.com").unwrap(),
            "m",
            "p",
            NonZeroUsize::new(1).unwrap(),
        )
        .unwrap();
        assert!(Client::vllm(remote).is_err());
        let opted_in = VllmConfig::new(
            Url::parse("http://example.com").unwrap(),
            "m",
            "p",
            NonZeroUsize::new(1).unwrap(),
        )
        .unwrap()
        .allow_remote_http(true);
        assert!(Client::vllm(opted_in).is_ok());
        assert!(
            VllmConfig::new(
                Url::parse("https://example.com").unwrap(),
                "m",
                " ",
                NonZeroUsize::new(1).unwrap()
            )
            .is_err()
        );
        assert!(
            VllmConfig::new(
                Url::parse("https://example.com").unwrap(),
                "m",
                "p".repeat(MAX_PIPELINE_ID_BYTES + 1),
                NonZeroUsize::new(1).unwrap()
            )
            .is_err()
        );
    }
}
