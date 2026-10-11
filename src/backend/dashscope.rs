use std::{fmt, num::NonZeroUsize, time::Duration};

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use url::Url;

use super::{
    DEFAULT_CONNECT_TIMEOUT, DEFAULT_REQUEST_TIMEOUT, MAX_MODEL_BYTES, MAX_PIPELINE_ID_BYTES,
};
use crate::transport::{MAX_DIAGNOSTIC_CHARS, sanitize_metadata, validate_identifier};
use crate::validation::{
    IndexedEmbedding, validate_data_url, validate_embeddings, validate_media_url,
};
use crate::{
    BatchEmbeddingResponse, Diagnostics, DimensionMode, EmbeddingSpace, Error, MultimodalPart,
    Usage,
};

pub(crate) const PROFILE_REVISION: &str = "qwen3-vl-embedding-dashscope-native-v1";
pub(crate) const MAX_DASHSCOPE_CONTENTS: usize = 20;
const MAX_DASHSCOPE_IMAGES: usize = 10;
const MAX_DASHSCOPE_VIDEOS: usize = 1;
const DASHSCOPE_DIMENSIONS: [usize; 7] = [256, 512, 768, 1024, 1536, 2048, 2560];

/// DashScope configuration for an exact native multimodal embedding endpoint.
#[cfg(feature = "dashscope")]
pub struct DashScopeConfig {
    pub(crate) endpoint: Url,
    pub(crate) configured_model: String,
    pub(crate) pipeline_id: String,
    pub(crate) expected_dimensions: NonZeroUsize,
    pub(crate) dimension_mode: DimensionMode,
    pub(crate) requested_dimensions: Option<NonZeroUsize>,
    pub(crate) api_key: SecretString,
    pub(crate) connect_timeout: Duration,
    pub(crate) request_timeout: Duration,
    pub(crate) allow_remote_http: bool,
}

#[cfg(feature = "dashscope")]
impl DashScopeConfig {
    pub fn new(
        full_endpoint: Url,
        configured_model: impl Into<String>,
        pipeline_id: impl Into<String>,
        expected_native_dimensions: NonZeroUsize,
        api_key: SecretString,
    ) -> Result<Self, Error> {
        let configured_model = configured_model.into();
        let pipeline_id = pipeline_id.into();
        validate_identifier("configured model", &configured_model, MAX_MODEL_BYTES)?;
        validate_identifier("pipeline_id", &pipeline_id, MAX_PIPELINE_ID_BYTES)?;
        validate_dashscope_endpoint(&full_endpoint)?;
        if api_key.expose_secret().is_empty() {
            return Err(Error::Config {
                message: "API key must not be empty".into(),
            });
        }
        Ok(Self {
            endpoint: full_endpoint,
            configured_model,
            pipeline_id,
            expected_dimensions: expected_native_dimensions,
            dimension_mode: DimensionMode::Native,
            requested_dimensions: None,
            api_key,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            allow_remote_http: false,
        })
    }

    pub fn with_server_dimensions(mut self, dimensions: NonZeroUsize) -> Result<Self, Error> {
        if !DASHSCOPE_DIMENSIONS.contains(&dimensions.get()) {
            return Err(Error::Config {
                message: "unsupported DashScope server dimension".into(),
            });
        }
        self.expected_dimensions = dimensions;
        self.dimension_mode = DimensionMode::Server;
        self.requested_dimensions = Some(dimensions);
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

#[cfg(feature = "dashscope")]
impl fmt::Debug for DashScopeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DashScopeConfig")
            .field("endpoint", &self.endpoint)
            .field("configured_model", &self.configured_model)
            .field("pipeline_id", &self.pipeline_id)
            .field("expected_dimensions", &self.expected_dimensions)
            .field("dimension_mode", &self.dimension_mode)
            .field("api_key", &"<redacted>")
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .field("allow_remote_http", &self.allow_remote_http)
            .finish()
    }
}

#[cfg(feature = "dashscope")]
#[derive(Clone, Copy)]
pub(crate) enum DashScopeOperation {
    Text,
    Fusion,
}

#[cfg(feature = "dashscope")]
impl DashScopeOperation {
    fn expected_type(self) -> &'static str {
        match self {
            Self::Text => "vl",
            Self::Fusion => "fusion",
        }
    }
}

#[cfg(feature = "dashscope")]
#[derive(Serialize)]
pub(crate) struct DashScopeRequest<'a> {
    model: &'a str,
    input: DashScopeInput<'a>,
    parameters: DashScopeParameters,
}

#[cfg(feature = "dashscope")]
#[derive(Serialize)]
struct DashScopeInput<'a> {
    pub(crate) contents: Vec<DashScopeContent<'a>>,
}

#[cfg(feature = "dashscope")]
#[derive(Serialize)]
#[serde(untagged)]
enum DashScopeContent<'a> {
    Text { text: &'a str },
    Image { image: &'a str },
    Video { video: &'a str },
}

#[cfg(feature = "dashscope")]
#[derive(Serialize)]
struct DashScopeParameters {
    pub(crate) enable_fusion: bool,
    pub(crate) output_type: DashScopeOutputType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dimension: Option<usize>,
}

#[cfg(feature = "dashscope")]
#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum DashScopeOutputType {
    Dense,
}

#[cfg(feature = "dashscope")]
#[derive(Deserialize)]
struct DashScopeWireResponse {
    output: DashScopeWireOutput,
    #[serde(default)]
    usage: Option<DashScopeWireUsage>,
    #[serde(default)]
    request_id: Option<String>,
}

#[cfg(feature = "dashscope")]
#[derive(Deserialize)]
struct DashScopeWireOutput {
    embeddings: Vec<DashScopeWireEmbedding>,
}

#[cfg(feature = "dashscope")]
#[derive(Deserialize)]
struct DashScopeWireEmbedding {
    index: usize,
    embedding: Vec<f32>,
    #[serde(rename = "type")]
    kind: String,
}

#[cfg(feature = "dashscope")]
#[derive(Deserialize)]
struct DashScopeWireUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default, rename = "image_tokens")]
    _image_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[cfg(feature = "dashscope")]
pub(crate) fn build_dashscope_text_request<'a>(
    model: &'a str,
    texts: &'a [String],
    dimensions: Option<NonZeroUsize>,
) -> DashScopeRequest<'a> {
    DashScopeRequest {
        model,
        input: DashScopeInput {
            contents: texts
                .iter()
                .map(|text| DashScopeContent::Text { text })
                .collect(),
        },
        parameters: DashScopeParameters {
            enable_fusion: false,
            output_type: DashScopeOutputType::Dense,
            dimension: dimensions.map(NonZeroUsize::get),
        },
    }
}

#[cfg(feature = "dashscope")]
pub(crate) fn build_dashscope_fusion_request<'a>(
    model: &'a str,
    parts: &'a [MultimodalPart],
    dimensions: Option<NonZeroUsize>,
) -> Result<DashScopeRequest<'a>, Error> {
    if parts.is_empty() || parts.len() > MAX_DASHSCOPE_CONTENTS {
        return Err(Error::InvalidInput {
            message: format!(
                "DashScope fusion input must contain 1 to {MAX_DASHSCOPE_CONTENTS} items"
            ),
        });
    }
    let mut image_count = 0;
    let mut video_count = 0;
    let mut contents = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            MultimodalPart::Text(text) => contents.push(DashScopeContent::Text { text }),
            MultimodalPart::ImageDataUrl(image) => {
                image_count += 1;
                if image_count > MAX_DASHSCOPE_IMAGES {
                    return Err(Error::InvalidInput {
                        message: format!(
                            "DashScope fusion input must contain at most {MAX_DASHSCOPE_IMAGES} images"
                        ),
                    });
                }
                validate_data_url(image)?;
                contents.push(DashScopeContent::Image { image });
            }
            MultimodalPart::ImageUrl(url) => {
                image_count += 1;
                if image_count > MAX_DASHSCOPE_IMAGES {
                    return Err(Error::InvalidInput {
                        message: format!(
                            "DashScope fusion input must contain at most {MAX_DASHSCOPE_IMAGES} images"
                        ),
                    });
                }
                validate_media_url(url)?;
                contents.push(DashScopeContent::Image {
                    image: url.as_str(),
                });
            }
            MultimodalPart::VideoUrl(url) => {
                video_count += 1;
                if video_count > MAX_DASHSCOPE_VIDEOS {
                    return Err(Error::InvalidInput {
                        message: "DashScope fusion input must contain at most one video".into(),
                    });
                }
                validate_media_url(url)?;
                contents.push(DashScopeContent::Video {
                    video: url.as_str(),
                });
            }
            MultimodalPart::VideoSource(_) => {
                return Err(Error::Unsupported {
                    provider: crate::Provider::DashScope,
                    feature: "VideoSource; use VideoUrl",
                });
            }
        }
    }
    Ok(DashScopeRequest {
        model,
        input: DashScopeInput { contents },
        parameters: DashScopeParameters {
            enable_fusion: true,
            output_type: DashScopeOutputType::Dense,
            dimension: dimensions.map(NonZeroUsize::get),
        },
    })
}

#[cfg(feature = "dashscope")]
pub(crate) fn validate_dashscope_endpoint(endpoint: &Url) -> Result<(), Error> {
    if endpoint.cannot_be_a_base()
        || endpoint.host().is_none()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
        || !matches!(endpoint.scheme(), "http" | "https")
    {
        return Err(Error::Config {
            message: "DashScope endpoint must be an absolute HTTP(S) URL without userinfo, query, or fragment".into(),
        });
    }
    if endpoint.path() == "/" {
        return Err(Error::Config {
            message: "DashScope endpoint must include the full native POST path".into(),
        });
    }
    Ok(())
}

#[cfg(feature = "dashscope")]
pub(crate) fn parse_dashscope_response(
    body: &[u8],
    expected_count: usize,
    space: &EmbeddingSpace,
    operation: DashScopeOperation,
) -> Result<BatchEmbeddingResponse, Error> {
    let wire: DashScopeWireResponse =
        serde_json::from_slice(body).map_err(|_| Error::InvalidResponse {
            message: "response does not match the DashScope schema".into(),
        })?;
    if wire
        .output
        .embeddings
        .iter()
        .any(|item| item.kind != operation.expected_type())
    {
        return Err(Error::InvalidResponse {
            message: "DashScope embedding type does not match the requested operation".into(),
        });
    }
    let embeddings = validate_embeddings(
        wire.output
            .embeddings
            .into_iter()
            .map(|item| IndexedEmbedding {
                index: item.index,
                embedding: item.embedding,
            })
            .collect(),
        expected_count,
        space.expected_dimensions.get(),
    )?;
    Ok(BatchEmbeddingResponse {
        embeddings,
        space: space.clone(),
        usage: Usage {
            input_tokens: wire.usage.as_ref().and_then(|usage| usage.input_tokens),
            total_tokens: wire.usage.and_then(|usage| usage.total_tokens),
        },
        diagnostics: Diagnostics {
            request_id: wire
                .request_id
                .as_deref()
                .and_then(|value| sanitize_metadata(value, MAX_DIAGNOSTIC_CHARS)),
            response_id: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Client, DimensionMode, Error, MultimodalPart, Provider};
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde_json::json;

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
    fn dashscope_text_payload_native_and_server_goldens() {
        let texts = vec!["first".to_owned(), "second".to_owned()];
        let native =
            serde_json::to_value(build_dashscope_text_request("model", &texts, None)).unwrap();
        assert_eq!(
            native,
            json!({
                "model": "model",
                "input": {"contents": [{"text": "first"}, {"text": "second"}]},
                "parameters": {"enable_fusion": false, "output_type": "dense"}
            })
        );
        assert!(native["parameters"].get("dimension").is_none());

        assert_eq!(
            serde_json::to_value(build_dashscope_text_request(
                "model",
                &texts,
                NonZeroUsize::new(512)
            ))
            .unwrap(),
            json!({
                "model": "model",
                "input": {"contents": [{"text": "first"}, {"text": "second"}]},
                "parameters": {
                    "enable_fusion": false,
                    "output_type": "dense",
                    "dimension": 512
                }
            })
        );
    }
    #[test]
    fn dashscope_fusion_preserves_contents_and_duplicate_images() {
        let image = png_data_url(1, 1);
        for (parts, expected) in [
            (
                vec![
                    MultimodalPart::ImageDataUrl(image.clone()),
                    MultimodalPart::Text("caption".into()),
                ],
                json!([{"image": image}, {"text": "caption"}]),
            ),
            (
                vec![
                    MultimodalPart::Text("caption".into()),
                    MultimodalPart::ImageDataUrl(image.clone()),
                ],
                json!([{"text": "caption"}, {"image": image}]),
            ),
            (
                vec![
                    MultimodalPart::ImageDataUrl(image.clone()),
                    MultimodalPart::ImageDataUrl(image.clone()),
                ],
                json!([{"image": image}, {"image": image}]),
            ),
        ] {
            let value = serde_json::to_value(
                build_dashscope_fusion_request("model", &parts, None).unwrap(),
            )
            .unwrap();
            assert_eq!(value["input"]["contents"], expected);
            assert_eq!(value["parameters"]["enable_fusion"], true);
            assert_eq!(value["parameters"]["output_type"], "dense");
            assert!(value["parameters"].get("dimension").is_none());
        }
        assert!(
            build_dashscope_fusion_request(
                "model",
                &[MultimodalPart::Text("text-only".into())],
                None
            )
            .is_ok()
        );
    }
    #[test]
    fn dashscope_input_and_dimension_limits_are_enforced() {
        assert!(build_dashscope_fusion_request("m", &[], None).is_err());
        let too_many: Vec<_> = (0..=MAX_DASHSCOPE_CONTENTS)
            .map(|_| MultimodalPart::Text("x".into()))
            .collect();
        assert!(build_dashscope_fusion_request("m", &too_many[..20], None).is_ok());
        assert!(build_dashscope_fusion_request("m", &too_many, None).is_err());
        let image = png_data_url(1, 1);
        let too_many_images: Vec<_> = (0..=MAX_DASHSCOPE_IMAGES)
            .map(|_| MultimodalPart::ImageDataUrl(image.clone()))
            .collect();
        assert!(build_dashscope_fusion_request("m", &too_many_images[..10], None).is_ok());
        assert!(build_dashscope_fusion_request("m", &too_many_images, None).is_err());
        let video = Url::parse("https://media.example").unwrap();
        assert!(
            build_dashscope_fusion_request(
                "m",
                &[
                    MultimodalPart::VideoUrl(video.clone()),
                    MultimodalPart::Text("between".into()),
                    MultimodalPart::ImageUrl(video.clone()),
                ],
                None,
            )
            .is_ok()
        );
        assert!(
            build_dashscope_fusion_request(
                "m",
                &[
                    MultimodalPart::VideoUrl(video.clone()),
                    MultimodalPart::VideoUrl(video)
                ],
                None,
            )
            .is_err()
        );

        for dimension in [255, 257, 4096] {
            let config = DashScopeConfig::new(
                Url::parse("https://example.com/native/embeddings").unwrap(),
                "m",
                "p",
                NonZeroUsize::new(256).unwrap(),
                SecretString::from("key".to_owned()),
            )
            .unwrap();
            assert!(
                config
                    .with_server_dimensions(NonZeroUsize::new(dimension).unwrap())
                    .is_err()
            );
        }
        let server = DashScopeConfig::new(
            Url::parse("https://example.com/native/embeddings").unwrap(),
            "m",
            "p",
            NonZeroUsize::new(256).unwrap(),
            SecretString::from("key".to_owned()),
        )
        .unwrap()
        .with_server_dimensions(NonZeroUsize::new(512).unwrap())
        .unwrap();
        assert_eq!(server.dimension_mode, DimensionMode::Server);
        assert_eq!(server.expected_dimensions.get(), 512);
        assert_eq!(server.requested_dimensions.unwrap().get(), 512);
    }
    #[tokio::test]
    async fn dashscope_text_count_limits_are_enforced_before_http() {
        let client = Client::new(
            DashScopeConfig::new(
                Url::parse("http://127.0.0.1:9/native").unwrap(),
                "m",
                "p",
                NonZeroUsize::new(256).unwrap(),
                SecretString::from("key".to_owned()),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(matches!(
            client.embed_texts(&[] as &[&str]).await,
            Err(Error::InvalidInput { .. })
        ));
        let too_many = vec!["x".to_owned(); MAX_DASHSCOPE_CONTENTS + 1];
        assert!(matches!(
            client.embed_texts(&too_many).await,
            Err(Error::InvalidInput { .. })
        ));
    }
    #[test]
    fn dashscope_full_endpoint_transport_and_debug_are_safe() {
        let endpoint = Url::parse("https://example.com/exact/native/path").unwrap();
        let secret = "dashscope-secret";
        let config = DashScopeConfig::new(
            endpoint.clone(),
            "model",
            "pipeline",
            NonZeroUsize::new(256).unwrap(),
            SecretString::from(secret.to_owned()),
        )
        .unwrap();
        assert_eq!(config.endpoint, endpoint);
        assert!(!format!("{config:?}").contains(secret));
        assert!(Client::new(config).is_ok());

        for invalid in [
            "https://example.com/",
            "https://user@example.com/native",
            "https://example.com/native?query=x",
            "https://example.com/native#fragment",
            "ftp://example.com/native",
        ] {
            assert!(
                DashScopeConfig::new(
                    Url::parse(invalid).unwrap(),
                    "m",
                    "p",
                    NonZeroUsize::new(256).unwrap(),
                    SecretString::from("key".to_owned())
                )
                .is_err()
            );
        }
        assert!(
            DashScopeConfig::new(
                Url::parse("https://example.com/native").unwrap(),
                "m",
                "p",
                NonZeroUsize::new(256).unwrap(),
                SecretString::from(String::new())
            )
            .is_err()
        );
        let remote_http = || {
            DashScopeConfig::new(
                Url::parse("http://example.com/native").unwrap(),
                "m",
                "p",
                NonZeroUsize::new(256).unwrap(),
                SecretString::from("key".to_owned()),
            )
            .unwrap()
        };
        assert!(Client::new(remote_http()).is_err());
        assert!(Client::new(remote_http().allow_remote_http(true)).is_ok());
    }
    #[test]
    fn dashscope_response_sorts_normalizes_and_maps_usage() {
        let space = EmbeddingSpace {
            provider: Provider::DashScope,
            configured_model: "m".into(),
            pipeline_id: "p".into(),
            profile_revision: PROFILE_REVISION,
            dimension_mode: DimensionMode::Native,
            expected_dimensions: NonZeroUsize::new(2).unwrap(),
        };
        let response = parse_dashscope_response(
            br#"{"output":{"embeddings":[{"index":1,"embedding":[0,2],"type":"vl"},{"index":0,"embedding":[3,4],"type":"vl"}]},"usage":{"input_tokens":7,"image_tokens":11,"total_tokens":18},"request_id":"dash-request"}"#,
            2,
            &space,
            DashScopeOperation::Text,
        )
        .unwrap();
        assert_eq!(response.embeddings(), &[vec![0.6, 0.8], vec![0.0, 1.0]]);
        assert_eq!(response.usage().input_tokens(), Some(7));
        assert_eq!(response.usage().total_tokens(), Some(18));
        assert_eq!(response.diagnostics().request_id(), Some("dash-request"));
        assert_eq!(response.diagnostics().response_id(), None);
        assert!(matches!(
            parse_dashscope_response(
                br#"{"output":{"embeddings":[{"index":0,"embedding":[1],"type":"vl"}]}}"#,
                1,
                &space,
                DashScopeOperation::Text
            ),
            Err(Error::Dimension {
                expected: 2,
                actual: 1
            })
        ));
        for body in [
            br#"{"output":{"embeddings":[{"index":0,"embedding":[3,4]}]}}"#.as_slice(),
            br#"{"output":{"embeddings":[{"index":0,"embedding":[3,4],"type":7}]}}"#
                .as_slice(),
            br#"{"output":{"embeddings":[{"index":0,"embedding":[3,4],"type":"provider-private-type"}]}}"#
                .as_slice(),
        ] {
            let error = parse_dashscope_response(body, 1, &space, DashScopeOperation::Text)
                .expect_err("invalid DashScope type was accepted");
            assert!(matches!(error, Error::InvalidResponse { .. }));
            assert!(!format!("{error:?}").contains("provider-private-type"));
        }
    }
}
