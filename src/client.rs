#[cfg(any(feature = "vllm", feature = "dashscope"))]
use std::num::NonZeroUsize;
use std::{fmt, time::Duration};

#[cfg(any(feature = "vllm", feature = "dashscope", feature = "sglang"))]
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest::{
    header::{CONTENT_TYPE, RETRY_AFTER},
    redirect::Policy,
};
#[cfg(any(feature = "vllm", feature = "dashscope", feature = "sglang"))]
use secrecy::ExposeSecret;
#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "sglang"))]
use serde::Deserialize;
use serde::Serialize;
use url::Url;

#[cfg(any(feature = "llama-cpp", feature = "sglang"))]
use crate::DimensionMode;
#[cfg(any(feature = "vllm", feature = "llama-cpp"))]
use crate::backend::EncodingFormat;
#[cfg(feature = "dashscope")]
use crate::backend::dashscope::{
    self, DashScopeConfig, DashScopeOperation, MAX_DASHSCOPE_CONTENTS,
    build_dashscope_fusion_request, build_dashscope_text_request, parse_dashscope_response,
};
#[cfg(any(feature = "vllm", feature = "llama-cpp"))]
use crate::backend::format_qwen_text_inputs;
#[cfg(feature = "llama-cpp")]
use crate::backend::llama_cpp::{
    self, LlamaCppConfig, LlamaTextRequest, build_llama_fusion_request,
};
#[cfg(feature = "sglang")]
use crate::backend::sglang::{
    self, SglangConfig, build_fusion_request as build_sglang_fusion_request,
    build_text_request as build_sglang_text_request,
};
#[cfg(feature = "vllm")]
use crate::backend::vllm::{self, TextRequest, VllmConfig, build_fusion_request};
use crate::transport::{
    MAX_DIAGNOSTIC_CHARS, api_error, parse_retry_after, read_limited, sanitize_metadata,
    validate_transport,
};
#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "sglang"))]
use crate::validation::{IndexedEmbedding, validate_embeddings};
use crate::{ContentPart, EmbedOptions, EmbeddingResponse, EmbeddingSpace, Error, Provider};
#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "sglang"))]
use crate::{Diagnostics, Usage};

const MAX_RESPONSE_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 64 * 1024;

/// Reusable embedding client. Its backend, endpoint, profile, and vector space are fixed.
#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    endpoint: Url,
    backend: Backend,
    pub(crate) space: EmbeddingSpace,
    request_timeout: Duration,
}

#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
#[derive(Clone)]
enum Backend {
    #[cfg(feature = "vllm")]
    Vllm {
        requested_dimensions: Option<NonZeroUsize>,
    },
    #[cfg(feature = "llama-cpp")]
    LlamaCpp { media_marker: String },
    #[cfg(feature = "dashscope")]
    DashScope {
        requested_dimensions: Option<NonZeroUsize>,
    },
    #[cfg(feature = "sglang")]
    Sglang,
}

#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
#[derive(Clone, Copy)]
enum ResponseKind {
    #[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "sglang"))]
    Oai,
    #[cfg(feature = "dashscope")]
    DashScope(DashScopeOperation),
}

#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
impl Backend {
    fn provider(&self) -> Provider {
        match self {
            #[cfg(feature = "vllm")]
            Self::Vllm { .. } => Provider::Vllm,
            #[cfg(feature = "llama-cpp")]
            Self::LlamaCpp { .. } => Provider::LlamaCpp,
            #[cfg(feature = "dashscope")]
            Self::DashScope { .. } => Provider::DashScope,
            #[cfg(feature = "sglang")]
            Self::Sglang => Provider::Sglang,
        }
    }
}

#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
impl Client {
    #[cfg(feature = "vllm")]
    pub fn vllm(config: VllmConfig) -> Result<Self, Error> {
        validate_transport(&config.endpoint, config.allow_remote_http)?;

        let mut headers = HeaderMap::new();
        if let Some(api_key) = config.api_key {
            let mut value = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(|_| Error::Config {
                message: "API key is not valid in an Authorization header".into(),
            })?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }

        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .redirect(Policy::none())
            .default_headers(headers)
            .build()
            .map_err(|source| Error::Transport { source })?;

        Ok(Self {
            http,
            endpoint: config.endpoint,
            backend: Backend::Vllm {
                requested_dimensions: config.requested_dimensions,
            },
            space: EmbeddingSpace {
                provider: Provider::Vllm,
                configured_model: config.configured_model,
                pipeline_id: config.pipeline_id,
                profile_revision: vllm::PROFILE_REVISION,
                dimension_mode: config.dimension_mode,
                expected_dimensions: config.expected_dimensions,
            },
            request_timeout: config.request_timeout,
        })
    }

    #[cfg(feature = "llama-cpp")]
    pub fn llama_cpp(config: LlamaCppConfig) -> Result<Self, Error> {
        validate_transport(&config.endpoint, config.allow_remote_http)?;
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .redirect(Policy::none())
            .build()
            .map_err(|source| Error::Transport { source })?;

        Ok(Self {
            http,
            endpoint: config.endpoint,
            backend: Backend::LlamaCpp {
                media_marker: config.media_marker,
            },
            space: EmbeddingSpace {
                provider: Provider::LlamaCpp,
                configured_model: config.configured_model,
                pipeline_id: config.pipeline_id,
                profile_revision: llama_cpp::PROFILE_REVISION,
                dimension_mode: DimensionMode::Native,
                expected_dimensions: config.expected_dimensions,
            },
            request_timeout: config.request_timeout,
        })
    }

    #[cfg(feature = "dashscope")]
    pub fn dashscope(config: DashScopeConfig) -> Result<Self, Error> {
        validate_transport(&config.endpoint, config.allow_remote_http)?;
        let mut value =
            HeaderValue::from_str(&format!("Bearer {}", config.api_key.expose_secret())).map_err(
                |_| Error::Config {
                    message: "API key is not valid in an Authorization header".into(),
                },
            )?;
        value.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, value);
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .redirect(Policy::none())
            .default_headers(headers)
            .build()
            .map_err(|source| Error::Transport { source })?;

        Ok(Self {
            http,
            endpoint: config.endpoint,
            backend: Backend::DashScope {
                requested_dimensions: config.requested_dimensions,
            },
            space: EmbeddingSpace {
                provider: Provider::DashScope,
                configured_model: config.configured_model,
                pipeline_id: config.pipeline_id,
                profile_revision: dashscope::PROFILE_REVISION,
                dimension_mode: config.dimension_mode,
                expected_dimensions: config.expected_dimensions,
            },
            request_timeout: config.request_timeout,
        })
    }

    #[cfg(feature = "sglang")]
    pub fn sglang(config: SglangConfig) -> Result<Self, Error> {
        validate_transport(&config.endpoint, config.allow_remote_http)?;
        let mut headers = HeaderMap::new();
        if let Some(api_key) = config.api_key {
            let mut value = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(|_| Error::Config {
                message: "API key is not valid in an Authorization header".into(),
            })?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .redirect(Policy::none())
            .default_headers(headers)
            .build()
            .map_err(|source| Error::Transport { source })?;
        Ok(Self {
            http,
            endpoint: config.endpoint,
            backend: Backend::Sglang,
            space: EmbeddingSpace {
                provider: Provider::Sglang,
                configured_model: config.configured_model,
                pipeline_id: config.pipeline_id,
                profile_revision: sglang::PROFILE_REVISION,
                dimension_mode: DimensionMode::Native,
                expected_dimensions: config.expected_dimensions,
            },
            request_timeout: config.request_timeout,
        })
    }

    pub async fn embed_texts(
        &self,
        texts: &[String],
        options: &EmbedOptions,
    ) -> Result<EmbeddingResponse, Error> {
        if texts.is_empty() {
            return Err(Error::InvalidInput {
                message: "text input must not be empty".into(),
            });
        }
        match &self.backend {
            #[cfg(feature = "vllm")]
            Backend::Vllm {
                requested_dimensions,
            } => {
                let input = format_qwen_text_inputs(texts);
                let payload = TextRequest {
                    model: &self.space.configured_model,
                    input: &input,
                    encoding_format: EncodingFormat::Float,
                    normalize: true,
                    dimensions: requested_dimensions.map(NonZeroUsize::get),
                };
                self.execute(&payload, texts.len(), options, ResponseKind::Oai)
                    .await
            }
            #[cfg(feature = "llama-cpp")]
            Backend::LlamaCpp { .. } => {
                let input = format_qwen_text_inputs(texts);
                let payload = LlamaTextRequest {
                    input: &input,
                    encoding_format: EncodingFormat::Float,
                    embd_normalize: 2,
                };
                self.execute(&payload, texts.len(), options, ResponseKind::Oai)
                    .await
            }
            #[cfg(feature = "dashscope")]
            Backend::DashScope {
                requested_dimensions,
            } => {
                if texts.len() > MAX_DASHSCOPE_CONTENTS {
                    return Err(Error::InvalidInput {
                        message: format!(
                            "DashScope text input must contain at most {MAX_DASHSCOPE_CONTENTS} items"
                        ),
                    });
                }
                let payload = build_dashscope_text_request(
                    &self.space.configured_model,
                    texts,
                    *requested_dimensions,
                );
                self.execute(
                    &payload,
                    texts.len(),
                    options,
                    ResponseKind::DashScope(DashScopeOperation::Text),
                )
                .await
            }
            #[cfg(feature = "sglang")]
            Backend::Sglang => {
                let payload = build_sglang_text_request(&self.space.configured_model, texts);
                self.execute(&payload, texts.len(), options, ResponseKind::Oai)
                    .await
            }
        }
    }

    pub async fn embed_fusion(
        &self,
        parts: &[ContentPart],
        options: &EmbedOptions,
    ) -> Result<EmbeddingResponse, Error> {
        match &self.backend {
            #[cfg(feature = "vllm")]
            Backend::Vllm {
                requested_dimensions,
            } => {
                let payload = build_fusion_request(
                    &self.space.configured_model,
                    parts,
                    *requested_dimensions,
                )?;
                self.execute(&payload, 1, options, ResponseKind::Oai).await
            }
            #[cfg(feature = "llama-cpp")]
            Backend::LlamaCpp { media_marker } => {
                let payload = build_llama_fusion_request(media_marker, parts)?;
                self.execute(&payload, 1, options, ResponseKind::Oai).await
            }
            #[cfg(feature = "dashscope")]
            Backend::DashScope {
                requested_dimensions,
            } => {
                let payload = build_dashscope_fusion_request(
                    &self.space.configured_model,
                    parts,
                    *requested_dimensions,
                )?;
                self.execute(
                    &payload,
                    1,
                    options,
                    ResponseKind::DashScope(DashScopeOperation::Fusion),
                )
                .await
            }
            #[cfg(feature = "sglang")]
            Backend::Sglang => {
                let payload = build_sglang_fusion_request(&self.space.configured_model, parts)?;
                self.execute(&payload, 1, options, ResponseKind::Oai).await
            }
        }
    }

    async fn execute<T: Serialize + ?Sized>(
        &self,
        payload: &T,
        expected_count: usize,
        options: &EmbedOptions,
        response_kind: ResponseKind,
    ) -> Result<EmbeddingResponse, Error> {
        let body = serde_json::to_vec(payload).map_err(|_| Error::InvalidInput {
            message: "request cannot be serialized".into(),
        })?;
        let timeout = options.timeout.unwrap_or(self.request_timeout);
        if timeout.is_zero() {
            return Err(Error::InvalidInput {
                message: "request timeout must be greater than zero".into(),
            });
        }

        let mut response = self
            .http
            .post(self.endpoint.clone())
            .header(CONTENT_TYPE, "application/json")
            .timeout(timeout)
            .body(body)
            .send()
            .await
            .map_err(|source| Error::Transport { source })?;
        let status = response.status();
        let request_id = response
            .headers()
            .get("x-request-id")
            .or_else(|| response.headers().get("request-id"))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| sanitize_metadata(value, MAX_DIAGNOSTIC_CHARS));
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_retry_after);

        if !status.is_success() {
            let (body, truncated) = read_limited(&mut response, MAX_ERROR_BYTES, true).await?;
            return Err(api_error(status, &body, truncated, request_id, retry_after));
        }

        let (body, _) = read_limited(&mut response, MAX_RESPONSE_JSON_BYTES, false).await?;
        match response_kind {
            #[cfg(feature = "dashscope")]
            ResponseKind::DashScope(operation) => {
                parse_dashscope_response(&body, expected_count, &self.space, operation)
            }
            #[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "sglang"))]
            ResponseKind::Oai => {
                let wire: WireResponse =
                    serde_json::from_slice(&body).map_err(|source| Error::InvalidResponse {
                        message: format!("response is not valid JSON: {source}"),
                    })?;
                let embeddings = validate_embeddings(
                    wire.data,
                    expected_count,
                    self.space.expected_dimensions.get(),
                )?;
                Ok(EmbeddingResponse {
                    embeddings,
                    space: self.space.clone(),
                    usage: Usage {
                        input_tokens: wire.usage.as_ref().and_then(|usage| usage.prompt_tokens),
                        total_tokens: wire.usage.and_then(|usage| usage.total_tokens),
                    },
                    diagnostics: Diagnostics {
                        request_id,
                        response_id: wire
                            .id
                            .as_deref()
                            .and_then(|value| sanitize_metadata(value, MAX_DIAGNOSTIC_CHARS)),
                    },
                })
            }
        }
    }
}

#[cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]
impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("provider", &self.backend.provider())
            .field("endpoint", &self.endpoint)
            .field("space", &self.space)
            .field("request_timeout", &self.request_timeout)
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "sglang"))]
#[derive(Deserialize)]
struct WireResponse {
    data: Vec<IndexedEmbedding>,
    #[serde(default)]
    usage: Option<WireUsage>,
    #[serde(default)]
    id: Option<String>,
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "sglang"))]
#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[cfg(all(test, feature = "vllm", feature = "llama-cpp"))]
mod tests {
    use super::*;
    use crate::ContentPart;
    use crate::backend::{llama_cpp::LlamaCppConfig, vllm::VllmConfig};
    use secrecy::SecretString;

    #[test]
    fn debug_redacts_secrets_media_and_vectors() {
        let secret = "top-secret";
        let config = VllmConfig::new(
            Url::parse("https://example.com").unwrap(),
            "m",
            "p",
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap()
        .with_api_key(SecretString::from(secret.to_owned()))
        .unwrap()
        .with_server_dimensions(NonZeroUsize::new(2).unwrap());
        assert!(!format!("{config:?}").contains(secret));
        let client = Client::vllm(config).unwrap();
        assert!(!format!("{client:?}").contains(secret));
        let image = ContentPart::ImageDataUrl("data:image/png;base64,SECRET".into());
        assert!(!format!("{image:?}").contains("SECRET"));
        let text = ContentPart::Text("ordinary-text-secret".into());
        assert!(!format!("{text:?}").contains("ordinary-text-secret"));
        let video = ContentPart::VideoSource("video-source-secret".into());
        assert!(!format!("{video:?}").contains("video-source-secret"));

        let diagnostics = Diagnostics {
            request_id: Some("request-secret".into()),
            response_id: Some("response-secret".into()),
        };
        let diagnostics_debug = format!("{diagnostics:?}");
        assert!(!diagnostics_debug.contains("request-secret"));
        assert!(!diagnostics_debug.contains("response-secret"));

        let response = EmbeddingResponse {
            embeddings: vec![vec![0.123456, 0.987654]],
            space: EmbeddingSpace {
                provider: Provider::Vllm,
                configured_model: "m".into(),
                pipeline_id: "p".into(),
                profile_revision: vllm::PROFILE_REVISION,
                dimension_mode: DimensionMode::Native,
                expected_dimensions: NonZeroUsize::new(2).unwrap(),
            },
            usage: Usage::default(),
            diagnostics: Diagnostics::default(),
        };
        let debug = format!("{response:?}");
        assert!(!debug.contains("0.123456"));

        let marker = "private-media-marker";
        let llama_config = LlamaCppConfig::new(
            Url::parse("http://localhost:8080/v1/models").unwrap(),
            "identity-model",
            "pipeline",
            NonZeroUsize::new(2048).unwrap(),
            marker,
        )
        .unwrap();
        assert!(!format!("{llama_config:?}").contains(marker));
        let llama = Client::llama_cpp(llama_config).unwrap();
        let client_debug = format!("{llama:?}");
        assert!(client_debug.contains("LlamaCpp"));
        assert!(!client_debug.contains(marker));
    }
}
