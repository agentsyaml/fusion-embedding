use std::{fmt, num::NonZeroUsize, time::Duration};

use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use url::Url;

use super::{
    DEFAULT_CONNECT_TIMEOUT, DEFAULT_REQUEST_TIMEOUT, EncodingFormat, MAX_MODEL_BYTES,
    MAX_PIPELINE_ID_BYTES,
};
use crate::transport::validate_identifier;
use crate::validation::validate_data_url;
use crate::{ContentPart, Error};

pub(crate) const PROFILE_REVISION: &str = "qwen3-vl-embedding-sglang-jinja-v1";
const EMBEDDINGS_PATH: &str = "/v1/embeddings";

/// SGLang standard `/v1/embeddings` configuration. The server applies Jinja.
#[cfg(feature = "sglang")]
pub struct SglangConfig {
    pub(crate) endpoint: Url,
    pub(crate) configured_model: String,
    pub(crate) pipeline_id: String,
    pub(crate) expected_dimensions: NonZeroUsize,
    pub(crate) api_key: Option<SecretString>,
    pub(crate) connect_timeout: Duration,
    pub(crate) request_timeout: Duration,
    pub(crate) allow_remote_http: bool,
}

#[cfg(feature = "sglang")]
impl SglangConfig {
    pub fn new(
        endpoint: Url,
        model: impl Into<String>,
        pipeline_id: impl Into<String>,
        native_dimensions: NonZeroUsize,
    ) -> Result<Self, Error> {
        let configured_model = model.into();
        let pipeline_id = pipeline_id.into();
        validate_identifier("configured model", &configured_model, MAX_MODEL_BYTES)?;
        validate_identifier("pipeline_id", &pipeline_id, MAX_PIPELINE_ID_BYTES)?;
        Ok(Self {
            endpoint: normalize_endpoint(endpoint)?,
            configured_model,
            pipeline_id,
            expected_dimensions: native_dimensions,
            api_key: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            allow_remote_http: false,
        })
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

    pub fn allow_remote_http(mut self, allow: bool) -> Self {
        self.allow_remote_http = allow;
        self
    }
}

#[cfg(feature = "sglang")]
impl fmt::Debug for SglangConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SglangConfig")
            .field("endpoint", &self.endpoint)
            .field("configured_model", &self.configured_model)
            .field("pipeline_id", &self.pipeline_id)
            .field("expected_dimensions", &self.expected_dimensions)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .field("allow_remote_http", &self.allow_remote_http)
            .finish()
    }
}

#[cfg(feature = "sglang")]
#[derive(Serialize)]
pub(crate) struct SglangRequest<'a> {
    model: &'a str,
    input: Vec<SglangInput>,
    encoding_format: EncodingFormat,
}

#[cfg(feature = "sglang")]
#[derive(Serialize)]
struct SglangInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    video: Option<String>,
}

#[cfg(feature = "sglang")]
pub(crate) fn build_text_request<'a>(model: &'a str, texts: &'a [String]) -> SglangRequest<'a> {
    SglangRequest {
        model,
        input: texts
            .iter()
            .map(|text| SglangInput {
                text: Some(text.clone()),
                image: None,
                video: None,
            })
            .collect(),
        encoding_format: EncodingFormat::Float,
    }
}

#[cfg(feature = "sglang")]
pub(crate) fn build_fusion_request<'a>(
    model: &'a str,
    parts: &'a [ContentPart],
) -> Result<SglangRequest<'a>, Error> {
    let mut image = None;
    let mut video = None;
    let mut text = None;
    for part in parts {
        match part {
            ContentPart::Text(value) => {
                if value.is_empty() {
                    return Err(Error::InvalidInput {
                        message: "SGLang fusion text must not be empty".into(),
                    });
                }
                if text.is_some() {
                    return Err(Error::InvalidInput {
                        message: "SGLang public embedding protocol supports at most one text per input item"
                            .into(),
                    });
                }
                text = Some(value.clone());
            }
            ContentPart::ImageDataUrl(data_url) => {
                if image.is_some() {
                    return Err(Error::InvalidInput {
                        message: "SGLang public embedding protocol supports at most one image per input item"
                            .into(),
                    });
                }
                if video.is_some() || text.is_some() {
                    return Err(Error::InvalidInput {
                        message: "SGLang fusion parts must use canonical image, video, text order"
                            .into(),
                    });
                }
                image = Some(validate_data_url(data_url)?.to_owned());
            }
            ContentPart::VideoSource(source) => {
                if source.trim().is_empty() {
                    return Err(Error::InvalidInput {
                        message: "SGLang video source must not be empty".into(),
                    });
                }
                if video.is_some() {
                    return Err(Error::InvalidInput {
                        message: "SGLang public embedding protocol supports at most one video per input item"
                            .into(),
                    });
                }
                if text.is_some() {
                    return Err(Error::InvalidInput {
                        message: "SGLang fusion parts must use canonical image, video, text order"
                            .into(),
                    });
                }
                video = Some(source.clone());
            }
        }
    }
    if text.is_none() && image.is_none() && video.is_none() {
        return Err(Error::InvalidInput {
            message: "SGLang fusion input must contain a non-empty text, image, or video".into(),
        });
    }
    Ok(SglangRequest {
        model,
        input: vec![SglangInput { text, image, video }],
        encoding_format: EncodingFormat::Float,
    })
}

#[cfg(feature = "sglang")]
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
            message: "SGLang endpoint path must be / or /v1/models".into(),
        });
    }
    endpoint.set_path(EMBEDDINGS_PATH);
    Ok(endpoint)
}
