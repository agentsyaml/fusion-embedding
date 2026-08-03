use std::{fmt, num::NonZeroUsize, time::Duration};

/// Per-call options. No request can override model, profile, or dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmbedOptions {
    pub timeout: Option<Duration>,
}

/// An ordered fusion input part.
#[derive(Clone)]
pub enum ContentPart {
    Text(String),
    ImageDataUrl(String),
}

impl fmt::Debug for ContentPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(_) => f.write_str("Text(<redacted>)"),
            Self::ImageDataUrl(_) => f.write_str("ImageDataUrl(<redacted>)"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Provider {
    Vllm,
    LlamaCpp,
    DashScope,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DimensionMode {
    Native,
    Server,
}

/// Opaque identity of vectors that may safely share an index.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct EmbeddingSpace {
    pub(crate) provider: Provider,
    pub(crate) configured_model: String,
    pub(crate) pipeline_id: String,
    pub(crate) profile_revision: &'static str,
    pub(crate) dimension_mode: DimensionMode,
    pub(crate) expected_dimensions: NonZeroUsize,
}

impl EmbeddingSpace {
    pub fn provider(&self) -> Provider {
        self.provider
    }

    pub fn configured_model(&self) -> &str {
        &self.configured_model
    }

    pub fn pipeline_id(&self) -> &str {
        &self.pipeline_id
    }

    pub fn profile_revision(&self) -> &'static str {
        self.profile_revision
    }

    pub fn dimension_mode(&self) -> DimensionMode {
        self.dimension_mode
    }

    pub fn expected_dimensions(&self) -> NonZeroUsize {
        self.expected_dimensions
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    pub(crate) input_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
}

impl Usage {
    pub fn input_tokens(&self) -> Option<u64> {
        self.input_tokens
    }

    pub fn total_tokens(&self) -> Option<u64> {
        self.total_tokens
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct Diagnostics {
    pub(crate) request_id: Option<String>,
    pub(crate) response_id: Option<String>,
}

impl fmt::Debug for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Diagnostics")
            .field("has_request_id", &self.request_id.is_some())
            .field("has_response_id", &self.response_id.is_some())
            .finish()
    }
}

impl Diagnostics {
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn response_id(&self) -> Option<&str> {
        self.response_id.as_deref()
    }
}

/// Provider-neutral response. Embeddings are ordered to match the input.
#[derive(Clone)]
pub struct EmbeddingResponse {
    pub(crate) embeddings: Vec<Vec<f32>>,
    pub(crate) space: EmbeddingSpace,
    pub(crate) usage: Usage,
    pub(crate) diagnostics: Diagnostics,
}

impl EmbeddingResponse {
    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    pub fn into_embeddings(self) -> Vec<Vec<f32>> {
        self.embeddings
    }

    pub fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }
}

impl fmt::Debug for EmbeddingResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dimensions: Vec<_> = self.embeddings.iter().map(Vec::len).collect();
        f.debug_struct("EmbeddingResponse")
            .field("embedding_dimensions", &dimensions)
            .field("space", &self.space)
            .field("usage", &self.usage)
            .field("diagnostics", &self.diagnostics)
            .finish()
    }
}
