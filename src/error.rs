use std::{fmt, time::Duration};

use reqwest::StatusCode;
use thiserror::Error as ThisError;

use crate::Provider;

#[derive(ThisError)]
pub enum Error {
    #[error("invalid configuration: {message}")]
    Config { message: String },
    #[error("invalid input: {message}")]
    InvalidInput { message: String },
    #[error("{provider:?} does not support {feature}")]
    Unsupported {
        provider: Provider,
        feature: &'static str,
    },
    #[error("HTTP transport failed: {source}")]
    Transport {
        #[source]
        source: reqwest::Error,
    },
    #[error("provider API returned {status}")]
    Api {
        status: StatusCode,
        code: Option<String>,
        message: String,
        request_id: Option<String>,
        retry_after: Option<Duration>,
        transient: bool,
    },
    #[error("invalid provider response: {message}")]
    InvalidResponse { message: String },
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    Dimension { expected: usize, actual: usize },
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config { message } => f.debug_struct("Config").field("message", message).finish(),
            Self::InvalidInput { message } => f
                .debug_struct("InvalidInput")
                .field("message", message)
                .finish(),
            Self::Unsupported { provider, feature } => f
                .debug_struct("Unsupported")
                .field("provider", provider)
                .field("feature", feature)
                .finish(),
            Self::Transport { .. } => f.debug_struct("Transport").finish_non_exhaustive(),
            Self::Api {
                status,
                code,
                message,
                request_id,
                retry_after,
                transient,
            } => f
                .debug_struct("Api")
                .field("status", status)
                .field("has_code", &code.is_some())
                .field("has_message", &!message.is_empty())
                .field("has_request_id", &request_id.is_some())
                .field("retry_after", retry_after)
                .field("transient", transient)
                .finish(),
            Self::InvalidResponse { message } => f
                .debug_struct("InvalidResponse")
                .field("message", message)
                .finish(),
            Self::Dimension { expected, actual } => f
                .debug_struct("Dimension")
                .field("expected", expected)
                .field("actual", actual)
                .finish(),
        }
    }
}
