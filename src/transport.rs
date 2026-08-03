use std::{
    net::IpAddr,
    time::{Duration, SystemTime},
};

use reqwest::StatusCode;
use serde::Deserialize;
use url::{Host, Url};

use crate::Error;

const MAX_PROVIDER_MESSAGE_CHARS: usize = 1024;
pub(crate) const MAX_DIAGNOSTIC_CHARS: usize = 256;

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn validate_identifier(name: &str, value: &str, max_bytes: usize) -> Result<(), Error> {
    if value.trim().is_empty() {
        return Err(Error::Config {
            message: format!("{name} must not be empty"),
        });
    }
    if value.len() > max_bytes {
        return Err(Error::Config {
            message: format!("{name} exceeds {max_bytes} bytes"),
        });
    }
    if value.chars().any(char::is_control) {
        return Err(Error::Config {
            message: format!("{name} must not contain control characters"),
        });
    }
    Ok(())
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn validate_transport(endpoint: &Url, allow_remote_http: bool) -> Result<(), Error> {
    match endpoint.scheme() {
        "https" => Ok(()),
        "http" if endpoint_is_loopback(endpoint) || allow_remote_http => Ok(()),
        "http" => Err(Error::Config {
            message: "remote cleartext HTTP requires explicit opt-in".into(),
        }),
        _ => Err(Error::Config {
            message: "endpoint scheme must be HTTPS or explicitly allowed HTTP".into(),
        }),
    }
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
fn endpoint_is_loopback(endpoint: &Url) -> bool {
    match endpoint.host() {
        Some(Host::Ipv4(ip)) => IpAddr::V4(ip).is_loopback(),
        Some(Host::Ipv6(ip)) => IpAddr::V6(ip).is_loopback(),
        Some(Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
#[derive(Deserialize)]
struct WireErrorEnvelope {
    #[serde(default)]
    error: Option<WireError>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    code: Option<WireErrorCode>,
    #[serde(default)]
    request_id: Option<String>,
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
#[derive(Deserialize)]
struct WireError {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    code: Option<WireErrorCode>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    request_id: Option<String>,
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
#[derive(Deserialize)]
#[serde(untagged)]
enum WireErrorCode {
    String(String),
    Signed(i64),
    Unsigned(u64),
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
impl WireErrorCode {
    fn as_string(&self) -> String {
        match self {
            Self::String(value) => value.clone(),
            Self::Signed(value) => value.to_string(),
            Self::Unsigned(value) => value.to_string(),
        }
    }
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) async fn read_limited(
    response: &mut reqwest::Response,
    limit: usize,
    truncate: bool,
) -> Result<(Vec<u8>, bool), Error> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|source| Error::Transport { source })?
    {
        let remaining = limit.saturating_sub(body.len());
        if chunk.len() > remaining {
            if !truncate {
                return Err(Error::InvalidResponse {
                    message: format!("response body exceeds {limit} bytes"),
                });
            }
            body.extend_from_slice(&chunk[..remaining]);
            return Ok((body, true));
        }
        body.extend_from_slice(&chunk);
    }
    Ok((body, false))
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn api_error(
    status: StatusCode,
    body: &[u8],
    truncated: bool,
    request_id: Option<String>,
    retry_after: Option<Duration>,
) -> Error {
    let parsed: Option<WireErrorEnvelope> = (!truncated)
        .then(|| serde_json::from_slice(body).ok())
        .flatten();
    let body_request_id = parsed.as_ref().and_then(|envelope| {
        envelope
            .request_id
            .as_deref()
            .or_else(|| {
                envelope
                    .error
                    .as_ref()
                    .and_then(|error| error.request_id.as_deref())
            })
            .and_then(|value| sanitize_metadata(value, MAX_DIAGNOSTIC_CHARS))
    });
    let (message, code) = if truncated {
        ("provider returned an oversized error body".into(), None)
    } else if let Some(envelope) = parsed {
        let message = envelope
            .error
            .as_ref()
            .and_then(|error| error.message.as_deref())
            .or(envelope.message.as_deref())
            .and_then(sanitize_provider_message)
            .unwrap_or_else(|| "provider returned an error without a safe message".into());
        let code = error_code(&envelope);
        (message, code)
    } else {
        ("provider returned a non-JSON error body".into(), None)
    };
    Error::Api {
        status,
        code,
        message,
        request_id: request_id.or(body_request_id),
        retry_after,
        transient: matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504),
    }
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
fn error_code(envelope: &WireErrorEnvelope) -> Option<String> {
    let value = if let Some(code) = envelope
        .error
        .as_ref()
        .and_then(|error| error.code.as_ref())
    {
        code.as_string()
    } else if let Some(kind) = envelope
        .error
        .as_ref()
        .and_then(|error| error.kind.as_ref())
    {
        kind.clone()
    } else {
        envelope.code.as_ref()?.as_string()
    };
    sanitize_metadata(&value, 128)
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
fn sanitize_provider_message(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    if lower.contains("data:image/") || lower.contains("<html") || lower.contains("<!doctype") {
        return Some("provider returned a redacted error message".into());
    }
    sanitize_metadata(value, MAX_PROVIDER_MESSAGE_CHARS)
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn sanitize_metadata(value: &str, max_chars: usize) -> Option<String> {
    let mut output = String::new();
    for character in value.chars().filter(|character| !character.is_control()) {
        if output.chars().count() == max_chars {
            output.push('…');
            break;
        }
        output.push(character);
    }
    (!output.is_empty()).then_some(output)
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn parse_retry_after(value: &str) -> Option<Duration> {
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let retry_at = httpdate::parse_http_date(value).ok()?;
    Some(
        retry_at
            .duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_sanitized_and_retry_after_parses() {
        let error = api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            br#"{"error":{"message":"bad\u0000 data:image/png;base64,SECRET","type":"busy"}}"#,
            false,
            Some("req-1".into()),
            Some(Duration::from_secs(2)),
        );
        let debug = format!("{error:?}");
        assert!(!debug.contains("SECRET"));
        assert!(matches!(
            error,
            Error::Api {
                transient: true,
                ..
            }
        ));
        assert_eq!(parse_retry_after("42"), Some(Duration::from_secs(42)));
        assert!(parse_retry_after("not-a-date").is_none());
    }
    #[test]
    fn error_codes_accept_integer_string_type_and_top_level_shapes() {
        for (body, expected) in [
            (
                br#"{"error":{"message":"bad","code":123}}"#.as_slice(),
                "123",
            ),
            (
                br#"{"error":{"message":"bad","code":"nested"}}"#.as_slice(),
                "nested",
            ),
            (
                br#"{"error":{"message":"bad","type":"fallback"}}"#.as_slice(),
                "fallback",
            ),
            (br#"{"message":"bad","code":456}"#.as_slice(), "456"),
        ] {
            let error = api_error(StatusCode::BAD_REQUEST, body, false, None, None);
            match error {
                Error::Api { code, .. } => assert_eq!(code.as_deref(), Some(expected)),
                _ => panic!("expected API error"),
            }
        }
    }
    #[test]
    fn truncated_error_never_parses_json_prefix() {
        let secret = "truncated-secret";
        let error = api_error(
            StatusCode::BAD_GATEWAY,
            format!(r#"{{"error":{{"message":"{secret}","code":"secret-code"}}}}"#).as_bytes(),
            true,
            None,
            None,
        );
        match &error {
            Error::Api { code, message, .. } => {
                assert_eq!(code, &None);
                assert_eq!(message, "provider returned an oversized error body");
            }
            _ => panic!("expected API error"),
        }
        assert!(!format!("{error:?}").contains(secret));
    }
    #[test]
    fn api_error_debug_and_display_hide_plain_reflected_values() {
        let error = api_error(
            StatusCode::BAD_REQUEST,
            br#"{"error":{"message":"plain-reflected-secret","code":"secret-code"}}"#,
            false,
            Some("secret-request-id".into()),
            None,
        );
        match &error {
            Error::Api {
                message,
                code,
                request_id,
                ..
            } => {
                assert_eq!(message, "plain-reflected-secret");
                assert_eq!(code.as_deref(), Some("secret-code"));
                assert_eq!(request_id.as_deref(), Some("secret-request-id"));
            }
            _ => panic!("expected API error"),
        }
        for rendered in [format!("{error:?}"), error.to_string()] {
            assert!(!rendered.contains("plain-reflected-secret"));
            assert!(!rendered.contains("secret-code"));
            assert!(!rendered.contains("secret-request-id"));
        }
    }
}
