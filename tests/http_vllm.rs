#![cfg(feature = "vllm")]

mod common;

use std::{
    io::{self, Write},
    num::NonZeroUsize,
    time::Duration,
};

use common::*;
use fusion_embedding::{
    BatchEmbeddingResponse, Client, DimensionMode, EmbedOptions, Error, VllmConfig,
};
use secrecy::SecretString;
use serde_json::json;
use url::Url;

fn client(endpoint: Url, dimensions: usize) -> Client {
    let config = VllmConfig::new(
        endpoint,
        "configured-model",
        "test-pipeline",
        NonZeroUsize::new(dimensions).unwrap(),
    )
    .unwrap()
    .with_api_key(SecretString::from("local-test-key".to_owned()))
    .unwrap()
    .with_timeouts(IO_TIMEOUT, IO_TIMEOUT)
    .unwrap();
    Client::new(config).unwrap()
}

fn server_dimensions_client(endpoint: Url, dimensions: usize) -> Client {
    let config = VllmConfig::new(
        endpoint,
        "configured-model",
        "test-pipeline",
        NonZeroUsize::new(2048).unwrap(),
    )
    .unwrap()
    .with_server_dimensions(NonZeroUsize::new(dimensions).unwrap())
    .with_api_key(SecretString::from("local-test-key".to_owned()))
    .unwrap()
    .with_timeouts(IO_TIMEOUT, IO_TIMEOUT)
    .unwrap();
    Client::new(config).unwrap()
}

async fn embed(client: &Client) -> Result<BatchEmbeddingResponse, Error> {
    client
        .embed_texts_with_options(
            &["first".to_owned(), "second".to_owned()],
            EmbedOptions::default(),
        )
        .await
}

#[tokio::test]
async fn post_contract_and_chunked_success_are_observable() {
    let body = br#"{"id":"response-id","data":[{"index":1,"embedding":[0,5]},{"index":0,"embedding":[3,4]}],"usage":{"prompt_tokens":7,"total_tokens":9}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(
            stream,
            "200 OK",
            &[
                ("Content-Type", "application/json"),
                ("x-request-id", "request-id"),
            ],
            body,
            7,
        )
        .unwrap();
    });

    let response = embed(&client(endpoint, 2)).await.unwrap();
    let (request, ()) = server.join().unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/embeddings");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer local-test-key")
    );
    assert_eq!(
        request.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
        json!({
            "model": "configured-model",
            "input": [
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n"
            ],
            "encoding_format": "float",
            "normalize": true
        })
    );
    assert_eq!(response.embeddings(), &[vec![0.6, 0.8], vec![0.0, 1.0]]);
    for vector in response.embeddings() {
        let norm = vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
    assert_eq!(response.usage().input_tokens(), Some(7));
    assert_eq!(response.usage().total_tokens(), Some(9));
    assert_eq!(response.diagnostics().request_id(), Some("request-id"));
    assert_eq!(response.diagnostics().response_id(), Some("response-id"));
    let debug = format!("{response:?}");
    assert!(!debug.contains("request-id"));
    assert!(!debug.contains("response-id"));
}

#[tokio::test]
async fn vllm_server_dimensions_are_sent_and_recorded_in_space() {
    let body = br#"{"data":[{"index":1,"embedding":[0,5]},{"index":0,"embedding":[3,4]}],"usage":{"prompt_tokens":7,"total_tokens":9}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(stream, "200 OK", &[], body, 7).unwrap();
    });

    let response = embed(&server_dimensions_client(endpoint, 2)).await.unwrap();
    let (request, ()) = server.join().unwrap();
    assert_eq!(request.path, "/v1/embeddings");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
        json!({
            "model": "configured-model",
            "input": [
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n"
            ],
            "encoding_format": "float",
            "normalize": true,
            "dimensions": 2
        })
    );
    assert_eq!(response.space().dimension_mode(), DimensionMode::Server);
    assert_eq!(response.space().expected_dimensions().get(), 2);
    assert_eq!(response.embeddings(), &[vec![0.6, 0.8], vec![0.0, 1.0]]);
}

#[tokio::test]
async fn vllm_server_dimension_mismatch_does_not_fallback() {
    let body = br#"{"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[1]}]}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(stream, "200 OK", &[], body, 5).unwrap();
    });

    let error = embed(&server_dimensions_client(endpoint, 2))
        .await
        .unwrap_err();
    let (request, ()) = server.join().unwrap();
    let payload = serde_json::from_slice::<serde_json::Value>(&request.body).unwrap();
    assert_eq!(payload["dimensions"], 2);
    assert!(matches!(
        error,
        Error::Dimension {
            expected: 2,
            actual: 1
        }
    ));
}

#[tokio::test]
async fn success_body_over_eight_mib_is_rejected_over_http() {
    const CHUNK_SIZE: usize = 64 * 1024;
    const CHUNKS: usize = 8 * 1024 * 1024 / CHUNK_SIZE;
    let (endpoint, server) = spawn_server(|stream| {
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
        let chunk = [b' '; CHUNK_SIZE];
        let mut sent = 0;
        for _ in 0..CHUNKS {
            if write!(stream, "{CHUNK_SIZE:x}\r\n")
                .and_then(|()| stream.write_all(&chunk))
                .and_then(|()| stream.write_all(b"\r\n"))
                .is_err()
            {
                break;
            }
            sent += CHUNK_SIZE;
        }
        if sent == 8 * 1024 * 1024 && stream.write_all(b"1\r\n").is_ok() {
            if stream.write_all(b" ").is_ok() {
                sent += 1;
            }
            stream.write_all(b"\r\n0\r\n\r\n").ok();
        }
        sent
    });

    let result = embed(&client(endpoint, 2)).await;
    let (_request, sent) = server.join().unwrap();
    assert!(sent > 8 * 1024 * 1024, "server sent {sent} body bytes");
    match result {
        Err(Error::InvalidResponse { message }) => {
            assert_eq!(message, "response body exceeds 8388608 bytes");
        }
        other => panic!("expected oversized invalid response, got {other:?}"),
    }
}

#[tokio::test]
async fn oversized_error_body_is_truncated_without_parsing_its_json_prefix() {
    let prefix = br#"{"error":{"message":"must-not-parse","code":"must-not-parse-code"}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        stream
            .write_all(b"HTTP/1.1 400 Bad Request\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut body = vec![b' '; 64 * 1024 + 1];
        body[..prefix.len()].copy_from_slice(prefix);
        let mut sent = 0;
        for chunk in body.chunks(1024) {
            if let Err(error) = write!(stream, "{:x}\r\n", chunk.len()) {
                panic!("error response chunk header failed after {sent} body bytes: {error}");
            }
            let mut remaining = chunk;
            while !remaining.is_empty() {
                match stream.write(remaining) {
                    Ok(0) => panic!("error response write returned zero after {sent} body bytes"),
                    Ok(written) => {
                        sent += written;
                        remaining = &remaining[written..];
                    }
                    Err(error) => {
                        panic!("error response body failed after {sent} body bytes: {error}")
                    }
                }
            }
            if let Err(error) = stream.write_all(b"\r\n") {
                assert!(
                    sent > 64 * 1024
                        && matches!(
                            error.kind(),
                            io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
                        ),
                    "error response chunk trailer failed after {sent} body bytes: {error}"
                );
                return sent;
            }
        }
        if let Err(error) = stream.write_all(b"0\r\n\r\n") {
            assert!(
                sent > 64 * 1024
                    && matches!(
                        error.kind(),
                        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
                    ),
                "error response terminator failed after {sent} body bytes: {error}"
            );
        }
        sent
    });

    let error = embed(&client(endpoint, 2)).await.unwrap_err();
    let (_request, sent) = server.join().unwrap();
    assert!(sent > 64 * 1024, "server sent {sent} error body bytes");
    match &error {
        Error::Api {
            status,
            code,
            message,
            ..
        } => {
            assert_eq!(status.as_u16(), 400);
            assert_eq!(code, &None);
            assert_eq!(message, "provider returned an oversized error body");
        }
        other => panic!("expected API error, got {other:?}"),
    }
    for rendered in [format!("{error:?}"), error.to_string()] {
        assert!(!rendered.contains("must-not-parse"));
        assert!(!rendered.contains("must-not-parse-code"));
    }
}

#[tokio::test]
async fn api_error_shape_and_real_headers_are_observable_but_rendered_redacted() {
    let body = br#"{"error":{"message":"provider-private-message","code":123}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(
            stream,
            "429 Too Many Requests",
            &[("Retry-After", "7"), ("x-request-id", "private-request-id")],
            body,
            5,
        )
        .unwrap();
    });

    let error = embed(&client(endpoint, 2)).await.unwrap_err();
    let (_request, ()) = server.join().unwrap();
    match &error {
        Error::Api {
            status,
            code,
            message,
            request_id,
            retry_after,
            transient,
        } => {
            assert_eq!(status.as_u16(), 429);
            assert_eq!(code.as_deref(), Some("123"));
            assert_eq!(message, "provider-private-message");
            assert_eq!(request_id.as_deref(), Some("private-request-id"));
            assert_eq!(*retry_after, Some(Duration::from_secs(7)));
            assert!(transient);
        }
        other => panic!("expected API error, got {other:?}"),
    }
    for rendered in [format!("{error:?}"), error.to_string()] {
        assert!(!rendered.contains("provider-private-message"));
        assert!(!rendered.contains("private-request-id"));
        assert!(!rendered.contains("123"));
    }
}
