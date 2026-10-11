#![cfg(feature = "llama-cpp")]

mod common;

use std::num::NonZeroUsize;

use common::*;
use fusion_embedding::{Client, EmbedOptions, Error, LlamaCppConfig, MultimodalPart, Provider};
use serde_json::json;
use url::Url;

fn llama_client(endpoint: Url, dimensions: usize, marker: &str) -> Client {
    let config = LlamaCppConfig::new(
        endpoint,
        "identity-only-model",
        "llama-test-pipeline",
        NonZeroUsize::new(dimensions).unwrap(),
        marker,
    )
    .unwrap()
    .with_timeouts(IO_TIMEOUT, IO_TIMEOUT)
    .unwrap();
    Client::new(config).unwrap()
}

fn png_data_url() -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let mut bytes = vec![137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13];
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&1_u32.to_be_bytes());
    bytes.extend_from_slice(&1_u32.to_be_bytes());
    bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
    bytes.extend_from_slice(&[0; 4]);
    format!("data:image/png;base64,{}", STANDARD.encode(bytes))
}

#[tokio::test]
async fn llama_text_post_contract_sorts_and_normalizes_response() {
    let body = br#"{"id":"llama-text-response","data":[{"index":1,"embedding":[0,8]},{"index":0,"embedding":[6,8]}],"usage":{"prompt_tokens":11,"total_tokens":11}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(
            stream,
            "200 OK",
            &[("Content-Type", "application/json")],
            body,
            9,
        )
        .unwrap();
    });

    let llama = llama_client(endpoint, 2, "<marker>");
    let response = llama
        .embed_texts_with_options(
            &["first".to_owned(), "second".to_owned()],
            EmbedOptions::default(),
        )
        .await
        .unwrap();
    let (request, ()) = server.join().unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/embeddings");
    assert_eq!(
        request.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    let payload = serde_json::from_slice::<serde_json::Value>(&request.body).unwrap();
    assert_eq!(
        payload,
        json!({
            "input": [
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n"
            ],
            "encoding_format": "float",
            "embd_normalize": 2
        })
    );
    assert!(payload.get("model").is_none());
    assert!(payload.get("dimensions").is_none());
    assert_eq!(response.space().provider(), Provider::LlamaCpp);
    assert_eq!(response.embeddings(), &[vec![0.6, 0.8], vec![0.0, 1.0]]);
    assert_eq!(response.usage().input_tokens(), Some(11));
    assert_eq!(response.usage().total_tokens(), Some(11));
}

#[tokio::test]
async fn llama_fusion_post_uses_exact_wrapper_and_single_base64_array() {
    let body = br#"{"data":[{"index":0,"embedding":[3,4]}],"usage":{"prompt_tokens":17,"total_tokens":17}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(
            stream,
            "200 OK",
            &[("Content-Type", "application/json")],
            body,
            11,
        )
        .unwrap();
    });
    let image = png_data_url();
    let bare = image.split_once(',').unwrap().1.to_owned();
    let marker = "<fixed-marker>";
    let response = llama_client(endpoint, 2, marker)
        .embed_multimodal_with_options(
            &[
                MultimodalPart::ImageDataUrl(image),
                MultimodalPart::Text("caption".into()),
            ],
            EmbedOptions::default(),
        )
        .await
        .unwrap();
    let (request, ()) = server.join().unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/embeddings");
    let payload = serde_json::from_slice::<serde_json::Value>(&request.body).unwrap();
    assert_eq!(
        payload,
        json!({
            "input": {
                "prompt_string": "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n<fixed-marker> caption<|im_end|>\n<|im_start|>assistant\n",
                "multimodal_data": [bare]
            },
            "encoding_format": "float",
            "embd_normalize": 2
        })
    );
    assert!(payload.get("model").is_none());
    assert!(payload.get("dimensions").is_none());
    assert_eq!(response.embedding(), &[0.6, 0.8]);
    let norm = response
        .embedding()
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    assert_eq!(response.usage().input_tokens(), Some(17));
    assert_eq!(response.usage().total_tokens(), Some(17));
}

#[tokio::test]
async fn llama_remote_image_url_is_unsupported_before_http() {
    let llama = llama_client(Url::parse("http://127.0.0.1:9/").unwrap(), 2, "<marker>");
    let error = llama
        .embed_multimodal_with_options(
            &[MultimodalPart::ImageUrl(
                Url::parse("https://media.example/image").unwrap(),
            )],
            EmbedOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Unsupported {
            provider: Provider::LlamaCpp,
            ..
        }
    ));
}
