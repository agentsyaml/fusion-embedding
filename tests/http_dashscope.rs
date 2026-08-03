#![cfg(feature = "dashscope")]

mod common;

use std::num::NonZeroUsize;

use common::*;
use fusion_embedding::{
    BatchEmbeddingResponse, Client, DashScopeConfig, DimensionMode, EmbedOptions, Error,
    MultimodalPart, Provider,
};
use secrecy::SecretString;
use serde_json::json;
use url::Url;

fn dashscope_client(mut endpoint: Url, dimensions: Option<usize>) -> Client {
    endpoint.set_path("/api/native/embeddings");
    let mut config = DashScopeConfig::new(
        endpoint,
        "dashscope-model",
        "dashscope-test-pipeline",
        NonZeroUsize::new(2).unwrap(),
        SecretString::from("dashscope-test-key".to_owned()),
    )
    .unwrap()
    .with_timeouts(IO_TIMEOUT, IO_TIMEOUT)
    .unwrap();
    if let Some(dimensions) = dimensions {
        config = config
            .with_server_dimensions(NonZeroUsize::new(dimensions).unwrap())
            .unwrap();
    }
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
async fn dashscope_native_text_post_contract_and_response_are_observable() {
    let body = br#"{"output":{"embeddings":[{"index":1,"embedding":[0,5],"type":"vl"},{"index":0,"embedding":[3,4],"type":"vl"}]},"usage":{"input_tokens":9,"image_tokens":0,"total_tokens":9},"request_id":"dash-body-request"}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(
            stream,
            "200 OK",
            &[("Content-Type", "application/json")],
            body,
            7,
        )
        .unwrap();
    });

    let response = embed(&dashscope_client(endpoint, None)).await.unwrap();
    let (request, ()) = server.join().unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/api/native/embeddings");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer dashscope-test-key")
    );
    assert_eq!(
        request.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
        json!({
            "model": "dashscope-model",
            "input": {"contents": [{"text": "first"}, {"text": "second"}]},
            "parameters": {"enable_fusion": false, "output_type": "dense"}
        })
    );
    assert_eq!(response.space().provider(), Provider::DashScope);
    assert_eq!(response.space().dimension_mode(), DimensionMode::Native);
    assert_eq!(response.embeddings(), &[vec![0.6, 0.8], vec![0.0, 1.0]]);
    assert_eq!(response.usage().input_tokens(), Some(9));
    assert_eq!(response.usage().total_tokens(), Some(9));
    assert_eq!(
        response.diagnostics().request_id(),
        Some("dash-body-request")
    );
    assert_eq!(response.diagnostics().response_id(), None);
}

#[tokio::test]
async fn dashscope_fusion_post_preserves_independent_ordered_contents() {
    let body = br#"{"output":{"embeddings":[{"index":0,"embedding":[3,4],"type":"fusion"}]},"request_id":"fusion-request"}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(stream, "200 OK", &[], body, 9).unwrap();
    });
    let image = png_data_url();
    let response = dashscope_client(endpoint, None)
        .embed_multimodal_with_options(
            &[
                MultimodalPart::ImageDataUrl(image.clone()),
                MultimodalPart::Text("caption".into()),
                MultimodalPart::ImageDataUrl(image.clone()),
            ],
            EmbedOptions::default(),
        )
        .await
        .unwrap();
    let (request, ()) = server.join().unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/api/native/embeddings");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer dashscope-test-key")
    );
    assert_eq!(
        request.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
        json!({
            "model": "dashscope-model",
            "input": {"contents": [
                {"image": image},
                {"text": "caption"},
                {"image": image}
            ]},
            "parameters": {"enable_fusion": true, "output_type": "dense"}
        })
    );
    assert_eq!(response.embedding(), &[0.6, 0.8]);
}

#[tokio::test]
async fn dashscope_fusion_rejects_text_operation_response_type() {
    let body = br#"{"output":{"embeddings":[{"index":0,"embedding":[3,4],"type":"vl"}]}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(stream, "200 OK", &[], body, 8).unwrap();
    });
    let error = dashscope_client(endpoint, None)
        .embed_multimodal_with_options(
            &[MultimodalPart::ImageDataUrl(png_data_url())],
            EmbedOptions::default(),
        )
        .await
        .unwrap_err();
    let (_request, ()) = server.join().unwrap();
    match error {
        Error::InvalidResponse { message } => assert_eq!(
            message,
            "DashScope embedding type does not match the requested operation"
        ),
        other => panic!("expected invalid response, got {other:?}"),
    }
}

#[tokio::test]
async fn dashscope_top_level_api_error_uses_body_request_id_safely() {
    let body = br#"{"code":"InvalidParameter","message":"private dash message","request_id":"dash-error-request"}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(stream, "400 Bad Request", &[], body, 6).unwrap();
    });

    let error = embed(&dashscope_client(endpoint, None)).await.unwrap_err();
    let (_request, ()) = server.join().unwrap();
    match &error {
        Error::Api {
            status,
            code,
            message,
            request_id,
            ..
        } => {
            assert_eq!(status.as_u16(), 400);
            assert_eq!(code.as_deref(), Some("InvalidParameter"));
            assert_eq!(message, "private dash message");
            assert_eq!(request_id.as_deref(), Some("dash-error-request"));
        }
        other => panic!("expected DashScope API error, got {other:?}"),
    }
    for rendered in [format!("{error:?}"), error.to_string()] {
        assert!(!rendered.contains("private dash message"));
        assert!(!rendered.contains("dash-error-request"));
    }
}
