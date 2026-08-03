#![cfg(feature = "sglang")]

mod common;

use std::num::NonZeroUsize;

use common::*;
use fusion_embedding::{Client, ContentPart, EmbedOptions, Error, Provider, SglangConfig};
use secrecy::SecretString;
use serde_json::json;
use url::Url;

fn client(endpoint: Url, api_key: bool) -> Client {
    let config = SglangConfig::new(
        endpoint,
        "sglang-model",
        "sglang-pipeline",
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap()
    .with_timeouts(IO_TIMEOUT, IO_TIMEOUT)
    .unwrap();
    let config = if api_key {
        config
            .with_api_key(SecretString::from("test-key".to_owned()))
            .unwrap()
    } else {
        config
    };
    Client::sglang(config).unwrap()
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
async fn structured_text_auth_and_oai_response_contract() {
    let body = br#"{"id":"response-id","data":[{"index":1,"embedding":[0,5]},{"index":0,"embedding":[3,4]}],"usage":{"prompt_tokens":7,"total_tokens":9}}"#;
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(stream, "200 OK", &[("x-request-id", "request-id")], body, 7).unwrap()
    });
    let response = client(endpoint, true)
        .embed_texts(&["first".into(), "second".into()], &EmbedOptions::default())
        .await
        .unwrap();
    let (request, ()) = server.join().unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/embeddings");
    assert_eq!(
        request.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer test-key")
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
        json!({"model":"sglang-model","input":[{"text":"first"},{"text":"second"}],"encoding_format":"float"})
    );
    assert_eq!(response.embeddings(), &[vec![0.6, 0.8], vec![0.0, 1.0]]);
    assert_eq!(response.usage().input_tokens(), Some(7));
    assert_eq!(response.diagnostics().response_id(), Some("response-id"));
    assert_eq!(response.space().provider(), Provider::Sglang);
    assert_eq!(
        response.space().profile_revision(),
        "qwen3-vl-embedding-sglang-jinja-v1"
    );
}

#[tokio::test]
async fn image_video_and_combined_inputs_match_schema() {
    for (parts, expected) in [
        (
            vec![ContentPart::ImageDataUrl(png_data_url())],
            json!({"image": png_data_url().split_once(',').unwrap().1}),
        ),
        (
            vec![ContentPart::VideoSource(
                "https://media.example/video.mp4".into(),
            )],
            json!({"video":"https://media.example/video.mp4"}),
        ),
        (
            vec![
                ContentPart::ImageDataUrl(png_data_url()),
                ContentPart::VideoSource("video-base64".into()),
                ContentPart::Text("caption".into()),
            ],
            json!({"text":"caption","image":png_data_url().split_once(',').unwrap().1,"video":"video-base64"}),
        ),
    ] {
        let body = br#"{"data":[{"index":0,"embedding":[3,4]}]}"#;
        let (endpoint, server) =
            spawn_server(move |stream| write_chunked(stream, "200 OK", &[], body, 8).unwrap());
        client(endpoint, false)
            .embed_fusion(&parts, &EmbedOptions::default())
            .await
            .unwrap();
        let (request, ()) = server.join().unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
            json!({"model":"sglang-model","input":[expected],"encoding_format":"float"})
        );
    }
}

#[tokio::test]
async fn noncanonical_and_duplicate_fusion_parts_are_rejected() {
    let image = png_data_url();
    for parts in [
        vec![
            ContentPart::ImageDataUrl(image.clone()),
            ContentPart::ImageDataUrl(image),
        ],
        vec![
            ContentPart::VideoSource("one".into()),
            ContentPart::VideoSource("two".into()),
        ],
        vec![ContentPart::VideoSource(" ".into())],
        vec![
            ContentPart::Text("caption".into()),
            ContentPart::ImageDataUrl(png_data_url()),
        ],
        vec![
            ContentPart::VideoSource("video".into()),
            ContentPart::ImageDataUrl(png_data_url()),
        ],
        vec![
            ContentPart::ImageDataUrl(png_data_url()),
            ContentPart::Text("caption".into()),
            ContentPart::VideoSource("video".into()),
        ],
        vec![
            ContentPart::Text("first".into()),
            ContentPart::Text("second".into()),
        ],
        vec![ContentPart::Text(String::new())],
    ] {
        let error = client(Url::parse("http://127.0.0.1:9/").unwrap(), false)
            .embed_fusion(&parts, &EmbedOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
    }
}

#[tokio::test]
async fn api_error_and_dimension_mismatch_are_validated() {
    let (endpoint, server) = spawn_server(|stream| {
        write_chunked(
            stream,
            "400 Bad Request",
            &[],
            br#"{"error":{"message":"bad","code":"invalid"}}"#,
            8,
        )
        .unwrap()
    });
    assert!(matches!(
        client(endpoint, false)
            .embed_texts(&["x".into()], &EmbedOptions::default())
            .await,
        Err(Error::Api { .. })
    ));
    server.join().unwrap();
    let (endpoint, server) = spawn_server(|stream| {
        write_chunked(
            stream,
            "200 OK",
            &[],
            br#"{"data":[{"index":0,"embedding":[1]}]}"#,
            8,
        )
        .unwrap()
    });
    assert!(matches!(
        client(endpoint, false)
            .embed_texts(&["x".into()], &EmbedOptions::default())
            .await,
        Err(Error::Dimension { .. })
    ));
    server.join().unwrap();
}
