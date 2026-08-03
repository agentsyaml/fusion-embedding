#![cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]

use std::{env, fs, num::NonZeroUsize};
#[cfg(feature = "llama-cpp")]
use std::{net::IpAddr, time::Duration};

use base64::{Engine as _, engine::general_purpose::STANDARD};
#[cfg(feature = "dashscope")]
use fusion_embedding::DashScopeConfig;
#[cfg(feature = "llama-cpp")]
use fusion_embedding::LlamaCppConfig;
#[cfg(feature = "sglang")]
use fusion_embedding::SglangConfig;
#[cfg(feature = "vllm")]
use fusion_embedding::VllmConfig;
use fusion_embedding::{
    Client, ContentPart, DimensionMode, EmbedOptions, EmbeddingResponse, EmbeddingSpace, Provider,
};
#[cfg(any(feature = "vllm", feature = "dashscope", feature = "sglang"))]
use secrecy::SecretString;
#[cfg(feature = "llama-cpp")]
use serde_json::Value;
use url::Url;

#[cfg(feature = "llama-cpp")]
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);
#[cfg(feature = "llama-cpp")]
const MAX_PROPS_BYTES: usize = 1024 * 1024;
const COCO_RETRIEVAL_THRESHOLD: f64 = 0.50;
// Project-authored test descriptions based only on visible content, not COCO captions.
const COCO_TEST_DESCRIPTIONS: [&str; 3] = [
    "Two cats sleep beside remote controls.",
    "Three skiers descend a steep snowy slope.",
    "Three elephants stand on grass among trees.",
];
const COCO_CASES: [(u64, &str, usize); 3] = [
    (39769, "samples/coco-mini/000000039769.jpg", 0),
    (274708, "samples/coco-mini/000000274708.jpg", 1),
    (126216, "samples/coco-mini/000000126216.jpg", 2),
];

fn required(name: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("required environment variable {name} is missing or empty"))
}

fn optional(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn allow_remote_http(name: &str) -> bool {
    match env::var(name).as_deref() {
        Err(env::VarError::NotPresent) | Ok("") | Ok("false") => false,
        Ok("true") => true,
        _ => panic!("environment variable {name} must be true or false"),
    }
}

fn dimensions(name: &str) -> NonZeroUsize {
    required(name)
        .parse::<usize>()
        .ok()
        .and_then(NonZeroUsize::new)
        .unwrap_or_else(|| panic!("environment variable {name} must be a positive integer"))
}

#[cfg(any(feature = "llama-cpp", feature = "dashscope"))]
fn url(name: &str) -> Url {
    Url::parse(&required(name))
        .unwrap_or_else(|_| panic!("environment variable {name} must be a valid URL"))
}

#[cfg(feature = "vllm")]
fn vllm_url() -> Url {
    let (name, value) = optional("VLLM_MODELS_URL")
        .map(|value| ("VLLM_MODELS_URL", value))
        .unwrap_or_else(|| ("VLLM_BASE_URL", required("VLLM_BASE_URL")));
    Url::parse(&value).unwrap_or_else(|_| panic!("environment variable {name} must be a valid URL"))
}

#[cfg(feature = "sglang")]
fn sglang_url() -> Url {
    let (name, value) = optional("SGLANG_MODELS_URL")
        .map(|value| ("SGLANG_MODELS_URL", value))
        .unwrap_or_else(|| ("SGLANG_BASE_URL", required("SGLANG_BASE_URL")));
    Url::parse(&value).unwrap_or_else(|_| panic!("environment variable {name} must be a valid URL"))
}

fn image_data_url() -> String {
    image_data_url_from_path(&required("E2E_IMAGE_PATH"))
}

fn image_data_url_from_path(path: &str) -> String {
    let bytes = fs::read(path).unwrap_or_else(|_| panic!("image fixture cannot be read: {path}"));
    let media_type = if bytes.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]) {
        "image/png"
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        "image/jpeg"
    } else {
        panic!("image fixture must contain a PNG or JPEG image: {path}");
    };
    format!("data:{media_type};base64,{}", STANDARD.encode(bytes))
}

fn cosine(left: &[f32], right: &[f32]) -> Result<f64, &'static str> {
    if left.is_empty() || left.len() != right.len() {
        return Err("vectors must have equal non-zero dimensions");
    }
    let (dot, left_norm, right_norm) = left
        .iter()
        .zip(right)
        .try_fold(
            (0.0, 0.0, 0.0),
            |(dot, left_norm, right_norm), (&left, &right)| {
                if !left.is_finite() || !right.is_finite() {
                    return None;
                }
                let left = f64::from(left);
                let right = f64::from(right);
                Some((
                    dot + left * right,
                    left_norm + left * left,
                    right_norm + right * right,
                ))
            },
        )
        .ok_or("vectors must be finite")?;
    if !dot.is_finite() || !left_norm.is_finite() || !right_norm.is_finite() {
        return Err("cosine calculation is not finite");
    }
    let denominator = (left_norm * right_norm).sqrt();
    if denominator == 0.0 || !denominator.is_finite() {
        return Err("vectors must have non-zero norms");
    }
    Ok(dot / denominator)
}

#[test]
fn cosine_checks_shape_and_finite_values() {
    assert!((cosine(&[3.0, 4.0], &[0.0, 5.0]).unwrap() - 0.8).abs() < 1e-12);
    assert!(cosine(&[1.0], &[1.0, 0.0]).is_err());
    assert!(cosine(&[f32::NAN], &[1.0]).is_err());
}

async fn coco_retrieval_sanity(client: &Client, provider: Provider) {
    let descriptions = COCO_TEST_DESCRIPTIONS.map(str::to_owned);
    let description_vectors = client
        .embed_texts(&descriptions, &EmbedOptions::default())
        .await
        .unwrap_or_else(|error| panic!("COCO description embeddings failed: {error:?}"))
        .into_embeddings();

    for (image_id, path, matched_index) in COCO_CASES {
        let image_vector = client
            .embed_fusion(
                &[ContentPart::ImageDataUrl(image_data_url_from_path(path))],
                &EmbedOptions::default(),
            )
            .await
            .unwrap_or_else(|error| panic!("COCO image embedding failed: {error:?}"))
            .into_embeddings()
            .pop()
            .expect("COCO image embedding response must contain one vector");
        let scores: Vec<f64> = description_vectors
            .iter()
            .map(|description| {
                cosine(&image_vector, description).unwrap_or_else(|error| {
                    panic!(
                        "COCO cosine failed for provider={provider:?} image_id={image_id}: {error}"
                    )
                })
            })
            .collect();
        let matched = scores[matched_index];
        let best_mismatch = scores
            .iter()
            .enumerate()
            .filter_map(|(index, &score)| (index != matched_index).then_some(score))
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "E2E retrieval provider={provider:?} image_id={image_id} matched={matched:.6} best_mismatch={best_mismatch:.6}"
        );
        assert!(
            matched >= COCO_RETRIEVAL_THRESHOLD,
            "COCO retrieval failed: provider={provider:?} image_id={image_id} matched={matched:.6} best_mismatch={best_mismatch:.6} threshold={COCO_RETRIEVAL_THRESHOLD:.2}"
        );
        assert!(
            matched > best_mismatch,
            "COCO retrieval failed: provider={provider:?} image_id={image_id} matched={matched:.6} best_mismatch={best_mismatch:.6} threshold={COCO_RETRIEVAL_THRESHOLD:.2}"
        );
    }
}

#[cfg(feature = "llama-cpp")]
fn endpoint_is_loopback(endpoint: &Url) -> bool {
    endpoint.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    })
}

#[cfg(feature = "llama-cpp")]
async fn discover_llama_marker(models_url: &Url, remote_http_allowed: bool) -> String {
    assert!(
        models_url.username().is_empty()
            && models_url.password().is_none()
            && models_url.query().is_none()
            && models_url.fragment().is_none(),
        "LLAMA_CPP_MODELS_URL must not contain userinfo, query, or fragment"
    );
    assert!(
        matches!(models_url.path(), "/" | "/v1/models"),
        "LLAMA_CPP_MODELS_URL path must be / or /v1/models"
    );
    assert!(
        models_url.scheme() == "https"
            || (models_url.scheme() == "http"
                && (endpoint_is_loopback(models_url) || remote_http_allowed)),
        "LLAMA_CPP_MODELS_URL remote HTTP requires LLAMA_CPP_ALLOW_REMOTE_HTTP=true"
    );

    let mut props_url = models_url.clone();
    props_url.set_path("/props");
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(PROBE_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap_or_else(|_| panic!("failed to construct llama.cpp probe client"));
    let mut response = client
        .get(props_url)
        .send()
        .await
        .unwrap_or_else(|_| panic!("llama.cpp /props request failed"));
    assert!(
        response.status().is_success(),
        "llama.cpp /props returned a non-success status"
    );
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .unwrap_or_else(|_| panic!("llama.cpp /props response read failed"))
    {
        assert!(
            body.len().saturating_add(chunk.len()) <= MAX_PROPS_BYTES,
            "llama.cpp /props response exceeds the E2E limit"
        );
        body.extend_from_slice(&chunk);
    }
    let props: Value = serde_json::from_slice(&body)
        .unwrap_or_else(|_| panic!("llama.cpp /props returned invalid JSON"));
    assert_eq!(
        props.pointer("/modalities/vision").and_then(Value::as_bool),
        Some(true),
        "llama.cpp /props does not report vision=true"
    );
    props
        .get("media_marker")
        .and_then(Value::as_str)
        .filter(|marker| !marker.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| panic!("llama.cpp /props has no non-empty media_marker"))
}

fn validate_response(
    response: &EmbeddingResponse,
    provider: Provider,
    dimension_mode: DimensionMode,
    count: usize,
    expected_dimensions: NonZeroUsize,
    fixed_space: &mut Option<EmbeddingSpace>,
) {
    assert_eq!(response.space().provider(), provider);
    assert_eq!(response.space().dimension_mode(), dimension_mode);
    assert_eq!(response.space().expected_dimensions(), expected_dimensions);
    if let Some(space) = fixed_space {
        assert_eq!(
            response.space(),
            space,
            "embedding space changed between calls"
        );
    } else {
        *fixed_space = Some(response.space().clone());
    }
    assert_eq!(response.embeddings().len(), count);
    for vector in response.embeddings() {
        assert_eq!(vector.len(), expected_dimensions.get());
        assert!(vector.iter().all(|value| value.is_finite()));
        let norm = vector
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "embedding is not unit L2");
    }
}

async fn exercise_provider(
    client: &Client,
    provider: Provider,
    dimension_mode: DimensionMode,
    expected_dimensions: NonZeroUsize,
    image: String,
    repeat_image: bool,
) {
    let options = EmbedOptions::default();
    let mut fixed_space = None;
    let one_text = vec!["single E2E text".to_owned()];
    let response = client
        .embed_texts(&one_text, &options)
        .await
        .unwrap_or_else(|error| panic!("single-text embedding failed: {error:?}"));
    validate_response(
        &response,
        provider,
        dimension_mode,
        1,
        expected_dimensions,
        &mut fixed_space,
    );

    let two_texts = vec!["first E2E text".to_owned(), "second E2E text".to_owned()];
    let response = client
        .embed_texts(&two_texts, &options)
        .await
        .unwrap_or_else(|error| panic!("two-text embedding failed: {error:?}"));
    validate_response(
        &response,
        provider,
        dimension_mode,
        2,
        expected_dimensions,
        &mut fixed_space,
    );

    let response = client
        .embed_fusion(&[ContentPart::ImageDataUrl(image.clone())], &options)
        .await
        .unwrap_or_else(|error| panic!("image-only embedding failed: {error:?}"));
    validate_response(
        &response,
        provider,
        dimension_mode,
        1,
        expected_dimensions,
        &mut fixed_space,
    );

    let response = client
        .embed_fusion(
            &[
                ContentPart::ImageDataUrl(image.clone()),
                ContentPart::Text("E2E caption".to_owned()),
            ],
            &options,
        )
        .await
        .unwrap_or_else(|error| panic!("image-caption embedding failed: {error:?}"));
    validate_response(
        &response,
        provider,
        dimension_mode,
        1,
        expected_dimensions,
        &mut fixed_space,
    );

    if repeat_image {
        let response = client
            .embed_fusion(
                &[
                    ContentPart::ImageDataUrl(image.clone()),
                    ContentPart::ImageDataUrl(image),
                ],
                &options,
            )
            .await
            .unwrap_or_else(|error| panic!("repeated-image embedding failed: {error:?}"));
        validate_response(
            &response,
            provider,
            dimension_mode,
            1,
            expected_dimensions,
            &mut fixed_space,
        );
    }

    let (calls, embeddings) = if repeat_image { (5, 6) } else { (4, 5) };
    println!(
        "E2E provider={provider:?} mode={dimension_mode:?} calls={calls} embeddings={embeddings} dimensions={}",
        expected_dimensions.get()
    );
    coco_retrieval_sanity(client, provider).await;
}

#[cfg(feature = "llama-cpp")]
#[tokio::test]
#[ignore = "requires an explicitly configured live llama.cpp endpoint"]
async fn llama_cpp_e2e() {
    let models_url = url("LLAMA_CPP_MODELS_URL");
    let expected_dimensions = dimensions("LLAMA_CPP_DIMENSIONS");
    let remote_http_allowed = allow_remote_http("LLAMA_CPP_ALLOW_REMOTE_HTTP");
    let marker = match optional("LLAMA_CPP_MEDIA_MARKER") {
        Some(marker) => marker,
        None => discover_llama_marker(&models_url, remote_http_allowed).await,
    };
    let mut config = LlamaCppConfig::new(
        models_url,
        required("LLAMA_CPP_CONFIGURED_MODEL"),
        required("LLAMA_CPP_PIPELINE_ID"),
        expected_dimensions,
        marker,
    )
    .unwrap_or_else(|error| panic!("invalid llama.cpp E2E configuration: {error:?}"));
    if remote_http_allowed {
        config = config.allow_remote_http(true);
    }
    let client = Client::llama_cpp(config)
        .unwrap_or_else(|error| panic!("llama.cpp client construction failed: {error:?}"));
    exercise_provider(
        &client,
        Provider::LlamaCpp,
        DimensionMode::Native,
        expected_dimensions,
        image_data_url(),
        false,
    )
    .await;
}

#[cfg(feature = "vllm")]
#[tokio::test]
#[ignore = "requires an explicitly configured live vLLM endpoint"]
async fn vllm_e2e() {
    let expected_dimensions = dimensions("VLLM_DIMENSIONS");
    let remote_http_allowed = allow_remote_http("VLLM_ALLOW_REMOTE_HTTP");
    let send_dimensions = allow_remote_http("VLLM_SEND_DIMENSIONS");
    let mut config = VllmConfig::new(
        vllm_url(),
        required("VLLM_CONFIGURED_MODEL"),
        required("VLLM_PIPELINE_ID"),
        expected_dimensions,
    )
    .unwrap_or_else(|error| panic!("invalid vLLM E2E configuration: {error:?}"));
    let dimension_mode = if send_dimensions {
        config = config.with_server_dimensions(expected_dimensions);
        DimensionMode::Server
    } else {
        DimensionMode::Native
    };
    if let Some(api_key) = optional("VLLM_API_KEY") {
        config = config
            .with_api_key(SecretString::from(api_key))
            .unwrap_or_else(|error| panic!("invalid vLLM API key: {error:?}"));
    }
    if remote_http_allowed {
        config = config.allow_remote_http(true);
    }
    let client = Client::vllm(config)
        .unwrap_or_else(|error| panic!("vLLM client construction failed: {error:?}"));
    exercise_provider(
        &client,
        Provider::Vllm,
        dimension_mode,
        expected_dimensions,
        image_data_url(),
        false,
    )
    .await;
}

#[cfg(feature = "sglang")]
#[tokio::test]
#[ignore = "requires an explicitly configured live SGLang v0.5.11+ endpoint"]
async fn sglang_e2e() {
    let expected_dimensions = dimensions("SGLANG_DIMENSIONS");
    let remote_http_allowed = allow_remote_http("SGLANG_ALLOW_REMOTE_HTTP");
    let mut config = SglangConfig::new(
        sglang_url(),
        required("SGLANG_MODEL"),
        required("SGLANG_PIPELINE_ID"),
        expected_dimensions,
    )
    .unwrap_or_else(|error| panic!("invalid SGLang E2E configuration: {error:?}"));
    if let Some(api_key) = optional("SGLANG_API_KEY") {
        config = config
            .with_api_key(SecretString::from(api_key))
            .unwrap_or_else(|error| panic!("invalid SGLang API key: {error:?}"));
    }
    if remote_http_allowed {
        config = config.allow_remote_http(true);
    }
    let client = Client::sglang(config)
        .unwrap_or_else(|error| panic!("SGLang client construction failed: {error:?}"));
    exercise_provider(
        &client,
        Provider::Sglang,
        DimensionMode::Native,
        expected_dimensions,
        image_data_url(),
        false,
    )
    .await;
    let video = required("SGLANG_VIDEO_SOURCE");
    let options = EmbedOptions::default();
    let mut fixed_space = None;
    let video_only = client
        .embed_fusion(&[ContentPart::VideoSource(video.clone())], &options)
        .await
        .unwrap_or_else(|error| panic!("SGLang video-only embedding failed: {error:?}"));
    validate_response(
        &video_only,
        Provider::Sglang,
        DimensionMode::Native,
        1,
        expected_dimensions,
        &mut fixed_space,
    );
    let mixed = client
        .embed_fusion(
            &[
                ContentPart::ImageDataUrl(image_data_url()),
                ContentPart::VideoSource(video),
                ContentPart::Text("SGLang E2E caption".into()),
            ],
            &options,
        )
        .await
        .unwrap_or_else(|error| panic!("SGLang text-image-video embedding failed: {error:?}"));
    validate_response(
        &mixed,
        Provider::Sglang,
        DimensionMode::Native,
        1,
        expected_dimensions,
        &mut fixed_space,
    );
}

#[cfg(feature = "dashscope")]
async fn dashscope_e2e(
    endpoint_name: &str,
    key_name: &str,
    model_name: &str,
    pipeline_name: &str,
    native_dimensions_name: &str,
    dimensions_name: &str,
    allow_remote_http_name: Option<&str>,
) {
    let native_dimensions = dimensions(native_dimensions_name);
    let mut config = DashScopeConfig::new(
        url(endpoint_name),
        required(model_name),
        required(pipeline_name),
        native_dimensions,
        SecretString::from(required(key_name)),
    )
    .unwrap_or_else(|error| panic!("invalid DashScope E2E configuration: {error:?}"));
    let (expected_dimensions, dimension_mode) = match optional(dimensions_name) {
        Some(value) => {
            let dimensions = value
                .parse::<usize>()
                .ok()
                .and_then(NonZeroUsize::new)
                .unwrap_or_else(|| {
                    panic!("environment variable {dimensions_name} must be a positive integer")
                });
            config = config
                .with_server_dimensions(dimensions)
                .unwrap_or_else(|error| panic!("invalid DashScope server dimensions: {error:?}"));
            (dimensions, DimensionMode::Server)
        }
        None => (native_dimensions, DimensionMode::Native),
    };
    if allow_remote_http_name.is_some_and(allow_remote_http) {
        config = config.allow_remote_http(true);
    }
    let client = Client::dashscope(config)
        .unwrap_or_else(|error| panic!("DashScope client construction failed: {error:?}"));
    exercise_provider(
        &client,
        Provider::DashScope,
        dimension_mode,
        expected_dimensions,
        image_data_url(),
        true,
    )
    .await;
}

#[cfg(feature = "dashscope")]
#[tokio::test]
#[ignore = "requires an explicitly configured live DashScope official endpoint"]
async fn dashscope_official_e2e() {
    dashscope_e2e(
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_CONFIGURED_MODEL",
        "DASHSCOPE_PIPELINE_ID",
        "DASHSCOPE_NATIVE_DIMENSIONS",
        "DASHSCOPE_DIMENSIONS",
        None,
    )
    .await;
}

#[cfg(feature = "dashscope")]
#[tokio::test]
#[ignore = "requires an explicitly configured live DashScope enterprise endpoint"]
async fn dashscope_enterprise_e2e() {
    dashscope_e2e(
        "DASHSCOPE_EMBEDDING_BASE_URL",
        "DASHSCOPE_EMBEDDING_API_KEY",
        "DASHSCOPE_ENTERPRISE_CONFIGURED_MODEL",
        "DASHSCOPE_ENTERPRISE_PIPELINE_ID",
        "DASHSCOPE_ENTERPRISE_NATIVE_DIMENSIONS",
        "DASHSCOPE_ENTERPRISE_DIMENSIONS",
        Some("DASHSCOPE_ENTERPRISE_ALLOW_REMOTE_HTTP"),
    )
    .await;
}
