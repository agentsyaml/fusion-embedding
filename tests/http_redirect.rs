#![cfg(any(
    feature = "vllm",
    feature = "llama-cpp",
    feature = "dashscope",
    feature = "sglang"
))]

mod common;

use std::{
    io,
    net::TcpListener,
    num::NonZeroUsize,
    sync::mpsc::{self, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use common::*;
#[cfg(feature = "dashscope")]
use fusion_embedding::DashScopeConfig;
#[cfg(feature = "llama-cpp")]
use fusion_embedding::LlamaCppConfig;
#[cfg(feature = "sglang")]
use fusion_embedding::SglangConfig;
#[cfg(feature = "vllm")]
use fusion_embedding::VllmConfig;
use fusion_embedding::{Client, EmbedOptions, Error};
#[cfg(feature = "dashscope")]
use secrecy::SecretString;
use url::Url;

#[cfg(feature = "vllm")]
fn client(endpoint: Url) -> Client {
    Client::vllm(
        VllmConfig::new(
            endpoint,
            "configured-model",
            "test-pipeline",
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[cfg(feature = "llama-cpp")]
fn llama_client(endpoint: Url) -> Client {
    Client::llama_cpp(
        LlamaCppConfig::new(
            endpoint,
            "identity-only-model",
            "llama-test-pipeline",
            NonZeroUsize::new(2).unwrap(),
            "<fixed-marker>",
        )
        .unwrap(),
    )
    .unwrap()
}

#[cfg(feature = "dashscope")]
fn dashscope_client(mut endpoint: Url) -> Client {
    endpoint.set_path("/api/native/embeddings");
    Client::dashscope(
        DashScopeConfig::new(
            endpoint,
            "dashscope-model",
            "dashscope-test-pipeline",
            NonZeroUsize::new(2).unwrap(),
            SecretString::from("dashscope-test-key".to_owned()),
        )
        .unwrap(),
    )
    .unwrap()
}

#[cfg(feature = "sglang")]
fn sglang_client(endpoint: Url) -> Client {
    Client::sglang(
        SglangConfig::new(
            endpoint,
            "sglang-model",
            "sglang-pipeline",
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

async fn assert_redirect_not_followed<F>(status: &'static str, make_client: F)
where
    F: FnOnce(Url) -> Client,
{
    let target = TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let location = format!("http://{}/redirected", target.local_addr().unwrap());
    let (stop_tx, stop_rx) = mpsc::channel();
    let probe = thread::spawn(move || {
        let deadline = Instant::now() + IO_TIMEOUT;
        loop {
            match target.accept() {
                Ok(_) => return true,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    match stop_rx.try_recv() {
                        Ok(()) | Err(TryRecvError::Disconnected) => return false,
                        Err(TryRecvError::Empty) => {}
                    }
                    assert!(Instant::now() < deadline, "redirect probe timed out");
                    thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("redirect probe failed: {error}"),
            }
        }
    });
    let (endpoint, server) = spawn_server(move |stream| {
        write_chunked(stream, status, &[("Location", &location)], b"", 1).unwrap();
    });

    let client = make_client(endpoint);
    let result = client
        .embed_texts(
            &["first".to_owned(), "second".to_owned()],
            &EmbedOptions::default(),
        )
        .await;
    let (request, ()) = server.join().unwrap();
    let _ = (
        &request.method,
        &request.path,
        &request.headers,
        &request.body,
    );
    stop_tx.send(()).unwrap();
    let followed = probe.join().unwrap();
    assert!(matches!(result, Err(Error::Api { .. })));
    assert!(!followed, "client followed {status}");
}

#[tokio::test]
async fn redirects_are_api_errors_and_never_reach_the_location() {
    for status in ["307 Temporary Redirect", "308 Permanent Redirect"] {
        #[cfg(feature = "vllm")]
        assert_redirect_not_followed(status, client).await;
        #[cfg(feature = "llama-cpp")]
        assert_redirect_not_followed(status, llama_client).await;
        #[cfg(feature = "dashscope")]
        assert_redirect_not_followed(status, dashscope_client).await;
        #[cfg(feature = "sglang")]
        assert_redirect_not_followed(status, sglang_client).await;
    }
}
