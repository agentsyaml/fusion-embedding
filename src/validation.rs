use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Deserialize;

use crate::Error;

const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
const MAX_IMAGE_EDGE: u32 = 16_384;
const MAX_IMAGE_PIXELS: u64 = 40_000_000;

#[derive(Deserialize)]
pub(crate) struct IndexedEmbedding {
    pub(crate) index: usize,
    pub(crate) embedding: Vec<f32>,
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
#[derive(Clone, Copy)]
enum ImageKind {
    Png,
    Jpeg,
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn validate_data_url(url: &str) -> Result<&str, Error> {
    let (kind, encoded) = if let Some(encoded) = url.strip_prefix("data:image/png;base64,") {
        (ImageKind::Png, encoded)
    } else if let Some(encoded) = url.strip_prefix("data:image/jpeg;base64,") {
        (ImageKind::Jpeg, encoded)
    } else {
        return Err(Error::InvalidInput {
            message: "image must be a strict PNG or JPEG base64 data URL".into(),
        });
    };
    if encoded.is_empty() {
        return Err(Error::InvalidInput {
            message: "image base64 payload must not be empty".into(),
        });
    }
    if encoded.len() % 4 != 0
        || encoded.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4
        || !valid_base64_alphabet(encoded.as_bytes())
    {
        return Err(Error::InvalidInput {
            message: "image payload is not canonical standard base64".into(),
        });
    }
    let decoded = STANDARD.decode(encoded).map_err(|_| Error::InvalidInput {
        message: "image payload is not canonical standard base64".into(),
    })?;
    if decoded.is_empty() || decoded.len() > MAX_IMAGE_BYTES || STANDARD.encode(&decoded) != encoded
    {
        return Err(Error::InvalidInput {
            message: format!("image must be non-empty and no larger than {MAX_IMAGE_BYTES} bytes"),
        });
    }
    let (width, height) = match kind {
        ImageKind::Png => png_dimensions(&decoded),
        ImageKind::Jpeg => jpeg_dimensions(&decoded),
    }
    .ok_or_else(|| Error::InvalidInput {
        message: "image header, magic bytes, or dimensions do not match its media type; this check does not fully decode the image".into(),
    })?;
    if width == 0
        || height == 0
        || width > MAX_IMAGE_EDGE
        || height > MAX_IMAGE_EDGE
        || u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS
    {
        return Err(Error::InvalidInput {
            message: "image dimensions exceed the supported safety limits".into(),
        });
    }
    Ok(encoded)
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
fn valid_base64_alphabet(value: &[u8]) -> bool {
    let padding = value.iter().rev().take_while(|&&byte| byte == b'=').count();
    padding <= 2
        && value[..value.len() - padding]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
        && value[value.len() - padding..]
            .iter()
            .all(|&byte| byte == b'=')
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    // Header-only preflight: require the complete IHDR chunk, but leave CRC and
    // full image decoding to the provider.
    if bytes.len() < 33
        || bytes[..8] != [137, 80, 78, 71, 13, 10, 26, 10]
        || bytes[8..12] != [0, 0, 0, 13]
        || &bytes[12..16] != b"IHDR"
    {
        return None;
    }
    Some((
        u32::from_be_bytes(bytes[16..20].try_into().ok()?),
        u32::from_be_bytes(bytes[20..24].try_into().ok()?),
    ))
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 4 || bytes[..3] != [0xff, 0xd8, 0xff] {
        return None;
    }
    let mut position = 2;
    while position < bytes.len() {
        while position < bytes.len() && bytes[position] == 0xff {
            position += 1;
        }
        let marker = *bytes.get(position)?;
        position += 1;
        if marker == 0xd9 || marker == 0xda {
            return None;
        }
        if marker == 0x01 || (0xd0..=0xd8).contains(&marker) {
            continue;
        }
        let length = usize::from(u16::from_be_bytes([
            *bytes.get(position)?,
            *bytes.get(position + 1)?,
        ]));
        if length < 2 || position.checked_add(length)? > bytes.len() {
            return None;
        }
        if matches!(
            marker,
            0xc0 | 0xc1
                | 0xc2
                | 0xc3
                | 0xc5
                | 0xc6
                | 0xc7
                | 0xc9
                | 0xca
                | 0xcb
                | 0xcd
                | 0xce
                | 0xcf
        ) {
            if length < 7 {
                return None;
            }
            let height = u32::from(u16::from_be_bytes([
                bytes[position + 3],
                bytes[position + 4],
            ]));
            let width = u32::from(u16::from_be_bytes([
                bytes[position + 5],
                bytes[position + 6],
            ]));
            return Some((width, height));
        }
        position += length;
    }
    None
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn validate_embeddings(
    data: Vec<IndexedEmbedding>,
    expected_count: usize,
    expected_dimensions: usize,
) -> Result<Vec<Vec<f32>>, Error> {
    expected_count
        .checked_mul(expected_dimensions)
        .ok_or_else(|| Error::InvalidResponse {
            message: "embedding shape overflows addressable memory".into(),
        })?;
    if data.len() != expected_count {
        return Err(Error::InvalidResponse {
            message: format!(
                "expected {expected_count} embeddings, received {}",
                data.len()
            ),
        });
    }
    let mut ordered: Vec<Option<Vec<f32>>> = vec![None; expected_count];
    for item in data {
        if item.index >= expected_count {
            return Err(Error::InvalidResponse {
                message: format!("embedding index {} is out of range", item.index),
            });
        }
        if ordered[item.index].is_some() {
            return Err(Error::InvalidResponse {
                message: format!("embedding index {} is duplicated", item.index),
            });
        }
        if item.embedding.len() != expected_dimensions {
            return Err(Error::Dimension {
                expected: expected_dimensions,
                actual: item.embedding.len(),
            });
        }
        ordered[item.index] = Some(normalize_vector(item.embedding)?);
    }
    ordered
        .into_iter()
        .map(|item| {
            item.ok_or_else(|| Error::InvalidResponse {
                message: "provider response is missing an embedding index".into(),
            })
        })
        .collect()
}

#[cfg(any(feature = "vllm", feature = "llama-cpp", feature = "dashscope"))]
pub(crate) fn normalize_vector(vector: Vec<f32>) -> Result<Vec<f32>, Error> {
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidResponse {
            message: "embedding contains a non-finite value".into(),
        });
    }
    let norm = vector
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(Error::InvalidResponse {
            message: "embedding has a zero or non-finite norm".into(),
        });
    }
    let output: Vec<f32> = vector
        .into_iter()
        .map(|value| (f64::from(value) / norm) as f32)
        .collect();
    let output_norm_squared = output
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    if output.iter().any(|value| !value.is_finite())
        || !output_norm_squared.is_finite()
        || output_norm_squared == 0.0
    {
        return Err(Error::InvalidResponse {
            message: "normalized embedding is not finite and nonzero".into(),
        });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD;

    fn png_data_url(width: u32, height: u32) -> String {
        let mut bytes = vec![137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13];
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
        bytes.extend_from_slice(&[0; 4]);
        format!("data:image/png;base64,{}", STANDARD.encode(bytes))
    }

    fn jpeg_data_url(width: u16, height: u16) -> String {
        let mut bytes = vec![0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08];
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&[0x01, 0x01, 0x11, 0x00, 0xff, 0xd9]);
        format!("data:image/jpeg;base64,{}", STANDARD.encode(bytes))
    }

    fn item(index: usize, embedding: Vec<f32>) -> IndexedEmbedding {
        IndexedEmbedding { index, embedding }
    }

    #[test]
    fn data_url_rejects_bad_empty_magic_and_limits() {
        assert!(validate_data_url("https://example/image.png").is_err());
        assert!(validate_data_url("data:image/png;base64,").is_err());
        assert!(validate_data_url("data:image/png;base64,!!!!").is_err());
        assert!(validate_data_url("data:image/png;base64,eA==").is_err());
        let mut truncated_ihdr = STANDARD
            .decode(png_data_url(2, 3).split_once(',').unwrap().1)
            .unwrap();
        truncated_ihdr.truncate(32);
        assert!(
            validate_data_url(&format!(
                "data:image/png;base64,{}",
                STANDARD.encode(truncated_ihdr)
            ))
            .is_err()
        );
        assert!(
            validate_data_url(&jpeg_data_url(2, 3).replacen("image/jpeg", "image/png", 1)).is_err()
        );
        assert!(validate_data_url(&png_data_url(MAX_IMAGE_EDGE + 1, 1)).is_err());
        assert!(validate_data_url(&png_data_url(10_000, 10_000)).is_err());

        let oversized = format!(
            "data:image/png;base64,{}",
            "A".repeat(MAX_IMAGE_BYTES.div_ceil(3) * 4 + 4)
        );
        assert!(validate_data_url(&oversized).is_err());
        assert!(validate_data_url(&png_data_url(2, 3)).is_ok());
        assert!(validate_data_url(&jpeg_data_url(2, 3)).is_ok());
    }
    #[test]
    fn response_indices_are_complete_unique_and_sorted() {
        let result =
            validate_embeddings(vec![item(1, vec![0.0, 2.0]), item(0, vec![3.0, 4.0])], 2, 2)
                .unwrap();
        assert_eq!(result[0], vec![0.6, 0.8]);
        assert_eq!(result[1], vec![0.0, 1.0]);
        assert!(validate_embeddings(vec![item(0, vec![1.0])], 2, 1).is_err());
        assert!(validate_embeddings(vec![item(0, vec![1.0]), item(0, vec![1.0])], 2, 1).is_err());
        assert!(validate_embeddings(vec![item(1, vec![1.0])], 1, 1).is_err());
    }
    #[test]
    fn response_rejects_dimensions_nonfinite_and_zero() {
        assert!(matches!(
            validate_embeddings(vec![item(0, vec![1.0])], 1, 2),
            Err(Error::Dimension { .. })
        ));
        for vector in [
            vec![f32::NAN, 1.0],
            vec![f32::INFINITY, 1.0],
            vec![f32::NEG_INFINITY, 1.0],
            vec![0.0, 0.0],
        ] {
            assert!(validate_embeddings(vec![item(0, vector)], 1, 2).is_err());
        }
    }
    #[test]
    fn normalization_uses_l2_and_stays_finite() {
        let output = normalize_vector(vec![3.0, 4.0]).unwrap();
        let norm = output
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
