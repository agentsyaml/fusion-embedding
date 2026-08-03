use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    thread,
    time::{Duration, Instant},
};

use url::Url;

pub(crate) const IO_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) body: Vec<u8>,
}

pub(crate) fn read_request(stream: &mut TcpStream) -> io::Result<Request> {
    let mut raw = Vec::new();
    let mut buffer = [0; 4096];
    let header_end = loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before headers",
            ));
        }
        raw.extend_from_slice(&buffer[..read]);
        if let Some(position) = raw.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            break position + 4;
        }
    };

    let head = std::str::from_utf8(&raw[..header_end - 4])
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split_whitespace();
    let method = request_line.next().unwrap_or_default().to_owned();
    let path = request_line.next().unwrap_or_default().to_owned();
    let headers: HashMap<_, _> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    while raw.len() - header_end < content_length {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before body",
            ));
        }
        raw.extend_from_slice(&buffer[..read]);
    }

    Ok(Request {
        method,
        path,
        headers,
        body: raw[header_end..header_end + content_length].to_vec(),
    })
}

pub(crate) fn spawn_server<F, R>(respond: F) -> (Url, thread::JoinHandle<(Request, R)>)
where
    F: FnOnce(&mut TcpStream) -> R + Send + 'static,
    R: Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
    let handle = thread::spawn(move || {
        let deadline = Instant::now() + IO_TIMEOUT;
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < deadline,
                        "server timed out waiting for request"
                    );
                    thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("server accept failed: {error}"),
            }
        };
        stream.set_nonblocking(false).unwrap();
        stream.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
        stream.set_write_timeout(Some(IO_TIMEOUT)).unwrap();
        let request = read_request(&mut stream).unwrap();
        let result = respond(&mut stream);
        (request, result)
    });
    (endpoint, handle)
}

pub(crate) fn write_chunked(
    stream: &mut TcpStream,
    status: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    chunk_size: usize,
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nTransfer-Encoding: chunked\r\n"
    )?;
    for (name, value) in headers {
        write!(stream, "{name}: {value}\r\n")?;
    }
    write!(stream, "Connection: close\r\n\r\n")?;
    for chunk in body.chunks(chunk_size) {
        write!(stream, "{:x}\r\n", chunk.len())?;
        stream.write_all(chunk)?;
        stream.write_all(b"\r\n")?;
    }
    stream.write_all(b"0\r\n\r\n")
}
