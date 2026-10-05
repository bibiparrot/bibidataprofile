use serde::ser::{Serialize, Serializer};
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("The user home directory could not be determined")]
    HomeDirectory,
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error(
        "The bundled uv executable was not found. Run `npm run sidecar` before development builds."
    )]
    UvNotFound,
    #[error("uv failed while {action}: {message}")]
    Uv { action: String, message: String },
    #[error("marimo stopped before its server became ready: {0}")]
    MarimoExited(String),
    #[error("marimo did not become ready within {0} seconds")]
    MarimoTimeout(u64),
    #[error("No free loopback port was available near {0}")]
    PortUnavailable(u16),
    #[error("Background task failed: {0}")]
    Background(String),
}

impl AppError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

pub type AppResult<T> = Result<T, AppError>;
