use crate::error::{AppError, AppResult};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

pub const PYPI_INDEX: &str = "https://pypi.org/simple";
pub const ALIYUN_INDEX: &str = "https://mirrors.aliyun.com/pypi/simple/";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AppConfig {
    /// `system` follows the operating system; otherwise use a supported locale code.
    pub language: String,
    pub python_version: String,
    pub pip_index_url: String,
    pub marimo_package: String,
    pub marimo_port: u16,
    pub startup_timeout_seconds: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            language: "system".into(),
            python_version: "3.12".into(),
            pip_index_url: PYPI_INDEX.into(),
            marimo_package: "marimo>=0.24.0,<0.25".into(),
            marimo_port: 2718,
            startup_timeout_seconds: 600,
        }
    }
}

impl AppConfig {
    pub fn validate(&self) -> AppResult<()> {
        if self.python_version.trim().is_empty() {
            return Err(AppError::InvalidConfig(
                "python_version cannot be empty".into(),
            ));
        }
        if !(self.pip_index_url.starts_with("https://")
            || self.pip_index_url.starts_with("http://"))
        {
            return Err(AppError::InvalidConfig(
                "pip_index_url must be an HTTP(S) URL".into(),
            ));
        }
        if self.marimo_package.trim().is_empty() {
            return Err(AppError::InvalidConfig(
                "marimo_package cannot be empty".into(),
            ));
        }
        if self.marimo_port == 0 {
            return Err(AppError::InvalidConfig(
                "marimo_port must be between 1 and 65535".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AppPaths {
    pub root: PathBuf,
    pub config: PathBuf,
    pub python: PathBuf,
    pub cache: PathBuf,
    pub venv: PathBuf,
    pub environment: PathBuf,
    pub notebooks: PathBuf,
    pub logs: PathBuf,
}

impl AppPaths {
    pub fn discover() -> AppResult<Self> {
        let root = dirs::home_dir()
            .ok_or(AppError::HomeDirectory)?
            .join(".bibidataprofile");
        Ok(Self {
            config: root.join("config.toml"),
            python: root.join("python"),
            cache: root.join("cache"),
            venv: root.join("venv"),
            environment: root.join("environment.toml"),
            notebooks: root.join("notebooks"),
            logs: root.join("logs"),
            root,
        })
    }

    pub fn ensure_directories(&self) -> AppResult<()> {
        for path in [
            &self.root,
            &self.python,
            &self.cache,
            &self.notebooks,
            &self.logs,
        ] {
            fs::create_dir_all(path).map_err(|source| AppError::io(path, source))?;
        }
        Ok(())
    }
}

pub fn load_or_create(paths: &AppPaths, system_locale: &str) -> AppResult<AppConfig> {
    paths.ensure_directories()?;
    if paths.config.exists() {
        let source = fs::read_to_string(&paths.config)
            .map_err(|error| AppError::io(&paths.config, error))?;
        let config: AppConfig =
            toml::from_str(&source).map_err(|error| AppError::InvalidConfig(error.to_string()))?;
        config.validate()?;
        return Ok(config);
    }

    let mut config = AppConfig::default();
    if system_locale.starts_with("zh") {
        config.pip_index_url = ALIYUN_INDEX.into();
    }
    save(paths, &config)?;
    Ok(config)
}

pub fn save(paths: &AppPaths, config: &AppConfig) -> AppResult<()> {
    config.validate()?;
    paths.ensure_directories()?;
    let source = toml::to_string_pretty(config)
        .map_err(|error| AppError::InvalidConfig(error.to_string()))?;
    fs::write(&paths.config, source).map_err(|error| AppError::io(&paths.config, error))
}

pub fn venv_python(paths: &AppPaths) -> PathBuf {
    if cfg!(windows) {
        paths.venv.join("Scripts").join("python.exe")
    } else {
        paths.venv.join("bin").join("python")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_round_trips_through_toml() {
        let expected = AppConfig::default();
        let encoded = toml::to_string(&expected).expect("default configuration should serialize");
        let decoded: AppConfig =
            toml::from_str(&encoded).expect("serialized configuration should parse");
        assert_eq!(decoded.python_version, "3.12");
        assert_eq!(decoded.pip_index_url, PYPI_INDEX);
        assert_eq!(decoded.language, "system");
    }

    #[test]
    fn configuration_rejects_non_http_mirror() {
        let config = AppConfig {
            pip_index_url: "file:///untrusted/index".into(),
            ..AppConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
