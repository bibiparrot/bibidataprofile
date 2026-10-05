use crate::{
    config::{self, AppConfig, AppPaths},
    error::{AppError, AppResult},
    locale::{self, LocaleOption},
    runtime::{RuntimeInfo, SharedRuntime},
};
use serde::Serialize;
use std::{collections::HashMap, sync::Arc};
use tauri::State;

pub struct AppState {
    pub runtime: Arc<SharedRuntime>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BootstrapResponse {
    config: AppConfig,
    config_path: String,
    effective_locale: String,
    translations: HashMap<String, String>,
    locales: Vec<LocaleOption>,
}

#[tauri::command]
pub fn bootstrap() -> AppResult<BootstrapResponse> {
    let paths = AppPaths::discover()?;
    let config = config::load_or_create(&paths, &locale::system_locale())?;
    let effective_locale = locale::activate(&config);
    Ok(BootstrapResponse {
        config,
        config_path: paths.config.display().to_string(),
        effective_locale,
        translations: locale::translations(),
        locales: locale::options(),
    })
}

#[tauri::command]
pub fn save_settings(mut config: AppConfig) -> AppResult<BootstrapResponse> {
    config.language = if config.language == "system" {
        config.language
    } else {
        locale::normalize(&config.language).into()
    };
    let paths = AppPaths::discover()?;
    config::save(&paths, &config)?;
    bootstrap()
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
pub fn runtime_status(state: State<'_, AppState>) -> RuntimeInfo {
    state.runtime.status()
}

#[tauri::command]
pub async fn start_marimo(state: State<'_, AppState>) -> AppResult<RuntimeInfo> {
    let runtime = Arc::clone(&state.runtime);
    let paths = AppPaths::discover()?;
    let config = config::load_or_create(&paths, &locale::system_locale())?;
    tauri::async_runtime::spawn_blocking(move || runtime.start(&config, &paths))
        .await
        .map_err(|error| AppError::Background(error.to_string()))?
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
pub fn stop_marimo(state: State<'_, AppState>) {
    state.runtime.stop();
}
