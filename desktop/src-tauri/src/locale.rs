use crate::config::AppConfig;
use serde::Serialize;
use std::collections::HashMap;

pub const SUPPORTED_LOCALES: &[&str] = &[
    "en", "zh-CN", "ja", "ko", "ru", "fr", "es", "pt", "it", "de", "la",
];

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocaleOption {
    pub code: String,
    pub name: String,
}

pub fn system_locale() -> String {
    normalize(&sys_locale::get_locale().unwrap_or_else(|| "en".into())).into()
}

pub fn effective_locale(config: &AppConfig) -> String {
    if config.language == "system" {
        system_locale()
    } else {
        normalize(&config.language).into()
    }
}

pub fn normalize(input: &str) -> &'static str {
    let locale = input.replace('_', "-").to_ascii_lowercase();
    if locale.starts_with("zh") {
        "zh-CN"
    } else if locale.starts_with("ja") {
        "ja"
    } else if locale.starts_with("ko") {
        "ko"
    } else if locale.starts_with("ru") {
        "ru"
    } else if locale.starts_with("fr") {
        "fr"
    } else if locale.starts_with("es") {
        "es"
    } else if locale.starts_with("pt") {
        "pt"
    } else if locale.starts_with("it") {
        "it"
    } else if locale.starts_with("de") {
        "de"
    } else if locale.starts_with("la") {
        "la"
    } else {
        "en"
    }
}

pub fn activate(config: &AppConfig) -> String {
    let locale = effective_locale(config);
    rust_i18n::set_locale(&locale);
    locale
}

pub fn translations() -> HashMap<String, String> {
    [
        "title",
        "subtitle",
        "settings",
        "language",
        "system_language",
        "pip_mirror",
        "custom_mirror",
        "save_restart",
        "cancel",
        "starting",
        "phase_preparing",
        "phase_python",
        "phase_venv",
        "phase_marimo",
        "phase_server",
        "phase_ready",
        "phase_stopped",
        "phase_error",
        "retry",
        "config_path",
        "first_run_note",
    ]
    .into_iter()
    .map(|key| {
        (
            key.to_owned(),
            rust_i18n::t!(format!("ui.{key}")).into_owned(),
        )
    })
    .collect()
}

pub fn options() -> Vec<LocaleOption> {
    SUPPORTED_LOCALES
        .iter()
        .map(|code| LocaleOption {
            code: (*code).into(),
            name: rust_i18n::t!(format!("locales.{code}")).into_owned(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_regions_to_supported_locales() {
        assert_eq!(normalize("zh_TW"), "zh-CN");
        assert_eq!(normalize("fr-CA"), "fr");
        assert_eq!(normalize("ko_KR"), "ko");
        assert_eq!(normalize("unknown"), "en");
    }
}
