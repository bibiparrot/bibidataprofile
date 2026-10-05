mod commands;
mod config;
mod error;
mod locale;
mod runtime;

rust_i18n::i18n!("locales", fallback = "en");

use commands::AppState;
use runtime::SharedRuntime;
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
/// Starts the native bibidataprofile application event loop.
///
/// # Panics
///
/// Panics when Tauri cannot build the application from its compile-time
/// configuration, which is a non-recoverable packaging error.
pub fn run() {
    let runtime = SharedRuntime::new();
    let shutdown_runtime = Arc::clone(&runtime);
    let app = tauri::Builder::default()
        .manage(AppState { runtime })
        .invoke_handler(tauri::generate_handler![
            commands::bootstrap,
            commands::save_settings,
            commands::runtime_status,
            commands::start_marimo,
            commands::stop_marimo,
        ])
        .build(tauri::generate_context!())
        .expect("error while building bibidataprofile");

    app.run(move |app_handle, event| {
        if matches!(
            event,
            tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }
        ) {
            shutdown_runtime.stop();
            let _ = app_handle.get_webview_window("main");
        }
    });
}
