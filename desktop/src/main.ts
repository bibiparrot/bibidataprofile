import { invoke } from "@tauri-apps/api/core";
import appIcon from "../assets/bibi-icon.png";
import "./styles.css";

type Config = {
  language: string;
  python_version: string;
  pip_index_url: string;
  marimo_package: string;
  marimo_port: number;
  startup_timeout_seconds: number;
};

type LocaleOption = { code: string; name: string };
type Bootstrap = {
  config: Config;
  configPath: string;
  effectiveLocale: string;
  translations: Record<string, string>;
  locales: LocaleOption[];
};
type RuntimeInfo = {
  phase: string;
  detail: string | null;
  url: string | null;
};

const mirrors = [
  { name: "PyPI", url: "https://pypi.org/simple" },
  { name: "Aliyun / 阿里云", url: "https://mirrors.aliyun.com/pypi/simple/" },
  { name: "Tsinghua / 清华", url: "https://pypi.tuna.tsinghua.edu.cn/simple" },
];

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div class="shell">
    <header class="topbar">
      <div class="brand" aria-label="bibidataprofile">
        <img class="brand-icon" src="${appIcon}" alt="" />
        <span id="title">bibidataprofile</span>
      </div>
      <div class="top-actions">
        <span id="status-pill" class="status-pill"><span></span><b></b></span>
        <button id="settings-button" class="icon-button" type="button" aria-label="Settings" title="Settings">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"/><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.83 2.83-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V21h-4v-.08A1.7 1.7 0 0 0 8.94 19.4a1.7 1.7 0 0 0-1.88.34l-.06.06-2.83-2.83.06-.06A1.7 1.7 0 0 0 4.57 15 1.7 1.7 0 0 0 3 14H3v-4h.08A1.7 1.7 0 0 0 4.6 8.94a1.7 1.7 0 0 0-.34-1.88L4.2 7l2.83-2.83.06.06A1.7 1.7 0 0 0 9 4.57 1.7 1.7 0 0 0 10 3V3h4v.08A1.7 1.7 0 0 0 15.06 4.6a1.7 1.7 0 0 0 1.88-.34L17 4.2 19.83 7l-.06.06A1.7 1.7 0 0 0 19.43 9 1.7 1.7 0 0 0 21 10h.08v4H21a1.7 1.7 0 0 0-1.6 1Z"/></svg>
        </button>
      </div>
    </header>

    <main class="workspace">
      <section id="loading" class="loading-card">
        <div class="orb"><span></span></div>
        <p id="eyebrow" class="eyebrow"></p>
        <h1 id="loading-title"></h1>
        <p id="phase-message" class="phase-message"></p>
        <div id="install-progress" class="install-progress" role="progressbar" aria-valuemin="0" aria-valuemax="5">
          <span class="progress-spinner" aria-hidden="true"></span>
          <span class="progress-track" aria-hidden="true"><span id="progress-fill"></span></span>
        </div>
        <p id="first-run" class="first-run"></p>
        <button id="retry-button" class="primary hidden" type="button"></button>
      </section>
      <iframe id="marimo" title="marimo" class="hidden" allow="clipboard-read; clipboard-write"></iframe>
    </main>
  </div>

  <dialog id="settings-dialog">
    <form id="settings-form" method="dialog">
      <div class="dialog-heading">
        <div><p>bibidataprofile</p><h2 id="settings-title"></h2></div>
        <button id="close-settings" class="close-button" type="button" aria-label="Close">×</button>
      </div>
      <label>
        <span id="language-label"></span>
        <select id="language-select"></select>
      </label>
      <label>
        <span id="mirror-label"></span>
        <select id="mirror-select"></select>
      </label>
      <label id="custom-mirror-row" class="hidden">
        <span id="custom-mirror-label"></span>
        <input id="custom-mirror" type="url" required spellcheck="false" />
      </label>
      <p class="config-location"><span id="config-label"></span><code id="config-path"></code></p>
      <div class="dialog-actions">
        <button id="cancel-settings" class="secondary" type="button"></button>
        <button id="save-settings" class="primary" type="submit"></button>
      </div>
    </form>
  </dialog>
`;

const elements = {
  title: document.querySelector<HTMLElement>("#title")!,
  status: document.querySelector<HTMLElement>("#status-pill")!,
  statusText: document.querySelector<HTMLElement>("#status-pill b")!,
  loading: document.querySelector<HTMLElement>("#loading")!,
  eyebrow: document.querySelector<HTMLElement>("#eyebrow")!,
  loadingTitle: document.querySelector<HTMLElement>("#loading-title")!,
  phase: document.querySelector<HTMLElement>("#phase-message")!,
  progress: document.querySelector<HTMLElement>("#install-progress")!,
  progressFill: document.querySelector<HTMLElement>("#progress-fill")!,
  firstRun: document.querySelector<HTMLElement>("#first-run")!,
  retry: document.querySelector<HTMLButtonElement>("#retry-button")!,
  frame: document.querySelector<HTMLIFrameElement>("#marimo")!,
  settingsButton: document.querySelector<HTMLButtonElement>("#settings-button")!,
  dialog: document.querySelector<HTMLDialogElement>("#settings-dialog")!,
  closeSettings: document.querySelector<HTMLButtonElement>("#close-settings")!,
  cancelSettings: document.querySelector<HTMLButtonElement>("#cancel-settings")!,
  form: document.querySelector<HTMLFormElement>("#settings-form")!,
  settingsTitle: document.querySelector<HTMLElement>("#settings-title")!,
  languageLabel: document.querySelector<HTMLElement>("#language-label")!,
  mirrorLabel: document.querySelector<HTMLElement>("#mirror-label")!,
  customMirrorLabel: document.querySelector<HTMLElement>("#custom-mirror-label")!,
  language: document.querySelector<HTMLSelectElement>("#language-select")!,
  mirror: document.querySelector<HTMLSelectElement>("#mirror-select")!,
  customMirrorRow: document.querySelector<HTMLElement>("#custom-mirror-row")!,
  customMirror: document.querySelector<HTMLInputElement>("#custom-mirror")!,
  configLabel: document.querySelector<HTMLElement>("#config-label")!,
  configPath: document.querySelector<HTMLElement>("#config-path")!,
  save: document.querySelector<HTMLButtonElement>("#save-settings")!,
};

let bootstrap: Bootstrap;
let pollTimer: number | undefined;

const phaseSteps: Record<string, number> = {
  preparing: 0,
  python: 1,
  venv: 2,
  marimo: 3,
  server: 4,
  ready: 5,
};

function tr(key: string): string {
  return bootstrap?.translations[key] ?? key;
}

function setLocalizedContent() {
  document.documentElement.lang = bootstrap.effectiveLocale;
  elements.title.textContent = tr("title");
  elements.eyebrow.textContent = tr("subtitle");
  elements.loadingTitle.textContent = tr("starting");
  elements.firstRun.textContent = tr("first_run_note");
  elements.retry.textContent = tr("retry");
  elements.settingsButton.ariaLabel = tr("settings");
  elements.settingsButton.title = tr("settings");
  elements.settingsTitle.textContent = tr("settings");
  elements.languageLabel.textContent = tr("language");
  elements.mirrorLabel.textContent = tr("pip_mirror");
  elements.customMirrorLabel.textContent = tr("custom_mirror");
  elements.configLabel.textContent = `${tr("config_path")}:`;
  elements.configPath.textContent = bootstrap.configPath;
  elements.cancelSettings.textContent = tr("cancel");
  elements.save.textContent = tr("save_restart");
  renderSettingsValues();
}

function appendOption(select: HTMLSelectElement, value: string, label: string) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

function renderSettingsValues() {
  elements.language.replaceChildren();
  appendOption(elements.language, "system", tr("system_language"));
  bootstrap.locales.forEach((locale) => appendOption(elements.language, locale.code, locale.name));
  elements.language.value = bootstrap.config.language;

  elements.mirror.replaceChildren();
  mirrors.forEach((mirror) => appendOption(elements.mirror, mirror.url, mirror.name));
  appendOption(elements.mirror, "custom", tr("custom_mirror"));
  const known = mirrors.some((mirror) => mirror.url === bootstrap.config.pip_index_url);
  elements.mirror.value = known ? bootstrap.config.pip_index_url : "custom";
  elements.customMirror.value = bootstrap.config.pip_index_url;
  toggleCustomMirror();
}

function toggleCustomMirror() {
  elements.customMirrorRow.classList.toggle("hidden", elements.mirror.value !== "custom");
  elements.customMirror.disabled = elements.mirror.value !== "custom";
}

function renderStatus(info: RuntimeInfo) {
  const phaseMessage = tr(`phase_${info.phase}`);
  const message = info.phase === "error" && info.detail ? info.detail : phaseMessage;
  const step = phaseSteps[info.phase] ?? 0;
  const busy = info.phase !== "error" && info.phase !== "stopped";
  elements.status.dataset.phase = info.phase;
  elements.statusText.textContent = tr(`phase_${info.phase}`);
  elements.phase.textContent = message;
  elements.loading.setAttribute("aria-busy", String(busy));
  elements.progress.classList.toggle("hidden", !busy);
  elements.progress.setAttribute("aria-valuenow", String(step));
  elements.progress.setAttribute("aria-valuetext", message);
  elements.progressFill.style.width = `${Math.max(8, step * 20)}%`;
  if (info.phase === "error") {
    elements.loadingTitle.textContent = tr("phase_error");
    elements.retry.classList.remove("hidden");
  }
}

async function refreshStatus() {
  const info = await invoke<RuntimeInfo>("runtime_status");
  renderStatus(info);
}

async function start() {
  window.clearInterval(pollTimer);
  elements.frame.classList.add("hidden");
  elements.loading.classList.remove("hidden", "error");
  elements.retry.classList.add("hidden");
  elements.loadingTitle.textContent = tr("starting");
  renderStatus({ phase: "preparing", detail: null, url: null });
  pollTimer = window.setInterval(() => void refreshStatus(), 350);
  try {
    const info = await invoke<RuntimeInfo>("start_marimo");
    renderStatus(info);
    if (!info.url) throw new Error("marimo returned no URL");
    elements.frame.addEventListener(
      "load",
      () => {
        elements.loading.setAttribute("aria-busy", "false");
        elements.progress.classList.add("hidden");
        elements.loading.classList.add("hidden");
        elements.frame.classList.remove("hidden");
      },
      { once: true },
    );
    elements.frame.src = info.url;
  } catch (error) {
    elements.loading.classList.add("error");
    renderStatus({ phase: "error", detail: String(error), url: null });
  } finally {
    window.clearInterval(pollTimer);
  }
}

elements.settingsButton.addEventListener("click", () => {
  renderSettingsValues();
  elements.dialog.showModal();
});
elements.closeSettings.addEventListener("click", () => elements.dialog.close());
elements.cancelSettings.addEventListener("click", () => elements.dialog.close());
elements.mirror.addEventListener("change", toggleCustomMirror);
elements.retry.addEventListener("click", () => void start());
elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!elements.form.reportValidity()) return;
  elements.save.disabled = true;
  try {
    const pipIndexUrl = elements.mirror.value === "custom" ? elements.customMirror.value : elements.mirror.value;
    await invoke("stop_marimo");
    bootstrap = await invoke<Bootstrap>("save_settings", {
      config: {
        ...bootstrap.config,
        language: elements.language.value,
        pip_index_url: pipIndexUrl,
      },
    });
    setLocalizedContent();
    elements.dialog.close();
    await start();
  } finally {
    elements.save.disabled = false;
  }
});

async function initialize() {
  try {
    bootstrap = await invoke<Bootstrap>("bootstrap");
    setLocalizedContent();
    await start();
  } catch (error) {
    elements.loading.classList.add("error");
    elements.loading.setAttribute("aria-busy", "false");
    elements.progress.classList.add("hidden");
    elements.phase.textContent = String(error);
    elements.retry.classList.remove("hidden");
  }
}

void initialize();
