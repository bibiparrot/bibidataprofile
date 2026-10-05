"""Exercise the real marimo UI with example data and capture README screenshots.

Install playwright and its Chromium browser before running this script.
The screenshots show the analysis panel shared by the desktop and Python app.
"""
import argparse
from pathlib import Path
import os
import re
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def capture(url):
    global debug_page
    output = ROOT / "docs/screenshots"
    output.mkdir(parents=True, exist_ok=True)
    reports = ROOT / ".cache/screenshot-reports"
    reports.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        debug_page = page
        page.set_default_timeout(20000)
        def wait_result(label, timeout=60000):
            try:
                page.get_by_text(label, exact=True).wait_for(timeout=timeout)
            except Exception:
                print(page.locator("body").inner_text(), flush=True)
                print(page.get_by_role("button").all_text_contents(), flush=True)
                page.screenshot(path=str(ROOT / ".cache/ui-failure.png"))
                raise
        page.goto(url)
        page.get_by_role("button", name="导入数据", exact=True).wait_for()
        page.locator('input[type="text"]').first.fill(str(reports))
        page.get_by_role("button", name="导入数据", exact=True).click()
        page.get_by_role("heading", name="目标与变量设置").wait_for()
        page.get_by_role("button", name="数据预览", exact=True).click()
        page.get_by_role("table").first.wait_for()
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(output / "overview.png"), animations="disabled")
        print("Captured overview", flush=True)
        page.get_by_role("button", name="数据预览", exact=True).click()
        page.get_by_role("heading", name="目标与变量设置").scroll_into_view_if_needed()
        page.get_by_role("button").filter(has_text=re.compile(r"^amount$")).click()
        print("Selecting target", page.get_by_role("option").all_text_contents(), flush=True)
        page.get_by_role("option", name="value", exact=True).click()
        page.get_by_role("button").filter(has_text=re.compile(r"^amount,visits,segment")).wait_for()
        page.get_by_role("button", name="应用变量设置", exact=True).click()
        page.get_by_text(re.compile("已应用变量设置：value")).wait_for()
        print("Applied feature settings", flush=True)

        page.get_by_role("tab", name="模型构建", exact=True).click()
        expect(page.get_by_role("button", name="训练 LightGBM 模型", exact=True)).to_be_enabled()
        page.get_by_role("button", name="训练 LightGBM 模型", exact=True).click()
        wait_result("下载 model_train.pkl")
        assert (reports / "model_train.pkl").is_file()
        page.get_by_role("tab", name="模型构建", exact=True).evaluate('(el) => el.scrollIntoView({block: "start"})')
        page.screenshot(path=str(output / "model.png"), animations="disabled")

        page.get_by_role("tab", name="变量分箱", exact=True).click()
        page.get_by_role("button", name="生成变量分箱报告", exact=True).click()
        page.get_by_role("tabpanel", name="变量分箱", exact=True).locator('iframe').wait_for(timeout=180000)
        page.get_by_role("tab", name="变量分箱", exact=True).evaluate('(el) => el.scrollIntoView({block: "start"})')
        page.screenshot(path=str(output / "binning.png"), animations="disabled")
        assert list(reports.glob("*.xlsx"))

        page.get_by_role("tab", name="数据画像", exact=True).click()
        page.get_by_role("button", name="生成数据画像", exact=True).click()
        wait_result("下载 data_profile.html", timeout=180000)
        frame = page.get_by_role("tabpanel", name="数据画像", exact=True).frame_locator('iframe')
        frame.get_by_text("Overview", exact=True).first.wait_for(timeout=60000)
        page.get_by_role("tab", name="数据画像", exact=True).evaluate('(el) => el.scrollIntoView({block: "start"})')
        page.screenshot(path=str(output / "profile.png"), animations="disabled")

        page.get_by_role("tab", name="模型预测", exact=True).click()
        page.get_by_placeholder(".../model_train.pkl").fill(str(reports / "model_train.pkl"))
        page.get_by_role("button", name="预测并导出", exact=True).click()
        wait_result("下载 prediction_result.csv")
        assert (reports / "prediction_result.csv").is_file()
        page.get_by_role("tab", name="模型预测", exact=True).evaluate('(el) => el.scrollIntoView({block: "start"})')
        page.screenshot(path=str(output / "prediction.png"), animations="disabled")

        page.get_by_role("tab", name="数据导出", exact=True).click()
        page.get_by_role("button", name="导出当前数据", exact=True).click()
        wait_result("下载 dataset.csv")
        assert (reports / "dataset.csv").is_file()
        browser.close()
        print("UI smoke passed: import, training, binning, profiling, prediction and export")
        print(f"Captured five real application screenshots in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Use an already running app instead of starting a test server")
    args = parser.parse_args()
    if args.url:
        capture(args.url)
    else:
        log_path = ROOT / ".cache/screenshot-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "marimo", "run", str(ROOT / "src/bibidataprofile/marimo_app.py"),
                                        "--host", "127.0.0.1", "--port", "2741", "--headless", "--no-token"],
                                       env=environment, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text("utf-8"))
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:2741", timeout=1):
                            break
                    except OSError:
                        time.sleep(0.5)
                else:
                    raise RuntimeError("marimo server did not become ready")
                capture("http://127.0.0.1:2741")
            finally:
                process.terminate()
                process.wait(timeout=15)
