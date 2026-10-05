import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(desktop, "..");
const resources = join(desktop, "src-tauri", "resources");
const version = JSON.parse(readFileSync(join(desktop, "package.json"), "utf8")).version;
const pythonVersion = readFileSync(join(root, "pyproject.toml"), "utf8").match(/^version = "([^"]+)"/m)?.[1];
if (version !== pythonVersion) throw new Error("Python and desktop versions must match");
mkdirSync(resources, { recursive: true });
execFileSync(process.env.BIBIDATAPROFILE_UV || "uv", ["build", "--wheel", "--out-dir", resources, root], {
  stdio: "inherit", cwd: root, env: {
    ...process.env,
    UV_CACHE_DIR: join(root, ".cache", "uv"),
    UV_PYTHON_INSTALL_DIR: join(root, ".cache", "python"),
  },
});
const wheelName = `bibidataprofile-${version}-py3-none-any.whl`;
const digest = createHash("sha256").update(readFileSync(join(resources, wheelName))).digest("hex");
writeFileSync(join(resources, "app-wheel.rs"),
  `const APP_WHEEL_NAME: &str = "${wheelName}";\nconst APP_WHEEL_DIGEST: &str = "${digest}";\nconst APP_WHEEL: &[u8] = include_bytes!("${wheelName}");\n`);
console.log(`Embedded application prepared: ${wheelName}`);
