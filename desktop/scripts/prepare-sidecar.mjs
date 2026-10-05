import { execFileSync } from "node:child_process";
import { copyFileSync, chmodSync, mkdirSync } from "node:fs";
import { join } from "node:path";

function output(command, args) {
  return execFileSync(command, args, { encoding: "utf8" }).trim();
}

const rustInfo = output("rustc", ["-vV"]);
const host = process.env.TAURI_TARGET ?? rustInfo.match(/^host: (.+)$/m)?.[1];
if (!host) throw new Error("Could not determine the Rust target triple");

const isWindows = process.platform === "win32";
const uvOverride = process.env.BIBIDATAPROFILE_UV;
const uvPath = uvOverride || output(isWindows ? "where.exe" : "which", [isWindows ? "uv.exe" : "uv"]).split(/\r?\n/)[0];
const extension = host.includes("windows") ? ".exe" : "";
const destinationDir = join(process.cwd(), "src-tauri", "binaries");
const destination = join(destinationDir, `uv-${host}${extension}`);

mkdirSync(destinationDir, { recursive: true });
copyFileSync(uvPath, destination);
if (!isWindows) chmodSync(destination, 0o755);
console.log(`Prepared uv sidecar: ${destination}`);
