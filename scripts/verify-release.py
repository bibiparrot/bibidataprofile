"""Reject incomplete releases and write checksums for every deliverable."""
import hashlib
from pathlib import Path
import sys
import tomllib

root = Path(__file__).resolve().parents[1]
version = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]["version"]
folder = Path(sys.argv[1])
suffixes = [f"-{arch}-macOS.{ext}" for arch in ("arm64", "x86_64") for ext in ("dmg", "pkg", "zip")]
suffixes += [f"-linux-{arch}.{ext}" for arch in ("aarch64", "x86_64") for ext in ("AppImage", "rpm", "tar.gz")]
suffixes += ["-windows-x86_64_portable.zip", "-windows-x86_64_setup.exe", "-windows-x86_64_setup.msi"]
expected = [f"bibidataprofile-{version}{suffix}" for suffix in suffixes]
expected += [f"bibidataprofile-{version}-py3-none-any.whl"]
missing = [name for name in expected if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
if missing:
    raise RuntimeError(f"Release files missing or empty: {missing}")
lines = []
for name in sorted(expected):
    with (folder / name).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    lines.append(f"{digest}  {name}")
(folder / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Verified all 15 native packages, wheel and checksums for {version}")
