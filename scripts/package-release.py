"""Package native bundles under architecture-specific public release names."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(target, output):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    bundle = ROOT / "desktop/src-tauri/target" / target / "release/bundle"
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"bibidataprofile-{version}"

    def copy_one(pattern, name):
        matches = list(bundle.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {pattern}, found {matches}")
        shutil.copy2(matches[0], output / name)

    if "windows" in target:
        prefix += "-windows-x86_64"
        copy_one("nsis/*.exe", prefix + "_setup.exe")
        copy_one("msi/*.msi", prefix + "_setup.msi")
        portable = ROOT / f"desktop/artifacts/bibidataprofile_{version}_windows_x64_portable.zip"
        with zipfile.ZipFile(portable) as archive:
            for required in ("bibidataprofile.exe", "uv.exe", "README.txt", "bibimapy-LICENSE.txt"):
                if required not in archive.namelist():
                    raise RuntimeError(f"Missing portable runtime file: {required}")
        shutil.copy2(portable, output / (prefix + "_portable.zip"))
    elif "apple" in target:
        arch = "arm64" if target.startswith("aarch64") else "x86_64"
        prefix += f"-{arch}-macOS"
        copy_one("dmg/*.dmg", prefix + ".dmg")
        apps = list(bundle.glob("macos/*.app"))
        if len(apps) != 1:
            raise RuntimeError(f"Expected one macOS application, found {apps}")
        app = apps[0]
        for relative in ("Contents/MacOS/bibidataprofile", "Contents/MacOS/uv", "Contents/Resources/runtime/libomp.dylib"):
            if not (app / relative).is_file():
                raise RuntimeError(f"Missing macOS runtime file: {relative}")
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app), str(output / (prefix + ".zip"))], check=True)
        subprocess.run(["pkgbuild", "--component", str(app), "--install-location", "/Applications", "--identifier", "dev.bibiparrot.bibidataprofile", "--version", version, str(output / (prefix + ".pkg"))], check=True)
        payload = subprocess.check_output(["pkgutil", "--payload-files", str(output / (prefix + ".pkg"))], text=True)
        if "Contents/MacOS/bibidataprofile" not in payload:
            raise RuntimeError("macOS PKG has no application executable")
    elif "linux" in target:
        arch = "aarch64" if target.startswith("aarch64") else "x86_64"
        prefix += f"-linux-{arch}"
        copy_one("appimage/*.AppImage", prefix + ".AppImage")
        copy_one("rpm/*.rpm", prefix + ".rpm")
        appdirs = list(bundle.glob("appimage/*.AppDir"))
        if len(appdirs) != 1 or not (appdirs[0] / "AppRun").is_file():
            raise RuntimeError(f"Expected a runnable AppDir, found {appdirs}")
        # Preserve the full AppDir, including uv, GTK/WebKit libraries and resources.
        with tarfile.open(output / (prefix + ".tar.gz"), "w:gz") as archive:
            archive.add(appdirs[0], arcname=prefix)
        with tarfile.open(output / (prefix + ".tar.gz")) as archive:
            if f"{prefix}/AppRun" not in archive.getnames():
                raise RuntimeError("Linux portable archive has no AppRun")
    else:
        raise ValueError(f"Unsupported release target: {target}")
    for path in sorted(output.glob(prefix + "*")):
        print(f"Packaged {path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "desktop/artifacts/release")
    args = parser.parse_args()
    package(args.target, args.output.resolve())
