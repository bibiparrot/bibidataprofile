param(
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$CargoTargetDirectory = "",
    [string]$Version = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($Version)) {
    $package = Get-Content -LiteralPath (Join-Path $repositoryRoot "package.json") -Raw | ConvertFrom-Json
    $Version = $package.version
}

if ([string]::IsNullOrWhiteSpace($CargoTargetDirectory)) {
    $CargoTargetDirectory = Join-Path $repositoryRoot "src-tauri\target"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repositoryRoot "artifacts"
}

$targetRoot = [System.IO.Path]::GetFullPath($CargoTargetDirectory)
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$targetRelease = Join-Path $targetRoot "$TargetTriple\release"
$hostRelease = Join-Path $targetRoot "release"

$application = Join-Path $targetRelease "bibidataprofile.exe"
if (-not (Test-Path -LiteralPath $application -PathType Leaf)) {
    $application = Join-Path $hostRelease "bibidataprofile.exe"
}

$sidecar = Join-Path $repositoryRoot "src-tauri\binaries\uv-$TargetTriple.exe"
$portableReadme = Join-Path $repositoryRoot "assets\portable-readme.txt"

foreach ($requiredFile in @($application, $sidecar, $portableReadme)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required portable-package file was not found: $requiredFile"
    }
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$stagingRoot = Join-Path $outputRoot ".portable-staging-$PID"
$zipPath = Join-Path $outputRoot "bibidataprofile_${Version}_windows_x64_portable.zip"

try {
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    Copy-Item -LiteralPath $application -Destination (Join-Path $stagingRoot "bibidataprofile.exe")
    Copy-Item -LiteralPath $sidecar -Destination (Join-Path $stagingRoot "uv.exe")
    Copy-Item -LiteralPath $portableReadme -Destination (Join-Path $stagingRoot "README.txt")
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "LICENSE") -Destination (Join-Path $stagingRoot "bibimapy-LICENSE.txt")
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "UPSTREAM.md") -Destination (Join-Path $stagingRoot "bibimapy-UPSTREAM.md")
    Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal -Force
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        $resolvedStage = (Resolve-Path -LiteralPath $stagingRoot).Path
        if (-not $resolvedStage.StartsWith($outputRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean staging path outside the output directory: $resolvedStage"
        }
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
}

$archive = Get-Item -LiteralPath $zipPath
Write-Output "Portable package: $($archive.FullName) ($($archive.Length) bytes)"
