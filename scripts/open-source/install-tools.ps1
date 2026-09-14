# Install only under ignored target/open-source-tools; no global PATH changes.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$toolRoot = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path 'target/open-source-tools'
if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne [Runtime.InteropServices.Architecture]::X64) {
    throw 'This bootstrap currently supports x64 Windows and Linux only.'
}
if ($IsWindows) {
    $downloads = @(
        @{repo='rustsec/rustsec'; tag='cargo-audit/v0.22.2'; asset='cargo-audit-x86_64-pc-windows-msvc-v0.22.2.zip'; folder='cargo-audit'; sha='0a7316540862c13d954f648917ceacca593747baed6eec180fafa590be2710ab'},
        @{repo='gitleaks/gitleaks'; tag='v8.30.1'; asset='gitleaks_8.30.1_windows_x64.zip'; folder='gitleaks'; sha='d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e'},
        @{repo='EmbarkStudios/cargo-deny'; tag='0.20.2'; asset='cargo-deny-0.20.2-x86_64-pc-windows-msvc.tar.gz'; folder='cargo-deny'; sha='975a22143262fd27476d19ee00c7af67978426e40e1dee94eed6bbade1cf87dc'},
        @{repo='CycloneDX/cyclonedx-rust-cargo'; tag='cargo-cyclonedx-0.5.9'; asset='cargo-cyclonedx-x86_64-pc-windows-msvc.zip'; folder='cargo-cyclonedx'; sha='8750e00775661dcb75bc482c1a298839fd94e8a0c033b49905ba0f246ffed202'}
    )
} elseif ($IsLinux) {
    $downloads = @(
        @{repo='rustsec/rustsec'; tag='cargo-audit/v0.22.2'; asset='cargo-audit-x86_64-unknown-linux-musl-v0.22.2.tgz'; folder='cargo-audit'; sha='7fb9497f8594b389e5fce5ef9b92db08432996895b2e0c5a0167a69ed445c428'},
        @{repo='gitleaks/gitleaks'; tag='v8.30.1'; asset='gitleaks_8.30.1_linux_x64.tar.gz'; folder='gitleaks'; sha='551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb'},
        @{repo='EmbarkStudios/cargo-deny'; tag='0.20.2'; asset='cargo-deny-0.20.2-x86_64-unknown-linux-musl.tar.gz'; folder='cargo-deny'; sha='9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f'},
        @{repo='CycloneDX/cyclonedx-rust-cargo'; tag='cargo-cyclonedx-0.5.9'; asset='cargo-cyclonedx-x86_64-unknown-linux-musl.tar.xz'; folder='cargo-cyclonedx'; sha='9bd3e599314f50810c9d98b8b68a617ff9d3cc20873968d90b29d121f6b226ff'}
    )
} else { throw 'Unsupported OS. Install the pinned tool versions manually.' }
New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null
foreach ($download in $downloads) {
    $archive = Join-Path $toolRoot $download.asset
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri "https://github.com/$($download.repo)/releases/download/$($download.tag)/$($download.asset)" -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $download.sha) {
        throw "Tool archive checksum mismatch: $($download.asset)"
    }
    $destination = Join-Path $toolRoot $download.folder
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    if ($archive.EndsWith('.zip')) {
        Expand-Archive -LiteralPath $archive -DestinationPath $destination -Force
    } else {
        & tar -xf $archive -C $destination
        if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $($download.asset)" }
    }
    Write-Host "Verified and installed locally: $($download.folder) $($download.tag)"
}
$downloads | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $toolRoot 'tool-downloads.json') -Encoding utf8
