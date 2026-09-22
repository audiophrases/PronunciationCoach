$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$toolDirectory = Join-Path $projectRoot '.cache/mfa-tools'
$manager = Join-Path $toolDirectory 'Library/bin/micromamba.exe'
$environment = Join-Path $projectRoot '.cache/mfa-env'
$env:MAMBA_ROOT_PREFIX = Join-Path $projectRoot '.cache/mamba'
$env:MFA_ROOT_DIR = Join-Path $projectRoot '.cache/mfa-models'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath $manager)) {
    New-Item -ItemType Directory -Force $toolDirectory | Out-Null
    $archive = Join-Path $toolDirectory 'micromamba.tar.bz2'
    Invoke-WebRequest -Uri 'https://conda.anaconda.org/conda-forge/win-64/micromamba-2.9.0-0.tar.bz2' -OutFile $archive
    $expected = '97A336F4AB794BD96A6A4DA5E6ED63E75A1D31830414A182419B23D3B36F3FE0'
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
        throw 'Micromamba archive checksum mismatch.'
    }
    & tar -xf $archive -C $toolDirectory Library/bin/micromamba.exe
    if ($LASTEXITCODE -ne 0) { throw 'Micromamba extraction failed.' }
}
if (-not (Test-Path -LiteralPath (Join-Path $environment 'conda-meta/history'))) {
    & $manager create -y -p $environment -f (Join-Path $PSScriptRoot 'mfa-win-64.lock')
    if ($LASTEXITCODE -ne 0) { throw 'MFA environment installation failed.' }
}
& $manager run -p $environment mfa version
if ($LASTEXITCODE -ne 0) { throw 'MFA could not start.' }
function Get-PinnedModel($kind, $name, $extension, $checksum) {
    $directory = Join-Path $env:MFA_ROOT_DIR "pretrained_models/$kind"
    New-Item -ItemType Directory -Force $directory | Out-Null
    $destination = Join-Path $directory "$name.$extension"
    if ((Test-Path -LiteralPath $destination) -and
        (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -eq $checksum) { return }
    $url = "https://github.com/MontrealCorpusTools/mfa-models/releases/download/$kind-$name-v3.1.0/$name.$extension"
    $download = "$destination.download"
    Invoke-WebRequest -Uri $url -OutFile $download
    if ((Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash -ne $checksum) {
        throw "Model checksum mismatch: $name"
    }
    Move-Item -LiteralPath $download -Destination $destination -Force
}
Get-PinnedModel 'acoustic' 'english_mfa' 'zip' '2c08bd4f82c3943dd57ac09aeac99dbce17a1e1bfe9fd932c8d95cd13d971068'
Get-PinnedModel 'dictionary' 'english_us_mfa' 'dict' 'cbf4ed8cee1338e009b9b387bcb9ee789ce70b98c61ca9bc6ed4f4e63d4812d5'
Write-Host 'MFA is ready. Run launchers/compare_mfa.bat to review your saved recordings.'
