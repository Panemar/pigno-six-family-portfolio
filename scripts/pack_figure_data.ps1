param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$source = Join-Path $root 'tables\figure_data\F27.csv'
$compressed = "$source.gz"
$manifestDirectory = Join-Path $root 'data\external'
New-Item -ItemType Directory -Path $manifestDirectory -Force | Out-Null

$resolvedSource = (Resolve-Path -LiteralPath $source).Path
if (-not $resolvedSource.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'F27 target resolved outside the repository root.'
}

$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceBytes = (Get-Item -LiteralPath $source).Length

$inputStream = [System.IO.File]::OpenRead($source)
$outputStream = [System.IO.File]::Create($compressed)
try {
    $gzip = [System.IO.Compression.GZipStream]::new(
        $outputStream,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )
    try { $inputStream.CopyTo($gzip) } finally { $gzip.Dispose() }
} finally {
    $inputStream.Dispose()
    $outputStream.Dispose()
}

$sha = [System.Security.Cryptography.SHA256]::Create()
$compressedInput = [System.IO.File]::OpenRead($compressed)
try {
    $decompressor = [System.IO.Compression.GZipStream]::new(
        $compressedInput,
        [System.IO.Compression.CompressionMode]::Decompress
    )
    try {
        $roundTripHash = [System.BitConverter]::ToString($sha.ComputeHash($decompressor)).Replace('-', '').ToLowerInvariant()
    } finally { $decompressor.Dispose() }
} finally {
    $compressedInput.Dispose()
    $sha.Dispose()
}

if ($roundTripHash -ne $sourceHash) {
    throw "Round-trip SHA-256 mismatch: $roundTripHash != $sourceHash"
}

$record = [ordered]@{
    source = 'tables/figure_data/F27.csv'
    source_sha256 = $sourceHash
    source_bytes = $sourceBytes
    compressed = 'tables/figure_data/F27.csv.gz'
    compressed_sha256 = (Get-FileHash -LiteralPath $compressed -Algorithm SHA256).Hash.ToLowerInvariant()
    compressed_bytes = (Get-Item -LiteralPath $compressed).Length
    roundtrip_sha256 = $roundTripHash
    verified = $true
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $manifestDirectory 'COMPRESSED_DATA_MANIFEST.json') -Encoding utf8

# The source campaign remains untouched; only the verified duplicate in this release is removed.
Remove-Item -LiteralPath $resolvedSource -Force
$record | ConvertTo-Json
