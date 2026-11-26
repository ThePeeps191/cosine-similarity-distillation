<#
.SYNOPSIS
    Compile the CSD paper LaTeX source to PDF.
    Runs: pdflatex -> bibtex -> pdflatex -> pdflatex
    Reports page count and word count.

.USAGE
    .\compile_paper.ps1
#>

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $paperDir) { $paperDir = "." }
$texFile = Join-Path $paperDir "template.tex"
$pdfFile = Join-Path $paperDir "template.pdf"
$logFile = Join-Path $paperDir "template.log"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CSD Paper Compiler" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $texFile)) {
    Write-Host "ERROR: template.tex not found in $paperDir" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------
# 1. First pdflatex pass
# ------------------------------------------------------------------
Write-Host "[1/4] Running pdflatex (pass 1)..." -ForegroundColor Yellow
$result = Start-Process -FilePath "pdflatex" -ArgumentList "-interaction=nonstopmode", "template.tex" -WorkingDirectory $paperDir -NoNewWindow -Wait -PassThru
if ($result.ExitCode -ne 0 -and $result.ExitCode -ne 1) {
    Write-Host "  pdflatex pass 1 failed with exit code $($result.ExitCode)" -ForegroundColor Red
    exit $result.ExitCode
}
Write-Host "  OK" -ForegroundColor Green

# ------------------------------------------------------------------
# 2. Bibtex pass
# ------------------------------------------------------------------
Write-Host "[2/4] Running bibtex..." -ForegroundColor Yellow
$result = Start-Process -FilePath "bibtex" -ArgumentList "template" -WorkingDirectory $paperDir -NoNewWindow -Wait -PassThru
if ($result.ExitCode -ne 0) {
    Write-Host "  bibtex failed with exit code $($result.ExitCode)" -ForegroundColor Red
    exit $result.ExitCode
}
Write-Host "  OK" -ForegroundColor Green

# ------------------------------------------------------------------
# 3. Second pdflatex pass (incorporates bibliography)
# ------------------------------------------------------------------
Write-Host "[3/4] Running pdflatex (pass 2)..." -ForegroundColor Yellow
$result = Start-Process -FilePath "pdflatex" -ArgumentList "-interaction=nonstopmode", "template.tex" -WorkingDirectory $paperDir -NoNewWindow -Wait -PassThru
if ($result.ExitCode -ne 0 -and $result.ExitCode -ne 1) {
    Write-Host "  pdflatex pass 2 failed with exit code $($result.ExitCode)" -ForegroundColor Red
    exit $result.ExitCode
}
Write-Host "  OK" -ForegroundColor Green

# ------------------------------------------------------------------
# 4. Third pdflatex pass (resolves all cross-references)
# ------------------------------------------------------------------
Write-Host "[4/4] Running pdflatex (pass 3)..." -ForegroundColor Yellow
$result = Start-Process -FilePath "pdflatex" -ArgumentList "-interaction=nonstopmode", "template.tex" -WorkingDirectory $paperDir -NoNewWindow -Wait -PassThru
if ($result.ExitCode -ne 0 -and $result.ExitCode -ne 1) {
    Write-Host "  pdflatex pass 3 failed with exit code $($result.ExitCode)" -ForegroundColor Red
    exit $result.ExitCode
}
Write-Host "  OK" -ForegroundColor Green

Write-Host ""

# ------------------------------------------------------------------
# 5. Verify PDF was created
# ------------------------------------------------------------------
if (-not (Test-Path $pdfFile)) {
    Write-Host "ERROR: PDF was not generated" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------------
# 6. Check for warnings / unresolved references
# ------------------------------------------------------------------
if (Test-Path $logFile) {
    $log = Get-Content $logFile -Raw

    if ($log -match "Rerun to get") {
        Write-Host "WARNING: Some cross-references may not be resolved. Re-run the script." -ForegroundColor Yellow
    }

    if ($log -match "Warning.*Citation.*undefined") {
        Write-Host "WARNING: Some citations may be undefined. Check references.bib." -ForegroundColor Yellow
    }

    # Extract page count
    if ($log -match "Output written on template.pdf \((\d+) page") {
        $pages = $matches[1]
    } else {
        $pages = "?"
    }
}

# ------------------------------------------------------------------
# 7. Word count (approximate, from LaTeX source, stripping commands)
# ------------------------------------------------------------------
$texContent = Get-Content $texFile -Raw

# Remove LaTeX commands and environments
$clean = $texContent -replace '\\[a-zA-Z]+(\{[^}]*\}|\[[^\]]*\])*', ' '
$clean = $clean -replace '\\[^a-zA-Z]', ' '
$clean = $clean -replace '\{[^}]*\}', ' '
$clean = $clean -replace '\[[^\]]*\]', ' '
$clean = $clean -replace '^[^a-zA-Z]*$', ' '
$clean = $clean -replace '\s+', ' '
$wordCount = ($clean -split '\s+' | Where-Object { $_ -match '[a-zA-Z]' }).Count

# ------------------------------------------------------------------
# 8. Report
# ------------------------------------------------------------------
$pdfSizeInKB = [math]::Round((Get-Item $pdfFile).Length / 1KB, 1)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Compilation complete" -ForegroundColor Green
Write-Host "  PDF:     template.pdf ($pdfSizeInKB KB, $pages pages)" -ForegroundColor White
Write-Host "  Words:   ~$wordCount (body text, excluding commands)" -ForegroundColor White
Write-Host "  Output:  $pdfFile" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

# Open PDF
Start-Process $pdfFile
