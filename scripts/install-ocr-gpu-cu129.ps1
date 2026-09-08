<# Install the isolated OCR venv; do not run this inside the ASR/Torch venv. #>
param(
    [Parameter(Mandatory = $true)] [string] $Python,
    [string] $Venv = ".venv-ocr-cu129"
)
$ErrorActionPreference = "Stop"
& $Python -m venv $Venv
$ocrPython = Join-Path $Venv "Scripts\python.exe"
& $ocrPython -m pip install --upgrade pip
# Official Paddle cu129 artifact: compiled CUDA 12.9 and cuDNN 9.9.  It is
# intentionally not put in pyproject because normal PyPI must not choose a
# CUDA wheel. Update only with its SHA-verified release artifact.
& $ocrPython -m pip install --no-deps "https://www.paddlepaddle.org.cn/packages/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.3.0-cp311-cp311-win_amd64.whl"
& $ocrPython -m pip install -r requirements/ocr-gpu-cu129.lock
& $ocrPython -m pip install --no-deps .
& $ocrPython -m stock_content.cli.ocr_doctor
