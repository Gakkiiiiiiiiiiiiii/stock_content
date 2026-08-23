param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$TaskId,

    [string]$BaseBranch = "main",

    [string]$WorktreeRoot = "D:\project\worktrees"
)

$repo = (git rev-parse --show-toplevel 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($repo)) {
    throw "Run this script from inside a Git repository."
}

$repo = [System.IO.Path]::GetFullPath($repo)
$repoName = Split-Path -Leaf $repo
$worktree = [System.IO.Path]::GetFullPath((Join-Path $WorktreeRoot "$repoName-$TaskId"))
$branch = "codex/$TaskId"

if (Test-Path -LiteralPath $worktree) {
    throw "Worktree path already exists: $worktree"
}

git show-ref --verify --quiet "refs/heads/$branch"
if ($LASTEXITCODE -eq 0) {
    throw "Branch already exists: $branch"
}

New-Item -ItemType Directory -Force -Path $WorktreeRoot | Out-Null

Write-Host "Repository : $repo"
Write-Host "Base       : $BaseBranch"
Write-Host "Worktree   : $worktree"
Write-Host "Branch     : $branch"

git status --short --branch
if ($LASTEXITCODE -ne 0) {
    throw "git status failed"
}

git worktree add -b $branch $worktree $BaseBranch
if ($LASTEXITCODE -ne 0) {
    throw "git worktree add failed"
}

Set-Location -LiteralPath $worktree

$codexExecutable = "codex"
$pathCodex = Get-Command codex -ErrorAction SilentlyContinue
if ($null -ne $pathCodex) {
    $pathVersion = (& $pathCodex.Source --version 2>$null | Out-String).Trim()
    if ($pathVersion -match "0\.118\.0") {
        $appBin = Join-Path $env:LOCALAPPDATA "OpenAI\Codex\bin"
        $newerCandidates = Get-ChildItem -LiteralPath $appBin -Filter "codex.exe" -File -Recurse -ErrorAction SilentlyContinue
        foreach ($candidate in $newerCandidates) {
            $candidateVersion = (& $candidate.FullName --version 2>$null | Out-String).Trim()
            if ($candidateVersion -and $candidateVersion -notmatch "0\.118\.0") {
                $codexExecutable = $candidate.FullName
                break
            }
        }
    }
}

Write-Host ""
Write-Host "Worktree created. Run the baseline before implementation:"
Write-Host "  python -m pytest -q"
Write-Host "  python -m ruff check src tests scripts"
Write-Host ""
Write-Host "Then launch the interactive root supervisor:"
Write-Host "  & `"$codexExecutable`" --model gpt-5.6-sol"
Write-Host ""
Write-Host "Paste .codex/prompts/SOL_SUPERVISOR.md followed by the feature request."
