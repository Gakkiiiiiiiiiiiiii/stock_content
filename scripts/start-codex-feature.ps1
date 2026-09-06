param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$TaskId,

    [string]$BaseBranch = "main",

    [string]$WorktreeRoot = "D:\project\worktrees",

    [string]$FeatureRequest,

    [ValidateSet("economy", "safe")]
    [string]$Mode = "economy",

    [switch]$NoLaunch
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

$sourceWorkflow = Join-Path $repo ".codex"
$targetWorkflow = Join-Path $worktree ".codex"
if (-not (Test-Path -LiteralPath $sourceWorkflow)) {
    throw "Workflow configuration is missing: $sourceWorkflow"
}

# Bootstrap from the source checkout so the new worktree uses the current
# workflow configuration even before these setup files are committed.
New-Item -ItemType Directory -Force -Path $targetWorkflow | Out-Null
Get-ChildItem -LiteralPath $sourceWorkflow -Force | Copy-Item -Destination $targetWorkflow -Recurse -Force

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

$workflow = switch ($Mode) {
    "economy" {
        [pscustomobject]@{
            Name = "Terra economy workflow"
            Model = "gpt-5.6-terra"
            PromptPath = Join-Path $targetWorkflow "prompts\ECONOMY_WORKER.md"
        }
    }
    "safe" {
        [pscustomobject]@{
            Name = "Sol safe workflow"
            Model = "gpt-5.6-sol"
            PromptPath = Join-Path $targetWorkflow "prompts\SAFE_SUPERVISOR.md"
        }
    }
}

Write-Host ""
Write-Host "Worktree created. Run the baseline before implementation:"
Write-Host "  python -m pytest -q"
Write-Host "  python -m ruff check src tests scripts"
Write-Host ""
if ($NoLaunch) {
    Write-Host "Worktree is ready. Start the $($workflow.Name) later with:"
    Write-Host "  & `"$codexExecutable`" --model $($workflow.Model)"
    return
}

$supervisorPath = $workflow.PromptPath
if (-not (Test-Path -LiteralPath $supervisorPath)) {
    throw "Supervisor prompt is missing: $supervisorPath"
}

$initialPrompt = (Get-Content -LiteralPath $supervisorPath -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($FeatureRequest)) {
    $initialPrompt = "$initialPrompt`r`n`r`n## Feature request`r`nAsk the user for the feature request, then follow this workflow."
} else {
    $initialPrompt = "$initialPrompt`r`n`r`n## Feature request`r`n$FeatureRequest"
}

Write-Host "Starting the $($workflow.Name) in the new worktree..."
& $codexExecutable --model $workflow.Model $initialPrompt
if ($LASTEXITCODE -ne 0) {
    throw "Codex exited with code $LASTEXITCODE"
}
