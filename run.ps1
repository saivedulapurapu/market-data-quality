<#
.SYNOPSIS
  Windows entry point. Usage: .\run.ps1 <setup|fetch|pipeline|benchmark|reports|all|test|cov|dashboard>
#>
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Need-Venv { if (-not (Test-Path $Py)) { throw "No .venv found. Run: .\run.ps1 setup" } }

switch ($Task) {
  "setup" {
    py -3.13 -m venv (Join-Path $PSScriptRoot ".venv")
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
  }
  "fetch"     { Need-Venv; & $Py -m mdq fetch }
  "pipeline"  { Need-Venv; & $Py -m mdq pipeline }
  "benchmark" { Need-Venv; & $Py -m mdq benchmark }
  "reports"   { Need-Venv; & $Py -m mdq reports }
  "all"       { Need-Venv; & $Py -m mdq all }
  "test"      { Need-Venv; & $Py -m pytest }
  "cov"       { Need-Venv; & $Py -m pytest --cov=mdq --cov-report=term-missing }
  "dashboard" { Need-Venv; & $Py -m streamlit run (Join-Path $PSScriptRoot "app\streamlit_app.py") }
  default     { Write-Host "Usage: .\run.ps1 <setup|fetch|pipeline|benchmark|reports|all|test|cov|dashboard>" }
}
