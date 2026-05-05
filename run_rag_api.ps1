$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"

if (-not $env:LLM_PROVIDER) {
  $env:LLM_PROVIDER = "openrouter"
}

if (-not $env:OPENROUTER_MODEL) {
  $env:OPENROUTER_MODEL = "openrouter/free"
}

if (-not $env:OPENROUTER_SITE_URL) {
  $env:OPENROUTER_SITE_URL = "http://localhost:3000"
}

if (-not $env:OPENROUTER_APP_NAME) {
  $env:OPENROUTER_APP_NAME = "Feasible"
}

if (-not $env:OPENROUTER_BASE_URL) {
  $env:OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
}

if (-not $env:OLLAMA_MODEL) {
  $env:OLLAMA_MODEL = "gpt-oss:20b"
}

if (-not $env:CHROMA_DIR) {
  $default = Join-Path $PSScriptRoot "rag\chroma_db"
  $active = Join-Path $PSScriptRoot "rag\chroma_db_active"
  $fresh = Join-Path $PSScriptRoot "rag\chroma_db_active_fresh\chroma_db"

  if (Test-Path (Join-Path $default "chroma.sqlite3")) {
    $env:CHROMA_DIR = $default
  } elseif (Test-Path (Join-Path $active "chroma.sqlite3")) {
    $env:CHROMA_DIR = $active
  } elseif (Test-Path (Join-Path $fresh "chroma.sqlite3")) {
    $env:CHROMA_DIR = $fresh
  } else {
    $env:CHROMA_DIR = $default
  }
}

if (-not $env:RAG_API_BASE_URL) {
  $env:RAG_API_BASE_URL = "http://localhost:8000"
}

if (-not $env:RAG_REQUEST_TIMEOUT_SECONDS) {
  $env:RAG_REQUEST_TIMEOUT_SECONDS = "60"
}

if (-not $env:RUN_TELEGRAM_BOT) {
  $env:RUN_TELEGRAM_BOT = "true"
}

Write-Host "Starting Cambodian Legal RAG API"
Write-Host "  Provider  : $env:LLM_PROVIDER"
if ($env:LLM_PROVIDER -eq "openrouter") {
  Write-Host "  Model     : $env:OPENROUTER_MODEL"
} else {
  Write-Host "  Model     : $env:OLLAMA_MODEL"
}
Write-Host "  Chroma DB : $env:CHROMA_DIR"
Write-Host "  URL       : http://localhost:8000"
Write-Host ""

if ($env:LLM_PROVIDER -eq "ollama") {
  Write-Host "Make sure Ollama is running separately:"
  Write-Host "  ollama serve"
} else {
  Write-Host "Make sure OPENROUTER_API_KEY is set in .env or your shell."
}
Write-Host ""

if ($env:RUN_TELEGRAM_BOT -eq "true") {
  if ($env:TELEGRAM_BOT_TOKEN) {
    Write-Host "Starting Telegram bot"
    Write-Host "  Backend   : $env:RAG_API_BASE_URL"
    Write-Host ""
    Start-Process -FilePath "python" `
      -ArgumentList "bot.py" `
      -WorkingDirectory (Join-Path $PSScriptRoot "telegram_bot") `
      -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $PSScriptRoot "telegram_bot_stdout.log") `
      -RedirectStandardError (Join-Path $PSScriptRoot "telegram_bot_stderr.log") | Out-Null
  } else {
    Write-Host "Skipping Telegram bot startup because TELEGRAM_BOT_TOKEN is not set."
    Write-Host ""
  }
}

Set-Location (Join-Path $PSScriptRoot "rag")
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
