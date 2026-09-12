# Установка Ollama как службы Windows через NSSM (урок 3).
#
# Зачем: десктопный Ollama живёт в сессии пользователя и умирает вместе с ней.
# Служба Windows стартует при загрузке ОС, перезапускается при падении
# и не зависит от того, залогинен ли кто-то на сервере.
#
# Подготовка:
#   1. Установите NSSM: winget install NSSM.NSSM  (или скачайте с https://nssm.cc)
#   2. Отключите автозапуск десктопного Ollama (иконка в трее -> Quit;
#      Параметры -> Приложения -> Автозагрузка -> Ollama = Выкл),
#      иначе служба не сможет занять порт 11434.
#   3. Запустите этот скрипт из PowerShell ОТ АДМИНИСТРАТОРА:
#      powershell -ExecutionPolicy Bypass -File .\install-ollama-service.ps1
#
# Удаление службы:  nssm stop OllamaService; nssm remove OllamaService confirm

param(
    [string]$ServiceName = "OllamaService",
    [string]$OllamaExe   = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
    [string]$ModelsDir   = "C:\ollama\models",
    [string]$LogDir      = "C:\ollama\logs",
    [string]$BindHost    = "127.0.0.1:11434",
    [string]$KeepAlive   = "30m",
    [int]$NumParallel    = 1,
    [int]$MaxQueue       = 128
)

$ErrorActionPreference = "Stop"

# --- Проверки перед установкой -------------------------------------------
$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) {
    Write-Error "NSSM не найден в PATH. Установите: winget install NSSM.NSSM"
}

if (-not (Test-Path $OllamaExe)) {
    Write-Error "ollama.exe не найден по пути: $OllamaExe. Укажите -OllamaExe явно."
}

# Служба работает под LocalSystem, у которого 'домашний' профиль -
# C:\Windows\system32\config\systemprofile. Если не задать OLLAMA_MODELS,
# модели уедут туда. Поэтому каталоги создаём явно.
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# --- Установка службы ------------------------------------------------------
& nssm install $ServiceName $OllamaExe serve

# Переменные окружения демона (каждая пара KEY=VALUE - отдельный аргумент)
& nssm set $ServiceName AppEnvironmentExtra `
    "OLLAMA_HOST=$BindHost" `
    "OLLAMA_MODELS=$ModelsDir" `
    "OLLAMA_KEEP_ALIVE=$KeepAlive" `
    "OLLAMA_NUM_PARALLEL=$NumParallel" `
    "OLLAMA_MAX_LOADED_MODELS=1" `
    "OLLAMA_MAX_QUEUE=$MaxQueue"

# Логи с ротацией (10 МБ на файл)
& nssm set $ServiceName AppStdout "$LogDir\ollama.log"
& nssm set $ServiceName AppStderr "$LogDir\ollama-err.log"
& nssm set $ServiceName AppRotateFiles 1
& nssm set $ServiceName AppRotateOnline 1
& nssm set $ServiceName AppRotateBytes 10485760

# Автостарт при загрузке ОС и перезапуск при падении через 5 секунд
& nssm set $ServiceName Start SERVICE_AUTO_START
& nssm set $ServiceName AppExit Default Restart
& nssm set $ServiceName AppRestartDelay 5000

& nssm start $ServiceName

Write-Host ""
Write-Host "Служба '$ServiceName' установлена и запущена." -ForegroundColor Green
Write-Host "Проверка:  Invoke-RestMethod http://127.0.0.1:11434/api/version"
Write-Host "Статус:    nssm status $ServiceName   (или Get-Service $ServiceName)"
Write-Host "Логи:      Get-Content $LogDir\ollama.log -Tail 50"
