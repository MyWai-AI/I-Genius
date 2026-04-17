# create_netrc.ps1
# Script per creare .netrc file dalle variabili d'ambiente
# Per autenticazione al registry Azure DevOps

$ErrorActionPreference = "Stop"  # Stop the script on any error

# Controlla se le variabili d'ambiente necessarie sono presenti
if (-not $env:MYWAI_ARTIFACTS_MAIL) {
    Write-Host "Errore: MYWAI_ARTIFACTS_MAIL non è definita"
    Write-Host "Impostala con: `$env:MYWAI_ARTIFACTS_MAIL='your_email@company.com'"
    exit 1
}

if (-not $env:MYWAI_ARTIFACTS_TOKEN) {
    Write-Host "Errore: MYWAI_ARTIFACTS_TOKEN non è definita"
    Write-Host "Impostala con: `$env:MYWAI_ARTIFACTS_TOKEN='your_token'"
    exit 1
}

# Determina la directory root del progetto (dove si trova pyproject.toml)
# Se lo script è in scripts/windows/, la root è due livelli sopra
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

# Crea il contenuto del file .netrc nella home directory (dove uv lo cerca)
# Route del progetto: pkgs.dev.azure.com/zenatek-mywai/_packaging/zenatek-mywai/pypi/simple/
# uv cerca il file .netrc nella home directory (~/.netrc) di default
$NETRC_FILE = Join-Path $env:USERPROFILE ".netrc"

# Crea anche una copia nella root del progetto per riferimento
$PROJECT_NETRC_FILE = Join-Path $PROJECT_ROOT ".netrc"

Write-Host "HOME=$env:USERPROFILE"
Write-Host "PROJECT_ROOT=$PROJECT_ROOT"
Write-Host "NETRC_FILE=$NETRC_FILE"

# Crea il contenuto del file .netrc
$netrcContent = @"
machine pkgs.dev.azure.com
    login $env:MYWAI_ARTIFACTS_MAIL
    password $env:MYWAI_ARTIFACTS_TOKEN
"@

# Crea il file .netrc nella home directory (dove uv lo cerca)
$netrcContent | Out-File -FilePath $NETRC_FILE -Encoding ASCII -NoNewline

# Imposta i permessi corretti (solo proprietario può leggere/scrivere)
$acl = Get-Acl $NETRC_FILE
$acl.SetAccessRuleProtection($true, $false)
$accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $env:USERNAME,
    "FullControl",
    "Allow"
)
$acl.SetAccessRule($accessRule)
Set-Acl -Path $NETRC_FILE -AclObject $acl

# Crea anche una copia nella root del progetto per riferimento
$netrcContent | Out-File -FilePath $PROJECT_NETRC_FILE -Encoding ASCII -NoNewline
$acl = Get-Acl $PROJECT_NETRC_FILE
$acl.SetAccessRuleProtection($true, $false)
$accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $env:USERNAME,
    "FullControl",
    "Allow"
)
$acl.SetAccessRule($accessRule)
Set-Acl -Path $PROJECT_NETRC_FILE -AclObject $acl

Write-Host ".netrc file creato con successo!"
Write-Host "File creato in home directory: $NETRC_FILE (uv lo userà automaticamente)"
Write-Host "Copia creata anche in: $PROJECT_NETRC_FILE"
Write-Host "File creato per autenticazione al registry Azure DevOps"

