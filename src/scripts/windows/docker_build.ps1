# build_docker.ps1
# Builds the Docker image and passes environment variables during the build process

$ErrorActionPreference = "Stop"  # Stop the script on any error

$IMAGE_NAME = "streamlit-template"
$TAG = "latest"

Write-Host ("Starting Docker build for image {0}:{1}" -f $IMAGE_NAME, $TAG)

# Load environment variables from .env file if present
if (Test-Path ".env") {
    Write-Host "Loading variables from .env file"
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Check that required environment variables are defined
if (-not $env:MYWAI_ARTIFACTS_MAIL) {
    Write-Host "Error: environment variable MYWAI_ARTIFACTS_MAIL is not defined."
    Write-Host "Set it with: `$env:MYWAI_ARTIFACTS_MAIL='your_email@company.com'"
    exit 1
}

if (-not $env:MYWAI_ARTIFACTS_TOKEN) {
    Write-Host "Error: environment variable MYWAI_ARTIFACTS_TOKEN is not defined."
    Write-Host "Set it with: `$env:MYWAI_ARTIFACTS_TOKEN='your_token'"
    exit 1
}

# Run the Docker build command with build arguments
docker build `
  -t "$($IMAGE_NAME):$($TAG)" `
  --build-arg MYWAI_ARTIFACTS_MAIL=$env:MYWAI_ARTIFACTS_MAIL `
  --build-arg MYWAI_ARTIFACTS_TOKEN=$env:MYWAI_ARTIFACTS_TOKEN `
  .

Write-Host ("Docker build completed for {0}:{1}" -f $IMAGE_NAME, $TAG)

