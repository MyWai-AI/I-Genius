#!/bin/bash

# docker_build.sh
# Builds the Docker image and passes environment variables during the build process

set -e  # Exit on any error

IMAGE_NAME="streamlit-template"
TAG="latest"

echo "Starting Docker build for image ${IMAGE_NAME}:${TAG}"

# Load environment variables from .env file if present
if [ -f ".env" ]; then
    echo "Loading variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Check that required environment variables are defined
if [ -z "$MYWAI_ARTIFACTS_MAIL" ]; then
    echo "Error: environment variable MYWAI_ARTIFACTS_MAIL is not defined."
    echo "Set it with: export MYWAI_ARTIFACTS_MAIL='your_email@company.com'"
    exit 1
fi

if [ -z "$MYWAI_ARTIFACTS_TOKEN" ]; then
    echo "Error: environment variable MYWAI_ARTIFACTS_TOKEN is not defined."
    echo "Set it with: export MYWAI_ARTIFACTS_TOKEN='your_token'"
    exit 1
fi

# Run the Docker build command with build arguments
docker build \
  -t "${IMAGE_NAME}:${TAG}" \
  --build-arg MYWAI_ARTIFACTS_MAIL="$MYWAI_ARTIFACTS_MAIL" \
  --build-arg MYWAI_ARTIFACTS_TOKEN="$MYWAI_ARTIFACTS_TOKEN" \
  .

echo "Docker build completed for ${IMAGE_NAME}:${TAG}"

