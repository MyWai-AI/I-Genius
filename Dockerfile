FROM python:3.10.14-slim

# Set noninteractive mode for apt-get and Conda
#ARG DEBIAN_FRONTEND=noninteractive

ARG MYWAI_ARTIFACTS_MAIL=""
ARG MYWAI_ARTIFACTS_TOKEN=""
ARG DEBUG_MODE="false"
ARG LOCAL_HOST_RUN_ENV="false"

ENV DEBUG_MODE=${DEBUG_MODE}
ENV LOCAL_HOST_RUN_ENV=${LOCAL_HOST_RUN_ENV}

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    libgl1 \
    build-essential \
    libglib2.0-0 \
    curl \
    supervisor \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip \
    && pip install uv

WORKDIR /mywai

COPY src/scripts/linux/create_netrc.sh scripts/create_netrc.sh
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md

RUN apt-get update && apt-get install -y dos2unix
RUN dos2unix scripts/create_netrc.sh

RUN chmod +x scripts/create_netrc.sh \
    && if [ -n "$MYWAI_ARTIFACTS_MAIL" ] && [ -n "$MYWAI_ARTIFACTS_TOKEN" ]; then \
        export MYWAI_ARTIFACTS_MAIL="$MYWAI_ARTIFACTS_MAIL" \
        && export MYWAI_ARTIFACTS_TOKEN="$MYWAI_ARTIFACTS_TOKEN" \
        && bash scripts/create_netrc.sh; \
    fi



RUN UV_INDEX_MYWAI_USERNAME=m.abdollahi@myw.ai \
    UV_INDEX_MYWAI_PASSWORD=$UV_INDEX_MYWAI_PASSWORD \
    uv sync

EXPOSE 8504

# copy repository
COPY . .

# Create the folder structure for data since they are .dockerignored to save space
RUN mkdir -p data/BAG data/Generic data/SVO data/uploads data/downloads

# Build sync_viewer
RUN cd src/streamlit_template/components/sync_viewer/frontend-src && \
    npm install && \
    npm run build && \
    npm cache clean --force && \
    rm -rf node_modules

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
