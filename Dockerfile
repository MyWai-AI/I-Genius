FROM python:3.10.14-slim

ARG DEBUG_MODE="false"
ARG LOCAL_HOST_RUN_ENV="false"

ENV DEBUG_MODE=${DEBUG_MODE}
ENV LOCAL_HOST_RUN_ENV=${LOCAL_HOST_RUN_ENV}

RUN apt-get update && apt-get install -y --no-install-recommends         wget         git         vim         libgl1         build-essential         libglib2.0-0         curl         supervisor         ffmpeg         && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - &&         apt-get update && apt-get install -y nodejs &&         rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install uv

WORKDIR /app

COPY . .

RUN uv sync

RUN mkdir -p data/BAG data/Generic data/SVO data/uploads data/downloads

RUN cd src/streamlit_template/components/sync_viewer/frontend-src &&         npm install &&         npm run build &&         npm cache clean --force &&         rm -rf node_modules

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8504

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
