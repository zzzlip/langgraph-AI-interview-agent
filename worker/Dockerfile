FROM python:3.10-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fontconfig \
        fonts-noto-cjk \
        libjpeg62-turbo \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libpng16-16 \
        libxml2 \
        libxslt1.1 \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/.data
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "-m", "ai_interview_worker.interface.cli"]
CMD ["health"]
