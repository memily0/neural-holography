FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./requirements.txt
RUN python -m pip install --only-binary=:all: -r requirements.txt \
    && python -m pip check

COPY src ./src
COPY models ./models
RUN test -s /app/models/solution.pt

ENTRYPOINT ["python", "-m", "src.predict", "--checkpoint", "/app/models/solution.pt", "--device", "cpu"]
CMD ["--help"]
