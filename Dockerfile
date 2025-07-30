FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# ffmpeg for video I/O
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps via uv into a project venv at /app/.venv
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --no-dev --no-install-project --locked
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --locked
# Install PyTorch (CUDA 12.1) **inside the same venv**
RUN . .venv/bin/activate && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision


# Use the uv-managed venv at runtime
CMD ["uv", "run", "python", "train_bair_rolling.py", "--help"]  