# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    unzip \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for the initial setup
WORKDIR /app

# --- MODIFICATION START: Replace 'git clone' with 'COPY' ---
COPY . .

# Set up UV environment variables relative to the new WORKDIR
ENV UV_PATH=./uv
ENV VENV_PATH=$UV_PATH/.venv

ENV UV_UNMANAGED_INSTALL=$UV_PATH
ENV UV_PYTHON_INSTALL_DIR="$UV_PATH/python"
ENV UV_PYTHON_BIN_DIR="$UV_PATH/python/bin"
ENV VIRTUAL_ENV=$VENV_PATH
ENV UV_PROJECT_ENVIRONMENT=$VENV_PATH
ENV UV_TOOL_DIR="$UV_PATH/tools"
ENV UV_TOOL_BIN_DIR="$UV_PATH/tools/bin"
ENV GRADIO_NODE_PATH="$VENV_PATH/lib/python3.12/site-packages/nodejs_wheel/bin/node"
ENV MODELS_DIR=/app/models

# Note: The ./uv/bin will be added to the PATH by the install script itself,
# but we ensure the parent uv directory is in the path.
ENV PATH="/app/uv:${PATH}"

# Install uv using the official script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.cargo/bin/uv /usr/local/bin/uv


# --- DEBUGGING SCRIPT ---
# This will show us exactly what the file system looks like in this layer.
RUN \
    echo "--- STARTING DEBUG SCRIPT -v1 ---" && \
    echo "Current user: $(whoami)" && \
    echo "Current directory (pwd): $(pwd)" && \
    echo "--- Listing contents of current directory recursively ---" && \
    ls -laR . && \
    echo "--- DEBUG SCRIPT FINISHED ---"

# --- FAILING INSTALLATION STEP (for confirmation) ---    
# Install dependencies using the project script
RUN uv venv ./uv/.venv && \
    uv pip install -e ".[cuda]"


# Make the entrypoint script from your project directory executable
RUN chmod +x docker-entrypoint.sh

# Expose port for web interface
EXPOSE 7860

# Set default command to run the web interface
ENTRYPOINT ["./urvc", "run", "--listen", "--listen-host", "0.0.0.0"]