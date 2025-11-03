FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download SAM checkpoint if not exists (large file, download during build)
RUN if [ ! -f "sam_vit_h.pth" ]; then \
    echo "Downloading SAM checkpoint (2.4GB)..."; \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth; \
    fi

# Expose port
EXPOSE ${PORT:-10000}

# Run the app
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}

