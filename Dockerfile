FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build time এই model download করো — runtime এ আর লাগবে না
RUN python -c "from rembg import new_session; new_session('u2net')"

# App copy করো
COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]