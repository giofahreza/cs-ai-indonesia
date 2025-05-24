FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (create requirements.txt if you don't have it)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose port
EXPOSE 8082

# Run the app
CMD ["python", "app.py"]