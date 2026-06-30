FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
