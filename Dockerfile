# Use a slim Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port and run the FastAPI app
# Note: we use api.main:app because your file is in the api/ folder
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]