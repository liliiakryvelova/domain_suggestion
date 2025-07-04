# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port
EXPOSE 10000

# Run the app
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "10000", "--reload"]
