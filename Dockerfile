# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY train_model.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for model output
RUN mkdir -p /app/model_output

# Set command to run the Python script
CMD ["python", "train_model.py"]
