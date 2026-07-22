# Start from an official Python image
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# The Lux game engine expects the model file to be at a certain path.
# You can set an environment variable to point to it.
ENV MODEL_PATH=models/model_final.h5

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
