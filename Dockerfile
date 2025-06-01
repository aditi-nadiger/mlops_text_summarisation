# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the app
COPY . .

# Streamlit needs this environment variable for clean logs
ENV PYTHONUNBUFFERED=1

# Expose the default Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "Scripts/app.py"]
