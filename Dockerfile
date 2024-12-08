FROM python:3.9-slim

WORKDIR /app

# System deps (adjust as needed)
RUN apt-get update && apt-get install -y git curl build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# For Flask
EXPOSE 5000

CMD ["python", "app.py"]
