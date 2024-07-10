FROM python:3.11.9-slim

# Copy requirements file
COPY requirements_docker.txt .

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependecies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy API
RUN mkdir /src/
COPY ./src /src

# Set workdir
WORKDIR /

# Expose app port
EXPOSE 9000

# Start application
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]