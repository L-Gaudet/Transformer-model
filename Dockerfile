# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.0-base

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Set the default command to run when the container starts
CMD ["python3", "SentimentTraining.py"]
