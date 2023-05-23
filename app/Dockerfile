# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the code and model file into the container
COPY main.py /app/main.py
COPY clothing_model.pkl /app/clothing_model.pkl

# Install the required dependencies
RUN pip install -r requirement.txt

# Set the command to run your code
CMD [ "python", "/app/main.py" ]