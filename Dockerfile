# filepath: c:\Chaitanya\MolFECAM\MolFECAM-Backend\Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12.11-slim-bookworm

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app and data directories into the container at /code
COPY ./app /code/app
COPY ./data /code/data

# Command to run the application
# It tells uvicorn to run the `app` object from the `app.main` module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]