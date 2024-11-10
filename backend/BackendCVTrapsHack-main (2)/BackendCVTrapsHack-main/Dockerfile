# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /fastapi

# Copy the requirements file into the container at /app
COPY requirements.txt /fastapi/

RUN pip install torch
RUN pip install PytorchWildlife==1.1.0
RUN pip install fastapi uvicorn python-multipart pydantic pandas numpy openpyxl

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY utils_clear.py /fastapi/
COPY main.py /fastapi/
COPY FSRCNN_x4.pb /fastapi/

# Make port 80 available to the world outside this container
EXPOSE 8080

# Set the maintainer label
LABEL maintainer="agishev agishev1961@gmail.com"

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]