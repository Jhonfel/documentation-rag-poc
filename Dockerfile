# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install the necessary packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code to the container
COPY . /app

# Expose port 5000 for the server
EXPOSE 5000

# Install Gunicorn
RUN pip install gunicorn

# Command to run the app using Gunicorn instead of the Flask development server
CMD ["gunicorn", "--workers=3", "--bind=0.0.0.0:5000", "app:app"]
