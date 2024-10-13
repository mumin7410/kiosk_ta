# Use the official Python image from the Docker Hub
FROM python:3.11

# Set environment variables to ensure Python output is sent straight to the terminal
# and prevent buffering issues
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Install OpenCV dependencies (libGL)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Expose the port that the Django application will run on
EXPOSE 8000

# Set the default command to run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]