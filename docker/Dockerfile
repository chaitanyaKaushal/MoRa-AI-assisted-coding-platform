FROM python:3.10-slim

# Prevent python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install basic dependencies if needed (e.g., numpy for math problems)
RUN pip install numpy

# The container expects to mount a volume at /app/sandbox
CMD ["tail", "-f", "/dev/null"]