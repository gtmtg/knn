FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3.7-dev wget \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements-build.txt \
    && pip install -r requirements.txt

# Download external resources.
RUN wget -i resources.txt

# Clean up build dependencies.
RUN grep -F -x -v -f requirements.txt requirements-build.txt | pip uninstall -y -r /dev/stdin \
    && apt-get purge -y --auto-remove gcc python3.7-dev

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 handler:app
