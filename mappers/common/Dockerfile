FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies.
COPY requirements-build.txt ./
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3.7-dev wget \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements-build.txt \
    && pip install -r requirements.txt

# Download external resources.
COPY resources.txt ./
RUN wget -i resources.txt

# Clean up build dependencies.
RUN grep -F -x -v -f requirements.txt requirements-build.txt | pip uninstall -y -r /dev/stdin \
    && apt-get purge -y --auto-remove gcc python3.7-dev wget

# Run the web service on container startup.
COPY . ./
CMD exec uvicorn --host "0.0.0.0" --port $PORT --workers 1 handler:mapper
