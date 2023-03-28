FROM rockylinux:9.1 AS base

# Install python3 and development tools
RUN dnf update -y && dnf install -y python3 python3-pip
# RUN dnf groupinstall -y development
RUN python3 -V

# Set up operations in /app
RUN mkdir -p /app
WORKDIR /app

# Install python requirements and create .env file
COPY *.sh requirements.txt ./
RUN python3 -m pip install -r requirements.txt && \
    echo -e "DB_HOST=localhost\nDB_PORT=27018\nDB_NAME=sustaindb\nMODELS_DIR=/tmp/validation-service/saved_models" > .env

# Copy in source files
RUN mkdir overlay resources
COPY overlay/ ./overlay/
COPY resources/ ./resources/


ENTRYPOINT ["sleep", "86400"]