FROM rockylinux:9.1 AS base

# Install python3 and development tools
RUN dnf update -y && dnf install -y python3 python3-pip
# RUN dnf groupinstall -y development
RUN python3 -V

# Set up operations in /app
RUN mkdir -p /app
WORKDIR /app

COPY *.sh requirements.txt overlay/ resources/ /app/

# Install python requirements and create .env file
RUN python3 -m pip install -r requirements.txt && \
    echo -e "DB_HOST=localhost\nDB_PORT=27018\nDB_NAME=sustaindb\nMODELS_DIR=/tmp/validation-service/saved_models" > .env


ENTRYPOINT ["sleep", "86400"]