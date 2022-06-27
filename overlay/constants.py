import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = int(os.environ.get("DB_PORT"))
DB_NAME = os.environ.get("DB_NAME")
DB_USERNAME = "sqsclient"
DB_PASSWORD = "ElongatedMuskrat76"
DB_AUTH_SOURCE = "admin"
MODELS_DIR = os.environ.get("MODELS_DIR")

if __name__ == "__main__":
    print(f'DB_HOST: {DB_HOST}')
    print(f'DB_PORT: {DB_PORT}')
    print(f'DB_NAME: {DB_NAME}')
    print(f'DB_USERNAME: {DB_USERNAME}')
    print(f'DB_PASSWORD: {DB_PASSWORD}')
    print(f'DB_AUTH_SOURCE: {DB_AUTH_SOURCE}')
    print(f'MODELS_DIR: {MODELS_DIR}')
