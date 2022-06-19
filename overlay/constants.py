import os
from dotenv import load_dotenv

import urllib.parse

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = int(os.environ.get("DB_PORT"))
DB_NAME = os.environ.get("DB_NAME")
MODELS_DIR = os.environ.get("MODELS_DIR")
# username = urllib.parse.quote_plus(str(os.environ.get('ROOT_MONGO_USER')))
# password = urllib.parse.quote_plus(str(os.environ.get('ROOT_MONGO_PASS')))
# username = "root"
# password = "rootPass"

username = "sqsclient"
password = "ElongatedMuskrat76"
# export READ_MONGO_PASS="ElongatedMuskrat76"
# export READ_MONGO_USER="sqsclient"

if __name__ == "__main__":
    print(f'DB_HOST: {DB_HOST}')
    print(f'DB_PORT: {DB_PORT}')
    print(f'DB_NAME: {DB_NAME}')
    print(f'MODELS_DIR: {MODELS_DIR}')
    print(f'username: {username}')
    print(f'password: {password}')
