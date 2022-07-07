import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = int(os.environ.get("DB_PORT"))
DB_NAME = os.environ.get("DB_NAME")
MODELS_DIR = os.environ.get("MODELS_DIR")
# username = os.environ.get('ROOT_MONGO_USER')
# password = os.environ.get('ROOT_MONGO_PASS')
username = "root"
password = "rootPass"

if __name__ == "__main__":
    print(f'DB_HOST: {DB_HOST}')
    print(f'DB_PORT: {DB_PORT}')
    print(f'DB_NAME: {DB_NAME}')
    print(f'MODELS_DIR: {MODELS_DIR}')
    print(f'username: {username}')
    print(f'password: {password}')
