from . import validation_pb2
from loguru import logger


# Just for testing chunk_file
def main():
    filename = "../uploads/my_model.zip"
    with open(filename, "rb") as f:
        for chunk in chunk_file(f):
            logger.info(chunk)


def chunk_file(f, request_id):
    for chunk in iter(lambda: f.read(4096), bytes()):
        logger.info(f"Read {len(chunk)} bytes")
        yield validation_pb2.FileChunk(id=f"{request_id}.zip", data=chunk)


def read_file_bytes(file_path):
    file_bytes = None
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return file_bytes


if __name__ == "__main__":
    main()
