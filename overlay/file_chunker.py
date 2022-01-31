import validation_pb2
from logging import info, error


# Just for testing chunk_file
def main():
    filename = "../uploads/my_model.zip"
    with open(filename, "rb") as f:
        for chunk in chunk_file(f):
            info(chunk)


def chunk_file(f, request_id):
    for chunk in iter(lambda: f.read(4096), bytes()):
        info(f"Read {len(chunk)} bytes")
        yield validation_pb2.FileChunk(id=f"{request_id}.zip", data=chunk)


if __name__ == "__main__":
    main()
