def chunk_file(f):
    for chunk in iter(lambda: f.read(4096), ''):
        yield chunk
