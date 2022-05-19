from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, SimpleProgress, Timer

# Progress Bar widgets
widgets = [SimpleProgress(), Percentage(), Bar(), Timer()]


def main():
    client = MongoClient("mongodb://localhost:27018")
    db = client["sustaindb"]
    col = db["noaa_nam"]
    total = col.count_documents()
    all_docs_cursor = col.find()
    seq = 0
    bar = ProgressBar(maxval=total, widgets=widgets).start()
    for doc in all_docs_cursor:
        doc_id = doc["_id"]
        col.find_one_and_update(filter={"_id": doc_id}, update={"$set": {"SEQUENCE": seq}})
        seq += 1

        if seq % 100 == 0:
            bar.update(seq)

    bar.finish()


if __name__ == "__main__":
    main()
