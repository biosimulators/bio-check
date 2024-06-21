from datetime import datetime

from pymongo.mongo_client import MongoClient


# TODO: Add mongo interaction here.

def timestamp():
    return str(datetime.utcnow())
