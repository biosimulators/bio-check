import uuid
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_DB_URI")


def unique_id(): return str(uuid.uuid4())
