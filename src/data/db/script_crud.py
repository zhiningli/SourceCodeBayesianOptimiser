from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import torch
import numpy as np

class ScriptRepository:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="script_database", collection_name="scripts"):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_scripts(self, script_object):
        # Ensure all fields are BSON-compatible
        def convert_to_bson_compatible(value):
            if isinstance(value, torch.Tensor):
                return value.tolist()  # Convert tensors to lists
            elif isinstance(value, np.ndarray):
                return value.tolist()  # Convert numpy arrays to lists
            elif isinstance(value, (float, int, str, list, dict)):
                return value  # Already compatible
            else:
                return str(value)  # Convert other types to strings for safety

        # Recursively convert script_object fields to BSON-compatible types
        bson_compatible_object = {
            key: convert_to_bson_compatible(value) for key, value in script_object.items()
        }

        try:
            result = self.collection.insert_one(bson_compatible_object)
            print(f"Data inserted with record id {result.inserted_id}")
            return result.inserted_id
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return None
