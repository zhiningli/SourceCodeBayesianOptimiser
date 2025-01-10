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
        """
        Save a script object into the MongoDB collection after ensuring all fields
        are BSON-compatible.
        """
        # Ensure all fields are BSON-compatible
        bson_compatible_object = self._convert_to_bson_compatible(script_object)

        try:
            result = self.collection.insert_one(bson_compatible_object)
            print(f"Data inserted with record id {result.inserted_id}")
            return result.inserted_id
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return None

    def update_script(self, script_object, script_name):
        """
        Update a script object in the MongoDB collection by its script_name
        after ensuring all fields are BSON-compatible, excluding the '_id' field.
        """
        # Ensure all fields are BSON-compatible
        bson_compatible_object = self._convert_to_bson_compatible(script_object)

        # Exclude the '_id' field from the update
        if "_id" in bson_compatible_object:
            bson_compatible_object.pop("_id")

        try:
            result = self.collection.update_one(
                {"script_name": script_name},  # Match by script_name
                {"$set": bson_compatible_object}
            )
        except ConnectionFailure as e:
            print("Connection failure:", e)


    def add_sequential_script_names(self):
        """
        Add sequential script names (script1, script2, ...) to all documents in the collection.
        """
        try:
            documents = list(self.collection.find())  # Retrieve all documents
            for index, document in enumerate(documents, start=1):
                script_name = f"script{index}"
                self.collection.update_one(
                    {"_id": document["_id"]},  # Match the document by its unique _id
                    {"$set": {"script_name": script_name}}  # Add/update the script_name field
                )
            print("Sequential script names added to all documents.")
        except ConnectionFailure as e:
            print("Connection failure:", e)

    def fetch_all_scripts(self):
        """
        Fetch all scripts from the collection.
        """
        try:
            scripts = list(self.collection.find())
            return scripts
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return []

    def _convert_to_bson_compatible(self, value):
        """
        Recursively convert values in a dictionary to BSON-compatible types.
        """
        def convert(value):
            if isinstance(value, torch.Tensor):
                return value.tolist()  # Convert tensors to lists
            elif isinstance(value, np.ndarray):
                return value.tolist()  # Convert numpy arrays to lists
            elif isinstance(value, (float, int, str, list, dict)):
                return value  # Already compatible
            else:
                return str(value)  # Convert other types to strings for safety

        if isinstance(value, dict):
            return {key: convert(val) for key, val in value.items()}
        else:
            return convert(value)
