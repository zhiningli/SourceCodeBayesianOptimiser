from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import torch
import numpy as np


class ModelRepository:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="script_database", collection_name="models"):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_models(self, model_object):
        """
        Save a model object into the MongoDB collection after ensuring all fields
        are BSON-compatible.
        """
        bson_compatible_object = self._convert_to_bson_compatible(model_object)
        try:
            result = self.collection.insert_one(bson_compatible_object)
            print(f"Data inserted with record id {result.inserted_id}")
            return result.inserted_id
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return None

    def fetch_all_models(self):
        """
        Fetch all models from the collection.
        """
        try:
            models = list(self.collection.find())
            return models
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return []

    def update_model(self, model_name, updated_fields):
        """
        Update a model by name with the given fields.
        
        Parameters:
        - model_name: The name of the model to update.
        - updated_fields: A dictionary of fields to update.
        
        Returns:
        - result: The result of the update operation.
        """
        bson_compatible_fields = self._convert_to_bson_compatible(updated_fields)

        try:
            result = self.collection.update_one(
                {"model_name": model_name}, 
                {"$set": bson_compatible_fields}  # Update field
            )
            if result.matched_count == 0:
                print(f"No model found with name: {model_name}")
            else:
                print(f"Model '{model_name}' updated successfully.")
            return result
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return None

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
