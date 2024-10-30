from pymongo import MongoClient
from datetime import datetime

class SourceCodeCRUD:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="source_code_database", collection_name="source_code_collection"):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_source_code(self, source_code_object, status):
        data = {
            "name": source_code_object.name,
            "source_code": source_code_object.get_source_code,
            "source_code_hyperparameters": source_code_object.source_code_hyperparameters,
            "optimalBOHyperparameters": source_code_object.optimalBOHyperParameters,
            "dataset_info": {
                "dataset_library": source_code_object.library,
                "dataset_id": source_code_object.dataset_id,
                "dataset_name": source_code_object.dataset_name,
            },
            "status": status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        result = self.collection.insert_one(data)
        print(f"Data inserted with record id {result.inserted_id}")
        return result.inserted_id

    def get_source_code(self, record_id):
        return self.collection.find_one({"_id": record_id})

    def update_source_code(self, record_id, update_data):
        result = self.collection.update_one({"_id": record_id}, {"$set": update_data})
        return result.modified_count

    def delete_source_code(self, record_id):
        result = self.collection.delete_one({"_id": record_id})
        return result.deleted_count
