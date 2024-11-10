from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from enum import Enum

class SourceCodeStatus(Enum):
    GENERATED_FROM_TEMPLATE = "generated_from_template"
    VALIDATED_TO_RUN = "validated_to_run"
    FAILED_VALIDATION = "failed_validation"
    ABLE_TO_RUN_BO = "able_to_run_bo"
    UNABLE_TO_RUN_BO = "unable_to_run_bo"
    EVALUATED_BY_SCRIPTS = "evaluated_by_script"
    FAILED_AUTO_EVALUATION = "failed_auto_evaluation"

class SourceCodeRepository:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="source_code_database", collection_name="source_code_collection"):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_source_code(self, source_code_object, status):
        try:
            data = self._prepare_document(source_code_object, status)
            result = self.collection.insert_one(data)
            print(f"Data inserted with record id {result.inserted_id}")
            return result.inserted_id
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return None

    def save_source_codes_batch(self, source_code_objects, status):
        documents = [self._prepare_document(obj, status) for obj in source_code_objects]
        try:
            result = self.collection.insert_many(documents)
            print(f"Batch insert successful with record ids {result.inserted_ids}")
            return result.inserted_ids
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return []

    def get_source_code(self, record_id=None, name=None, source_code_type=None, library=None, status = None):
        """
        Retrieve a single source code document from the database based on given filters.
        
        Parameters:
            record_id: The MongoDB ObjectId of the document.
            name: The name of the source code document.
            source_code_type: The type of the source code (e.g., "SVM").
            library: The library associated with the dataset (e.g., "sklearn", "openml").

        Returns:
            dict: The source code document, or None if no document is found.
        """
        query = {}
        
        # Build query based on provided parameters
        if record_id:
            query["_id"] = record_id
        if name:
            query["name"] = name
        if source_code_type:
            query["source_code_type"] = source_code_type
        if library:
            query["dataset_info.dataset_library"] = library
        if status:
            query["status"] = status

        # Execute the query and return a single document
        return self.collection.find_one(query)

    def find_source_codes(self, name=None, source_code_type=None, library=None, status=None, limit=10):
        """
        Retrieve multiple source code documents from the database based on given filters.
        
        Parameters:
            name: The name of the source code documents to search.
            source_code_type: The type of the source code (e.g., "SVM").
            library: The library associated with the dataset (e.g., "sklearn", "openml").
            limit: The maximum number of documents to return.

        Returns:
            list: A list of source code documents that match the criteria.
        """
        query = {}
        
        # Build query based on provided parameters
        if name:
            query["name"] = name
        if source_code_type:
            query["source_code_type"] = source_code_type
        if library:
            query["dataset_info.dataset_library"] = library
        if status:
            query["status"] = status

        # Execute the query and return multiple documents
        return list(self.collection.find(query).limit(limit))

    def update_source_code(self, record_id, update_data):
        try:
            result = self.collection.update_one({"_id": record_id}, {"$set": update_data})
            return result.modified_count
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return 0

    def delete_source_code(self, record_id):
        try:
            result = self.collection.delete_one({"_id": record_id})
            return result.deleted_count
        except ConnectionFailure as e:
            print("Connection failure:", e)
            return 0

    def _prepare_document(self, source_code_object, status):
        """Helper method to prepare the document data for insertion."""
        return {
            "name": source_code_object.name,
            "source_code_type": source_code_object.source_code_type,
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
    