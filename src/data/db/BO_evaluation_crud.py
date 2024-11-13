from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from enum import Enum

class BO_evaluation_db:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="source_code_database", collection_name="BO_evaluation_collection"):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_BO_evaluation_results(self, evaluation_results):
        """
        Save a new evaluation result to the collection.
        
        Parameters:
        evaluation_results (dict): A dictionary containing the evaluation result.
        
        Returns:
        ObjectId: The ID of the inserted document.
        """
        try:
            result = self.collection.insert_one(evaluation_results)
            print(f"Data inserted with record id {result.inserted_id}")
            return result.inserted_id
        except ConnectionFailure as e:
            print("Connection failure: ", e)
            return None

    def get_BO_evaluation_result_by_id(self, source_code_id):
        """
        Retrieve an evaluation result by source code ID.
        
        Parameters:
        source_code_id (str): The ID of the source code to search for.
        
        Returns:
        dict: The evaluation result if found, else None.
        """
        try:
            result = self.collection.find_one({"source_code_id": source_code_id})
            if result:
                print("Evaluation result found:", result)
            else:
                print("No evaluation result found for the given source code ID.")
            return result
        except ConnectionFailure as e:
            print("Connection failure: ", e)
            return None

    def update_BO_evaluation_result(self, source_code_id, updated_fields):
        """
        Update an existing evaluation result by source code ID.
        
        Parameters:
        source_code_id (str): The ID of the source code to update.
        updated_fields (dict): The fields to update in the evaluation result.
        
        Returns:
        dict: The result of the update operation.
        """
        try:
            result = self.collection.update_one(
                {"source_code_id": source_code_id},
                {"$set": updated_fields}
            )
            if result.modified_count > 0:
                print(f"Successfully updated the evaluation result for source code ID: {source_code_id}")
            else:
                print(f"No document was updated for source code ID: {source_code_id}")
            return result.raw_result
        except ConnectionFailure as e:
            print("Connection failure: ", e)
            return None

    def delete_BO_evaluation_result(self, source_code_id):
        """
        Delete an evaluation result by source code ID.
        
        Parameters:
        source_code_id (str): The ID of the source code to delete.
        
        Returns:
        dict: The result of the delete operation.
        """
        try:
            result = self.collection.delete_one({"source_code_id": source_code_id})
            if result.deleted_count > 0:
                print(f"Successfully deleted the evaluation result for source code ID: {source_code_id}")
            else:
                print(f"No document found to delete for source code ID: {source_code_id}")
            return result.raw_result
        except ConnectionFailure as e:
            print("Connection failure: ", e)
            return None

    def get_all_BO_evaluation_results(self):
        """
        Retrieve all evaluation results from the collection.
        
        Returns:
        list: A list of all evaluation result documents.
        """
        try:
            results = list(self.collection.find())
            print(f"Retrieved {len(results)} evaluation results.")
            return results
        except ConnectionFailure as e:
            print("Connection failure: ", e)
            return []

    def check_connection(self):
        """
        Check if the connection to the MongoDB server is successful.
        
        Returns:
        bool: True if connected, False otherwise.
        """
        try:
            self.client.admin.command('ping')
            print("Connection to MongoDB is successful.")
            return True
        except ConnectionFailure:
            print("Failed to connect to MongoDB.")
            return False
