from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime

class SourceCodeRepository:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="source_code_database", collection_name="source_code_collection"):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def save_source_code(self, source_code_object, status):
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    data = self._prepare_document(source_code_object, status)
                    # Insert the document within the transaction
                    result = self.collection.insert_one(data, session=session)
                    print(f"Data inserted with record id {result.inserted_id}")
                    return result.inserted_id

            except ConnectionFailure as e:
                print("Transaction aborted due to connection failure:", e)
                session.abort_transaction()
                return None

    def save_source_codes_batch(self, source_code_objects, status):
        """
        Batch process method for saving multiple source code objects in a single transaction.
        
        Parameters:
            source_code_objects (list): List of source code objects to insert.
            status (str): Status to be assigned to all source code objects.

        Returns:
            list: List of inserted document IDs if successful, or an empty list if the transaction failed.
        """
        # Prepare the list of documents to be inserted
        documents = [self._prepare_document(obj, status) for obj in source_code_objects]

        # Start a session for transactional batch insert
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    # Perform batch insert within the transaction
                    result = self.collection.insert_many(documents, session=session)
                    print(f"Batch insert successful with record ids {result.inserted_ids}")
                    return result.inserted_ids

            except ConnectionFailure as e:
                print("Transaction aborted due to connection failure:", e)
                session.abort_transaction()
                return []

    def get_source_code(self, record_id):
        return self.collection.find_one({"_id": record_id})

    def update_source_code(self, record_id, update_data):
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    result = self.collection.update_one({"_id": record_id}, {"$set": update_data}, session=session)
                    return result.modified_count

            except ConnectionFailure as e:
                print("Transaction aborted due to connection failure:", e)
                session.abort_transaction()
                return 0

    def delete_source_code(self, record_id):
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    result = self.collection.delete_one({"_id": record_id}, session=session)
                    return result.deleted_count

            except ConnectionFailure as e:
                print("Transaction aborted due to connection failure:", e)
                session.abort_transaction()
                return 0

    def _prepare_document(self, source_code_object, status):
        """Helper method to prepare the document data for insertion."""
        return {
            "name": source_code_object.name,
            "source_code_type": "SVM",
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
