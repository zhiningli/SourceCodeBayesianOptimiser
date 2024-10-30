from data.source_codes.source_code_builder import SVMSourceCode
from datetime import datetime
from pymongo import MongoClient

SVM_iris_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1597).buildKernel('rbf').buildC(0.5).buildGamma("scale").buildCoef0("0.5").build()  

def saveSourceCodeToMongoDB(source_code_object, status):
    
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

    client = MongoClient("mongodb://localhost:27017/")

    db = client["source_code_database"]
    collection = db["source_code_collection"]

    result = collection.insert_one(data)
    print(f"Data inserted with record id {result.inserted_id}")

    retrieved_data = collection.find_one({"_id": result.inserted_id})

    print("retrieved data: ", retrieved_data)

saveSourceCodeToMongoDB(SVM_iris_dataset, "generated_from_template")