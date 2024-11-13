from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from enum import Enum

class BO_evaluation_db:
    def __init__(self, db_url="mongodb://")