from pymongo import MongoClient

DB_NAME = "online_retail"

def get_mongo_client():
    return MongoClient(
        "mongodb://localhost:60115/?directConnection=true",
        maxPoolSize=50,
        minPoolSize=5,
        serverSelectionTimeoutMS=5000
    )
