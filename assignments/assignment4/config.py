from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

# --- Connection Details ---
# It's better to store credentials separately, but for this example, they are here.
PASSWORD = quote_plus("Chershil")
CLUSTER_URI_PART = "cluster61796.pbx15s7.mongodb.net"
DB_NAME = "online_retail_db" # You can name your database anything

# --- Construct the Full URI ---
MONGO_URI = f"mongodb+srv://Cluster61796:{PASSWORD}@{CLUSTER_URI_PART}/?retryWrites=true&w=majority&appName=Cluster61796"

def get_mongo_client():
    """
    Creates and returns a MongoClient object with connection pooling enabled.
    """
    # PyMongo's MongoClient handles connection pooling automatically.
    # We can customize the pool size with maxPoolSize and minPoolSize.
    client = MongoClient(
        MONGO_URI,
        server_api=ServerApi('1'),
        serverSelectionTimeoutMS=5000, # Timeout after 5 seconds
        maxPoolSize=50,
        minPoolSize=5
    )
    return client