import time
from config import get_mongo_client, DB_NAME

client = get_mongo_client()
db = client[DB_NAME]
collection = db['transactions']

# Read performance
start = time.time()
rows = list(collection.find({"items.quantity": {"$gt": 5}}))
end = time.time()
print(f"Read operation took: {end - start:.4f} seconds, documents fetched: {len(rows)}")

# Update performance
start = time.time()
collection.update_many({"items.unit_price": {"$lt": 10}}, {"$mul": {"items.$[].unit_price": 1.1}})
end = time.time()
print(f"Update operation took: {end - start:.4f} seconds")

# Delete performance
start = time.time()
collection.delete_many({"items.quantity": 1})
end = time.time()
print(f"Delete operation took: {end - start:.4f} seconds")

client.close()
