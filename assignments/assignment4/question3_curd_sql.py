import sqlite3
import time

# Use absolute path to your DB
DB_PATH = '/Users/kalpanapullagura/Documents/mlops2025w_142502009/online_retail.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Verify tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in the database:", cursor.fetchall())

# -------- READ Operation --------
start = time.time()
cursor.execute("SELECT * FROM invoice_items WHERE quantity > 5")
rows = cursor.fetchall()
end = time.time()
print(f"Read operation took: {end - start:.4f} seconds, rows fetched: {len(rows)}")

# -------- UPDATE Operation --------
start = time.time()
cursor.execute("UPDATE products SET unit_price = unit_price * 1.1 WHERE unit_price < 10")
conn.commit()
end = time.time()
print(f"Update operation took: {end - start:.4f} seconds")

# -------- DELETE Operation --------
start = time.time()
cursor.execute("DELETE FROM invoice_items WHERE quantity = 1")
conn.commit()
end = time.time()
print(f"Delete operation took: {end - start:.4f} seconds")

cursor.close()
conn.close()
