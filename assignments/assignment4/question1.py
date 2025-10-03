import pandas as pd
import sqlite3
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)

print(f"Dataset loaded: {len(df)} records")

# Clean the data
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Take first 5000 records
df = df.head(5000)
print(f"Using {len(df)} records for database")

# SQLite database setup
conn = sqlite3.connect('online_retail.db')
cursor = conn.cursor()

# Create normalized tables
cursor.execute("DROP TABLE IF EXISTS invoice_items")
cursor.execute("DROP TABLE IF EXISTS invoices")
cursor.execute("DROP TABLE IF EXISTS products")
cursor.execute("DROP TABLE IF EXISTS customers")

cursor.execute("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        country TEXT
    )
""")

cursor.execute("""
    CREATE TABLE products (
        stock_code TEXT PRIMARY KEY,
        description TEXT,
        unit_price REAL
    )
""")

cursor.execute("""
    CREATE TABLE invoices (
        invoice_no TEXT PRIMARY KEY,
        invoice_date TEXT,
        customer_id INTEGER,
        is_cancelled BOOLEAN,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
""")

cursor.execute("""
    CREATE TABLE invoice_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_no TEXT,
        stock_code TEXT,
        quantity INTEGER,
        FOREIGN KEY (invoice_no) REFERENCES invoices(invoice_no),
        FOREIGN KEY (stock_code) REFERENCES products(stock_code)
    )
""")


customers_df = df[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID'])
customers_data = list(customers_df.itertuples(index=False, name=None))
cursor.executemany("INSERT INTO customers (customer_id, country) VALUES (?, ?)", customers_data)


products_df = df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates(subset=['StockCode'])
products_data = list(products_df.itertuples(index=False, name=None))
cursor.executemany("INSERT INTO products (stock_code, description, unit_price) VALUES (?, ?, ?)", products_data)


# Insert real data - invoices
invoices_data = []
for invoice_no in df['InvoiceNo'].unique():
    invoice_df = df[df['InvoiceNo'] == invoice_no]
    is_cancelled = str(invoice_no).startswith('C')
    customer_id = int(invoice_df['CustomerID'].iloc[0])
    invoice_date = invoice_df['InvoiceDate'].iloc[0]
    invoices_data.append((str(invoice_no), str(invoice_date), customer_id, is_cancelled))

cursor.executemany("INSERT OR IGNORE INTO invoices VALUES (?, ?, ?, ?)", invoices_data)

# Insert real data - invoice items
for _, row in df.iterrows():
    cursor.execute(
        "INSERT INTO invoice_items (invoice_no, stock_code, quantity) VALUES (?, ?, ?)",
        (str(row['InvoiceNo']), row['StockCode'], int(row['Quantity']))
    )

conn.commit()

# Verify counts
cursor.execute("SELECT COUNT(*) FROM customers")
print(f"Customers: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM products")
print(f"Products: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM invoices")
print(f"Invoices: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM invoice_items")
print(f"Invoice Items: {cursor.fetchone()[0]}")

print("\n Database created successfully: online_retail.db")

# Show sample data
print("\nSample Customers:")
cursor.execute("SELECT * FROM customers LIMIT 5")
for row in cursor.fetchall():
    print(row)

print("\nSample Products:")
cursor.execute("SELECT * FROM products LIMIT 5")
for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()