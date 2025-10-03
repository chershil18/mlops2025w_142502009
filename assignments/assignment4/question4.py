# transaction_centric_local.py
import pandas as pd
from pymongo.errors import ConnectionFailure, OperationFailure

# Import the function to get a client and DB_NAME from config_local
try:
    from config_localatlas import get_mongo_client, DB_NAME
except ImportError:
    print("Error: Could not import from config_local.py. Make sure the file exists.")
    exit()

def load_and_clean_data(url: str, record_count: int) -> pd.DataFrame:
    print("Step 1: Loading and cleaning the dataset...")
    try:
        df = pd.read_excel(url)
        df.dropna(subset=['CustomerID', 'Description'], inplace=True)
        df['CustomerID'] = df['CustomerID'].astype(int)
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df_sample = df.head(record_count)
        print(f"Dataset ready. Using {len(df_sample)} records.\n")
        return df_sample
    except Exception as e:
        print(f"Error during data loading/cleaning: {e}")
        exit()

def implement_transaction_centric(db, df: pd.DataFrame):
    print("Starting Transaction-Centric Model...")
    transactions_collection = db['transactions']
    try:
        transactions_collection.drop()
        
        invoices = df.groupby('InvoiceNo')
        transaction_docs = []
        for invoice_no, invoice_df in invoices:
            customer_info = invoice_df.iloc[0]
            items_list = [
                {
                    "stock_code": row['StockCode'],
                    "description": row['Description'],
                    "quantity": int(row['Quantity']),
                    "unit_price": float(row['UnitPrice'])
                } for _, row in invoice_df.iterrows()
            ]
            
            transaction_doc = {
                "_id": str(invoice_no),
                "invoice_date": customer_info['InvoiceDate'],
                "is_cancelled": str(invoice_no).startswith('C'),
                "customer": {
                    "id": int(customer_info['CustomerID']),
                    "country": customer_info['Country']
                },
                "items": items_list
            }
            transaction_docs.append(transaction_doc)
            
        if transaction_docs:
            transactions_collection.insert_many(transaction_docs)
        
        print(f"Successfully inserted {len(transaction_docs)} documents into 'transactions' collection.")
        print(f"Total documents now: {transactions_collection.count_documents({})}")

        # Fetch one example document to verify
        sample_doc = transactions_collection.find_one()
        print("\nSample document from transactions collection:")
        print(sample_doc)

    except OperationFailure as e:
        print(f"Database operation failed (Transaction-Centric): {e.details}")
    except Exception as e:
        print(f"An unexpected error occurred (Transaction-Centric): {e}")

if __name__ == "__main__":
    DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    RECORDS_TO_PROCESS = 2000

    df_sample = load_and_clean_data(DATASET_URL, RECORDS_TO_PROCESS)

    client = None
    try:
        print("\nStep 2: Connecting to MongoDB Atlas Local Deployment...")
        client = get_mongo_client()
        
        # Ping server
        client.admin.command('ping')
        print("MongoDB connection successful.")
        db = client[DB_NAME]

        # Insert into transaction-centric model
        implement_transaction_centric(db, df_sample)

    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if client:
            client.close()
            print("\nStep 3: MongoDB connection closed.")
