import pandas as pd
from pymongo.errors import ConnectionFailure, OperationFailure

# Import the function to get a client and the DB_NAME from your config file
try:
    from config import get_mongo_client, DB_NAME
except ImportError:
    print("Error: Could not import from config.py. Make sure the file exists and is in the same directory.")
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
    print("Starting Approach 1: Transaction-Centric Model")
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
        print(f"Total documents: {transactions_collection.count_documents({})}")

    except OperationFailure as e:
        print(f"Database operation failed (Transaction-Centric): {e.details}")
    except Exception as e:
        print(f"An unexpected error occurred (Transaction-Centric): {e}")

def implement_customer_centric(db, df: pd.DataFrame):
    print("\nStarting Approach 2: Customer-Centric Model")
    customers_collection = db['customers']
    try:
        customers_collection.drop()

        customers = df.groupby('CustomerID')
        customer_docs = []
        for customer_id, customer_df in customers:
            country = customer_df.iloc[0]['Country']
            invoices_list = []
            
            for invoice_no, invoice_df in customer_df.groupby('InvoiceNo'):
                items_list = [
                    {
                        "stock_code": row['StockCode'],
                        "description": row['Description'],
                        "quantity": int(row['Quantity']),
                        "unit_price": float(row['UnitPrice'])
                    } for _, row in invoice_df.iterrows()
                ]
                invoices_list.append({
                    "invoice_no": str(invoice_no),
                    "invoice_date": invoice_df.iloc[0]['InvoiceDate'],
                    "is_cancelled": str(invoice_no).startswith('C'),
                    "items": items_list
                })
            
            customer_doc = {
                "_id": int(customer_id),
                "country": country,
                "invoices": invoices_list
            }
            customer_docs.append(customer_doc)

        if customer_docs:
            customers_collection.insert_many(customer_docs)

        print(f"Successfully inserted {len(customer_docs)} documents into 'customers' collection.")
        print(f"Total documents: {customers_collection.count_documents({})}")
    
    except OperationFailure as e:
        print(f"Database operation failed (Customer-Centric): {e.details}")
    except Exception as e:
        print(f"An unexpected error occurred (Customer-Centric): {e}")

if __name__ == "__main__":
    DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    RECORDS_TO_PROCESS = 2000

    df_sample = load_and_clean_data(DATASET_URL, RECORDS_TO_PROCESS)

    client = None
    try:
        print("\nStep 2: Connecting to MongoDB Atlas...")
        client = get_mongo_client()
        
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print("MongoDB connection successful.")
        db = client[DB_NAME]

        # Run both data modeling implementations
        implement_transaction_centric(db, df_sample)
        implement_customer_centric(db, df_sample)

    except ConnectionFailure as e:
        print(f"MongoDB connection failed: Could not connect to server. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if client:
            client.close()
            print("\nStep 3: MongoDB connection closed.")