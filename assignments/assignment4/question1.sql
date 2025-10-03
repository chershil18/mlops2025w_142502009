

DROP TABLE IF EXISTS invoice_items;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS retail_staging;

-- Create normalized tables
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    country VARCHAR(255)
);

CREATE TABLE products (
    stock_code VARCHAR(255) PRIMARY KEY,
    description TEXT,
    unit_price DECIMAL(10, 2)
);

CREATE TABLE invoices (
    invoice_no VARCHAR(255) PRIMARY KEY,
    invoice_date TIMESTAMP,
    customer_id INTEGER,
    is_cancelled BOOLEAN,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE invoice_items (
    id INT AUTO_INCREMENT PRIMARY KEY, 
    invoice_no VARCHAR(255),
    stock_code VARCHAR(255),
    quantity INTEGER,
    FOREIGN KEY (invoice_no) REFERENCES invoices(invoice_no),
    FOREIGN KEY (stock_code) REFERENCES products(stock_code)
);

-- Create a temporary table to load the raw CSV data into
CREATE TABLE retail_staging (
    InvoiceNo VARCHAR(255),
    StockCode VARCHAR(255),
    Description TEXT,
    Quantity INTEGER,
    InvoiceDate VARCHAR(255), -- Load as text first, then convert
    UnitPrice DECIMAL(10, 2),
    CustomerID INTEGER,
    Country VARCHAR(255)
);



LOAD DATA LOCAL INFILE '/tmp/online_retail.csv'
INTO TABLE retail_staging
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;






-- Insert unique customers
INSERT INTO customers (customer_id, country)
SELECT
    CustomerID,
    Country
FROM retail_staging
GROUP BY CustomerID, Country;

-- Insert unique products
INSERT INTO products (stock_code, description, unit_price)
SELECT
    StockCode,
    Description,
    UnitPrice
FROM retail_staging
GROUP BY StockCode, Description, UnitPrice;

-- Insert unique invoices
INSERT INTO invoices (invoice_no, invoice_date, customer_id, is_cancelled)
SELECT
    InvoiceNo,
    STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'), -- Use TO_TIMESTAMP for PostgreSQL
    CustomerID,
    CASE WHEN LEFT(InvoiceNo, 1) = 'C' THEN TRUE ELSE FALSE END
FROM retail_staging
GROUP BY InvoiceNo, InvoiceDate, CustomerID;

-- Insert all invoice items
INSERT INTO invoice_items (invoice_no, stock_code, quantity)
SELECT
    InvoiceNo,
    StockCode,
    Quantity
FROM retail_staging;


-- Verify counts
SELECT 'Customers Count:', COUNT(*) FROM customers;
SELECT 'Products Count:', COUNT(*) FROM products;
SELECT 'Invoices Count:', COUNT(*) FROM invoices;
SELECT 'Invoice Items Count:', COUNT(*) FROM invoice_items;

-- Show sample data
SELECT * FROM customers LIMIT 5; -- Use TOP 5 for SQL Server
SELECT * FROM products LIMIT 5;  -- Use TOP 5 for SQL Server

-- Drop the temporary staging table
DROP TABLE retail_staging;

-- End of script