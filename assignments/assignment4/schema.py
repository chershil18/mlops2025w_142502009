transaction_centric_schema = {
    "_id": "<string> (InvoiceNo)",
    "invoice_date": "<datetime>",
    "is_cancelled": "<boolean>",
    "customer": {
        "id": "<int> (CustomerID)",
        "country": "<string>"
    },
    "items": [
        {
            "stock_code": "<string>",
            "description": "<string>",
            "quantity": "<int>",
            "unit_price": "<float>"
        }
    ]
}

customer_centric_schema = {
    "_id": "<int> (CustomerID)",
    "country": "<string>",
    "invoices": [
        {
            "invoice_no": "<string>",
            "invoice_date": "<datetime>",
            "is_cancelled": "<boolean>",
            "items": [
                {
                    "stock_code": "<string>",
                    "description": "<string>",
                    "quantity": "<int>",
                    "unit_price": "<float>"
                }
            ]
        }
    ]
}