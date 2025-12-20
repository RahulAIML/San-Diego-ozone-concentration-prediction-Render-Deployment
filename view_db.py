import sqlite3
import pandas as pd
import json

DB_PATH = 'd:/Ozone_Project_7th_dec/django_backend/db.sqlite3'

def view_logs():
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM api_predictionlog ORDER BY created_at DESC LIMIT 10"
        df = pd.read_sql_query(query, conn)
        
        print(f"\n--- Last 10 Predictions from {DB_PATH} ---\n")
        
        if df.empty:
            print("No logs found.")
            return

        for index, row in df.iterrows():
            print(f"ID: {row['id']}")
            print(f"Time: {row['created_at']}")
            print(f"Input: {row['input_data'][:100]}...") # Truncate
            print(f"Output: {row['predicted_output']}")
            print("-" * 50)
            
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    view_logs()
