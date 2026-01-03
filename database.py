import os
import sqlite3
import json
import logging
from datetime import datetime

# Optional: Try importing psycopg2 for Postgres
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

# --- Configuration ---
# Look for DATABASE_URL environment variable (Render sets this for Postgres)
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db_connection():
    """
    Returns a database connection.
    If DATABASE_URL is set and psycopg2 is available, connects to Postgres.
    Otherwise, defaults to local SQLite.
    """
    if DATABASE_URL and psycopg2:
        try:
            conn = psycopg2.connect(DATABASE_URL)
            return conn, 'postgres'
        except Exception as e:
            logging.error(f"Failed to connect to Postgres: {e}")
            # Fallback could happen here, or just raise. For now, let's raise/fail if config exists but fails.
            raise e
    else:
        # SQLite fallback
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DB_PATH = os.path.join(BASE_DIR, 'database.db')
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn, 'sqlite'

def init_db():
    """
    Initializes the database table if it doesn't exist.
    """
    conn, db_type = get_db_connection()
    try:
        if db_type == 'postgres':
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        input_data TEXT,
                        predicted_output TEXT
                    )
                ''')
            conn.commit()
        else:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_data TEXT,
                    predicted_output TEXT
                )
            ''')
            conn.commit()
    except Exception as e:
        logging.error(f"Error checking/creating table: {e}")
    finally:
        conn.close()

def insert_log(input_data_dict, output_data_dict):
    """
    Inserts a prediction log into the database.
    Returns the log_id or -1 on failure.
    """
    conn, db_type = get_db_connection()
    log_id = -1
    try:
        input_json = json.dumps(input_data_dict)
        output_json = json.dumps(output_data_dict)

        if db_type == 'postgres':
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO prediction_logs (input_data, predicted_output) VALUES (%s, %s) RETURNING id',
                    (input_json, output_json)
                )
                log_id = cur.fetchone()[0]
            conn.commit()
        else:
            c = conn.cursor()
            c.execute(
                'INSERT INTO prediction_logs (input_data, predicted_output) VALUES (?, ?)',
                (input_json, output_json)
            )
            log_id = c.lastrowid
            conn.commit()
    except Exception as e:
        logging.error(f"DB Insert Error: {e}")
    finally:
        conn.close()
    
    return log_id

def fetch_logs(limit=50):
    """
    Fetches the most recent logs.
    Returns a list of dictionaries.
    """
    conn, db_type = get_db_connection()
    data = []
    try:
        if db_type == 'postgres':
            # Use RealDictCursor to get dict-like objects
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT * FROM prediction_logs ORDER BY created_at DESC LIMIT %s', (limit,))
                rows = cur.fetchall()
        else:
            # SQLite
            c = conn.cursor()
            c.execute('SELECT * FROM prediction_logs ORDER BY created_at DESC LIMIT ?', (limit,))
            rows = c.fetchall()
        
        # Normalize rows to pure dicts
        for row in rows:
            # Handle potential differences in row access
            try:
                # Common fields
                r_id = row['id']
                r_created = row['created_at']
                r_input = row['input_data']
                r_output = row['predicted_output']
            except IndexError:
                 # Fallback if row factory failed (unlikely)
                 r_id = row[0]
                 # ... skipping complex fallback for now, assuming row factory works
            
            # Parse JSON if it's string
            try:
                in_d = json.loads(r_input) if isinstance(r_input, str) else r_input
            except:
                in_d = r_input
            
            try:
                out_d = json.loads(r_output) if isinstance(r_output, str) else r_output
            except:
                out_d = r_output

            data.append({
                "id": r_id,
                "created_at": r_created,
                "input": in_d,
                "output": out_d
            })

    except Exception as e:
        logging.error(f"DB Fetch Error: {e}")
    finally:
        conn.close()
    
    return data
