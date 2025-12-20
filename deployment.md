# Deployment and Demo Guide

## 1. Project Overview
This Django API provides endpoints to receive data, process a "prediction" (currently a mock demo), and store the input/output pairs in a SQLite database for audit and analysis.

**Project Location:** `d:\Ozone_Project_7th_dec\django_backend`

## 2. Running Locally for Demo

### Prerequisites
*   Python 3.10+
*   Dependencies installed (`pip install -r requirements.txt`)

### Steps
1.  **Navigate to the backend directory**:
    ```powershell
    cd d:\Ozone_Project_7th_dec\django_backend
    ```

2.  **Ensure migrations are applied** (Database setup):
    ```powershell
    python manage.py migrate
    ```

3.  **Start the server**:
    ```powershell
    python manage.py runserver
    ```
    The server will run at `http://127.0.0.1:8000/`.

## 3. API Endpoints

### POST /api/predict/
**Description**: Takes arbitrary JSON input, generates a mock prediction, preserves both in the database, and returns the result.

**URL**: `http://127.0.0.1:8000/api/predict/`

**Body (JSON)**:
```json
{
    "feature_1": 12.5,
    "feature_2": "high",
    "location": "Station_1"
}
```

**Response**:
```json
{
    "status": "success",
    "input_recieved": { ... },
    "prediction": {
        "prediction_score": 0.95,
        "class": "Ozone Normal",
        "mock_note": "This is a dummy prediction for demo purposes."
    },
    "log_id": 1,
    "message": "Prediction saved to database."
}
```

### GET /api/health/
**Description**: Check if server is running.
**URL**: `http://127.0.0.1:8000/api/health/`

## 4. Verifying Database Storage (Demo)
You can verify that data is being stored using the Django Admin shell or by inspecting the `db.sqlite3` file directly.

**Using Django Shell**:
```powershell
python manage.py shell
```
Inside the shell:
```python
from api.models import PredictionLog
# See the last 5 entries
for log in PredictionLog.objects.all().order_by('-id')[:5]:
    print(f"ID: {log.id} | Input: {log.input_data} | Output: {log.predicted_output}")
```

## 5. AWS Deployment (Elastic Beanstalk)

1.  **Install EB CLI**: `pip install awsebcli`
2.  **Initialize**: `eb init -p python-3.11 ozone-api`
3.  **Deploy**: `eb create ozone-api-env`
    *   The project includes a `Procfile` configured for Gunicorn.
    *   `settings.py` is configured to allow all hosts and serve static files.

## 6. AWS Deployment (App Runner)
1.  Push code to GitHub.
2.  Connect GitHub repo to AWS App Runner.
3.  **Build Command**: `pip install -r requirements.txt`
4.  **Start Command**: `gunicorn ozone_api.wsgi:application`
