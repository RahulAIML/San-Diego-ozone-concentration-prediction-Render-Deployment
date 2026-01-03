# Deployment Guide - Render.com

This project is configured for easy deployment on Render.com as a Web Service.

## Configuration
The project includes a `render.yaml` file that defines the service.

- **Service Name**: `ozone-predictor`
- **Environment**: Python
- **Build Command**: `npm install --prefix frontend && npm run build --prefix frontend && pip install -r requirements.txt`
  - This builds the React frontend and installs Python dependencies in one step.
- **Start Command**: `gunicorn app:app`
  - Starts the Flask production server.

## Steps to Deploy

1. **Push to GitHub**: Ensure your latest code is pushed to your GitHub repository.
2. **Login to Render**: Go to [dashboard.render.com](https://dashboard.render.com).
3. **Create New Web Service**:
   - Connect your GitHub repository.
   - Render should automatically detect `render.yaml`.
   - If not, manually select **"Web Service"** and use the settings above.
4. **Environment Variables**:
   - `PYTHON_VERSION`: `3.9.0` (or similar)
   - `VITE_API_URL`: Leave empty or set to `/` since the frontend is served by the same backend.
   - **`DATABASE_URL`**: Connection string for your PostgreSQL database (e.g., `postgres://user:pass@host:port/dbname`).
     - If not set, the app defaults to an ephemeral SQLite database.

## Troubleshooting
- **Build Fails**: Check if `npm run build` works locally. Ensure `frontend/dist` is created.
- **Application Error**: Check Render logs. Ensure `model_artifacts/` are committed and present.
- **Database**:
  - **SQLite (Default)**: If `DATABASE_URL` is omitted, `database.db` is used. This is ephemeral on Render (data is lost on redeploy).
  - **PostgreSQL**: Set `DATABASE_URL` for persistent storage. Ensure your Postgres service is running and accessible (internal URL for Render-to-Render communication).

## Directory Structure Verification
Ensure your repo looks like this for the build to work:
```
/
├── app.py
├── requirements.txt
├── render.yaml
├── model_artifacts/ ...
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── ...
└── ...
```

