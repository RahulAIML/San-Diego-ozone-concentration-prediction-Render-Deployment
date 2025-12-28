# 🚀 Ozone Ocean Predictor - Deployment Guide

**Repository**: [https://github.com/RahulAIML/San-Diago-ozone-concentration-prediction-Render-Deployment](https://github.com/RahulAIML/San-Diago-ozone-concentration-prediction-Render-Deployment)

This guide outlines how to deploy the fullstack application (Django Backend + React Frontend) to **Render**.

---

## 🏗️ Architecture Overview

The project is configured as a Monorepo containing:
1.  **Backend (`django_backend/`)**: Python Django API using `Gunicorn` and `Whitenoise`.
2.  **Frontend (`frontend/`)**: React + Vite application, served as a Static Site.
3.  **Database**: SQLite (Default for this demo). *Note: Ephemeral on Render.*

---

## ⚡ Deployment Option 1: Render Blueprints (Recommended)

This project includes a `render.yaml` file that automates the entire setup.

1.  **Sign Up / Login** to [Render.com](https://render.com/).
2.  **Creating the Blueprint**:
    *   Click **New +** and select **Blueprint**.
    *   Connect your GitHub account and select the repository: `San-Diago-ozone-concentration-prediction-Render-Deployment`.
    *   Give the blueprint a name (e.g., `ozone-predictor`).
3.  **Review Services**:
    *   Render will detect two services from `render.yaml`:
        *   `ozone-backend` (Web Service)
        *   `ozone-frontend` (Static Site)
4.  **Deploy**:
    *   Click **Apply**.
    *   Render will now build both services in parallel.
5.  **Environment Variables**:
    *   The Blueprint automatically handles `VITE_API_URL` linking the frontend to the backend.
    *   It also generates a secure `SECRET_KEY` for Django.

---

## 🛠️ Deployment Option 2: Manual Setup

If you prefer to configure services strictly by hand:

### 1. Backend Service
*   **Type**: Web Service
*   **Root Directory**: `django_backend`
*   **Runtime**: Python 3
*   **Build Command**: `./build.sh`
*   **Start Command**: `gunicorn ozone_api.wsgi:application`
*   **Environment Variables**:
    *   `PYTHON_VERSION`: `3.11.9`
    *   `SECRET_KEY`: (Generate a secure random string)
    *   `DEBUG`: `False`
    *   `ALLOWED_HOSTS`: `*`

### 2. Frontend Service
*   **Type**: Static Site
*   **Root Directory**: `frontend`
*   **Build Command**: `npm install && npm run build`
*   **Publish Directory**: `dist`
*   **Environment Variables**:
    *   `VITE_API_URL`: (Enter the URL of your deployed backend, e.g., `https://ozone-backend.onrender.com`)

---

## ⚠️ Important Considerations

### Database Persistence (SQLite)
*   **Current State**: The app uses `db.sqlite3`.
*   **Risk**: On Render, the file system is **ephemeral**. This means **all data (prediction logs) will be deleted** whenever the application is redeployed or restarted (which happens automatically).
*   **Production Solution**: To have persistent data, you should switch to a managed PostgreSQL database.
    1.  Create a PostgreSQL database on Render.
    2.  Update `settings.py` to use `dj-database-url`.
    3.  Add the `DATABASE_URL` environment variable to your Backend service.

### CORS
*   Currently, `CORS_ALLOW_ALL_ORIGINS = True` is set in `settings.py` to ensure the frontend can easily communicate with the backend.
*   For stricter security in the future, set this to only allow your frontend's specific domain.

---

## 🧪 Verification

Once deployed:
1.  Open your **Frontend URL** (provided by Render).
2.  Adjust the sliders (Temperature, CUTI, etc.).
3.  Click **Predict Ozone Level**.
4.  **Success Criteria**:
    *   You see a prediction result (e.g., "Good", "Moderate").
    *   The "Database History Log" table at the bottom updates with your new entry.
