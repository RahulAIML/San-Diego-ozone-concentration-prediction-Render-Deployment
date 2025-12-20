# Ozone Ocean Prediction Project

This project implements a Machine Learning system to predict Ozone concentrations in San Diego using Oceanographic and Meteorological data.

## Project Structure
- `model/`: Data Science scripts (Training, Evaluation, Regime Analysis).
- `backend/`: FastAPI application to serve the model.
- `frontend/`: React application for the user interface.
- `metrics/`: Performance metrics of the trained model.

## How to Run

### Prerequisites
- Python 3.8+
- Node.js 16+

### 1. Start the Backend
The backend serves the prediction API.
Open a terminal in the root folder (`d:\Ozone_Project_7th_dec`) and run:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*Wait for "Application startup complete".*

### 2. Start the Frontend
The frontend is the web interface.
Open a **new** terminal in the `frontend` folder (`d:\Ozone_Project_7th_dec\frontend`) and run:
```bash
npm run dev
```
Then open your browser and navigate to the URL shown (usually `http://localhost:5173`).

## Features
- **Interactive Dashboard**: Adjust Temperature, Wind, and Upwelling (CUTI) to see real-time predictions.
- **Regime Detection**: The system identifies if the current conditions match a "Marine Dominated" or "Stagnant" regime.
