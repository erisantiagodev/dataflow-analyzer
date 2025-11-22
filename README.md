# Data Flow Analyzer

A Python project using FastAPI, Pandas, and NumPy for data analysis.

## Setup

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Install dependencies (already done):

```bash
pip install -r requirements.txt
```

## Running the Application

Run the FastAPI server:

```bash
python src/main.py
```

Or using uvicorn directly:

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

View the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `POST /analyze` - Analyze data items grouped by category
- `POST /stats` - Calculate statistics for numeric values
- `POST /forecast/arima` - Forecast future values using ARIMA time series model

### ARIMA Forecasting Endpoint

The `/forecast/arima` endpoint uses ARIMA (AutoRegressive Integrated Moving Average) for time series forecasting.

**Request Body:**

```json
{
  "values": [10.5, 12.3, 15.7, 14.2, 16.8, 18.4, 17.9, 20.1, 22.5, 21.8],
  "order": [1, 1, 1],
  "steps": 5
}
```

**Parameters:**

- `values` (required): List of numeric time series data (minimum 10 values)
- `order` (optional): ARIMA order parameters (p, d, q). Default: [1, 1, 1]
  - p: Number of autoregressive terms
  - d: Degree of differencing
  - q: Number of moving average terms
- `steps` (optional): Number of future time periods to forecast. Default: 10

**Response:**

```json
{
  "forecast": [24.3, 25.8, 27.1, 28.5, 29.9],
  "model_order": [1, 1, 1]
}
```

## Running Tests

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src tests/
```

Run with verbose output:

```bash
pytest -v
```

## Code Quality

Format code with Black:

```bash
black src/ tests/
```

Lint with Ruff:

```bash
ruff check src/ tests/
```

## Project Structure

```
data_flow_analyzer/
├── src/
│   ├── __init__.py
│   └── main.py          # Main FastAPI application
├── tests/
│   ├── __init__.py
│   └── test_main.py     # Test cases
├── venv/                # Virtual environment
├── requirements.txt     # Project dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **Pydantic**: Data validation using Python type annotations
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Statsmodels**: Statistical modeling and time series analysis
- **Uvicorn**: ASGI server for running FastAPI
- **Pytest**: Testing framework
- **Black**: Code formatter
- **Ruff**: Fast Python linter
