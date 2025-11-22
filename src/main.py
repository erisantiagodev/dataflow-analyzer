from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI(title="Data Flow Analyzer", version="1.0.0")


class DataItem(BaseModel):
    name: str
    value: float
    category: str


class DataStats(BaseModel):
    mean: float
    median: float
    std: float
    count: int


class ArimaRequest(BaseModel):
    values: list[float]
    order: tuple[int, int, int] = (1, 1, 1)  # (p, d, q) parameters
    steps: int = 10  # number of forecast steps


class ArimaForecast(BaseModel):
    forecast: list[float]
    model_order: tuple[int, int, int]


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Welcome to Data Flow Analyzer API",
        "version": "1.0.0",
        "endpoints": ["/", "/health", "/analyze", "/stats", "/forecast/arima"],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/analyze", response_model=dict)
async def analyze_data(items: list[DataItem]):
    """
    Analyze a list of data items and return basic statistics.

    Args:
        items: List of DataItem objects containing name, value, and category

    Returns:
        Dictionary containing analysis results grouped by category
    """
    # Convert to pandas DataFrame
    df = pd.DataFrame([item.model_dump() for item in items])

    # Group by category and calculate statistics
    results = {}
    for category in df["category"].unique():
        category_data = df[df["category"] == category]["value"]
        results[category] = {
            "mean": float(np.mean(category_data)),
            "median": float(np.median(category_data)),
            "std": float(np.std(category_data)),
            "count": len(category_data),
            "sum": float(np.sum(category_data)),
        }

    return {"analysis": results}


@app.post("/stats", response_model=DataStats)
async def calculate_stats(values: list[float]):
    """
    Calculate statistics for a list of numeric values.

    Args:
        values: List of numeric values

    Returns:
        DataStats object with mean, median, std, and count
    """
    arr = np.array(values)

    return DataStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        count=len(arr),
    )


@app.post("/forecast/arima", response_model=ArimaForecast)
async def forecast_arima(request: ArimaRequest):
    """
    Forecast future values using ARIMA model.

    Args:
        request: ArimaRequest with values, order (p,d,q), and forecast steps

    Returns:
        ArimaForecast object with forecasted values and model parameters
    """

    if len(request.values) < 10:
        raise HTTPException(
            status_code=400, detail="At least 10 values are required for ARIMA."
        )

    try:
        model = ARIMA(request.values, order=request.order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=request.steps)

        return ArimaForecast(forecast=forecast.tolist(), model_order=request.order)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ARIMA model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
