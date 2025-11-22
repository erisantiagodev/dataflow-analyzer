import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import ArimaForecast, ArimaRequest, app, DataItem, DataStats

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["version"] == "1.0.0"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_analyze_data():
    """Test data analysis endpoint."""
    test_data = [
        {"name": "item1", "value": 10.0, "category": "A"},
        {"name": "item2", "value": 20.0, "category": "A"},
        {"name": "item3", "value": 15.0, "category": "B"},
    ]

    response = client.post("/analyze", json=test_data)
    assert response.status_code == 200

    data = response.json()
    assert "analysis" in data
    assert "A" in data["analysis"]
    assert "B" in data["analysis"]
    assert data["analysis"]["A"]["count"] == 2
    assert data["analysis"]["B"]["count"] == 1


def test_calculate_stats():
    """Test statistics calculation endpoint."""
    test_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    response = client.post("/stats", json=test_values)
    assert response.status_code == 200

    data = response.json()
    assert "mean" in data
    assert "median" in data
    assert "std" in data
    assert "count" in data
    assert data["mean"] == 3.0
    assert data["median"] == 3.0
    assert data["count"] == 5


def test_forecast_arima():
    """Test ARIMA Forecase calculations endpoint."""
    arima_test_values = {
        "values": [
            10.5,
            12.3,
            15.7,
            14.2,
            16.8,
            18.4,
            17.9,
            20.1,
            22.5,
            21.8,
            23.4,
            25.1,
        ],
        "order": [2, 1, 3],
        "steps": 5,
    }

    response = client.post("/forecast/arima", json=arima_test_values)
    assert response.status_code == 200

    data = response.json()
    assert "forecast" in data
    assert "model_order" in data
    assert len(data["forecast"]) == 5
    assert data["model_order"] == [2, 1, 3]
    assert data["forecast"][0] == pytest.approx(25.703259987406778, rel=1e-2)
    assert data["forecast"][1] == pytest.approx(27.083762122139053, rel=1e-2)
    assert data["forecast"][2] == pytest.approx(27.88086961703199, rel=1e-2)
    assert data["forecast"][3] == pytest.approx(29.12766641446821, rel=1e-2)
    assert data["forecast"][4] == pytest.approx(29.994117400760047, rel=1e-2)


def test_data_item_model():
    """Test DataItem Pydantic model."""
    item = DataItem(name="test", value=10.5, category="A")
    assert item.name == "test"
    assert item.value == 10.5
    assert item.category == "A"


def test_data_stats_model():
    """Test DataStats Pydantic model."""
    stats = DataStats(mean=5.0, median=5.0, std=1.5, count=10)
    assert stats.mean == 5.0
    assert stats.median == 5.0
    assert stats.std == 1.5
    assert stats.count == 10


def test_arima_request_model():
    """Test ArimaRequest Pydantic model."""
    arima_request = ArimaRequest(
        values=[3.2, 1.9, 6.5, 2.9, 5.9, 23.12, 61.20, 99.99, 100.21, 21.22],
        order=(1, 1, 1),
        steps=5,
    )
    assert arima_request.values == [
        3.2,
        1.9,
        6.5,
        2.9,
        5.9,
        23.12,
        61.20,
        99.99,
        100.21,
        21.22,
    ]
    assert arima_request.order == (1, 1, 1)
    assert arima_request.steps == 5


def test_arima_request_model_fail():
    """Test ArimaRequest Pydantic model."""
    arima_request = ArimaRequest(
        values=[3.2, 1.9, 6.5, 2.9],
        order=(1, 1, 1),
        steps=5,
    )

    response = client.post("/forecast/arima", json=arima_request.model_dump())
    assert response.status_code == 400
    assert response.json() == {"detail": "At least 10 values are required for ARIMA."}


def test_arima_forecast_model():
    """Test Arima Forecast Pydantic Model."""
    arima_forecast = ArimaForecast(
        forecast=[3.2, 1.9, 6.5, 2.9, 5.9, 23.12, 61.20, 99.99, 100.21, 21.22],
        model_order=[2, 1, 2],
    )
    assert arima_forecast.forecast == [
        3.2,
        1.9,
        6.5,
        2.9,
        5.9,
        23.12,
        61.20,
        99.99,
        100.21,
        21.22,
    ]
    assert arima_forecast.model_order == (2, 1, 2)
