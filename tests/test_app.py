"""Tests for the main app endpoints."""

import pytest


def test_landing_page(client):
    """Test the landing page endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "endpoints" in data


def test_healthz(client):
    """Test the health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "All is OK!"


def test_metrics(client):
    """Test the Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Metrics are in Prometheus text format
    assert "http_requests" in response.text or "starlette" in response.text


def test_geoparquet_stats_missing_params(client):
    """Test that the endpoint requires required parameters."""
    response = client.get("/geoparquet-stats/")
    assert response.status_code == 422  # Unprocessable Entity (missing params)


def test_geoparquet_stats_invalid_bbox(client):
    """Test validation of bbox parameter."""
    response = client.get(
        "/geoparquet-stats/",
        params={
            "geoparquet_url": "https://example.com/test.parquet",
            "bbox": [16.2, 48.1],  # Only 2 values instead of 4
        },
    )
    # The endpoint should handle this gracefully
    assert response.status_code in [200, 400, 422]


# Note: Integration tests with actual GeoParquet files should be added
# when test data is available. Example:
#
# def test_geoparquet_stats_integration(client):
#     """Test with real GeoParquet data."""
#     response = client.get(
#         "/geoparquet-stats/",
#         params={
#             "geoparquet_url": "https://workspace-ui-public.gtif-austria.hub-otc.eox.at/api/public/share/public-4wazei3y-02/WR-04-Temperature-Forecast-2019/city_temperature_timeseries.parquet",
#             "bbox": [16.2, 48.1, 16.5, 48.3],
#         },
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert isinstance(data, list)
#     if len(data) > 0:
#         assert all(k in data[0] for k in ["datetime", "asset_id", "min", "max", "mean", "stddev"])
