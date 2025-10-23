# COG Statistics Endpoint

A FastAPI microservice for extracting statistical evaluations (min, max, mean, stddev) from Cloud-Optimized GeoTIFF (COG) timeseries referenced in GeoParquet STAC collections.

## Features

- Extract statistics from COG timeseries for specified bounding boxes
- Parallel processing for improved performance
- Support for GeoParquet STAC collections
- Prometheus metrics endpoint
- Structured logging with correlation IDs
- Docker-based deployment

## Quick Start

### Using Docker Compose

1. Create a `.env` file (optional for S3 access):
```bash
AWS_S3_ENDPOINT=s3.amazonaws.com
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

2. Build and start the service:
```bash
docker-compose up --build
```

3. Access the API at http://localhost:3000

### API Usage

#### Get Statistics from GeoParquet

```bash
curl -X GET "http://localhost:3000/geoparquet-stats/?geoparquet_url=https://example.com/data.parquet&bbox=16.2&bbox=48.1&bbox=16.5&bbox=48.3"
```

**Parameters:**
- `geoparquet_url` (required): URL to the GeoParquet file containing STAC items
- `bbox` (required): Bounding box as 4 values [minx, miny, maxx, maxy] in EPSG:4326
- `start_date` (optional): Filter items from this date
- `end_date` (optional): Filter items until this date
- `max_workers` (optional): Number of parallel workers (default: 5)

**Response:**
```json
[
  {
    "datetime": "2019-01-01T00:00:00",
    "asset_id": "temperature",
    "min": 273.15,
    "max": 298.45,
    "mean": 285.32,
    "stddev": 5.67
  }
]
```

### Other Endpoints

- `GET /` - API landing page
- `GET /healthz` - Health check
- `GET /metrics` - Prometheus metrics

## Development

### Run Tests

```bash
make test
```

### Lint Code

```bash
make lint
```

### Format Code

```bash
make format
```

## Architecture

Based on the [cmems-backend](https://github.com/eoxhub-workspaces/cmems-backend) structure:

- **FastAPI** - Web framework
- **Gunicorn + Uvicorn** - ASGI server
- **Rioxarray** - COG processing
- **GeoPandas** - GeoParquet handling
- **Xarray** - Statistics computation
- **Prometheus** - Metrics
- **Structlog** - Structured logging

## License

MIT License
