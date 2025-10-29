import concurrent.futures
import datetime
import functools
import logging
import math
import time
from typing import Annotated, Any

import fsspec
import geopandas as gpd
import numpy as np
import pydantic
import rioxarray
import structlog
import xarray as xr
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, HTTPException, Query, Request
from shapely.geometry import box
from starlette_exporter import PrometheusMiddleware, handle_metrics

from pydantic import BaseModel, Field, confloat
from typing import List, Optional
import datetime

FloatNoNan = confloat(allow_inf_nan=False)

import rioxarray

logging.basicConfig(level=logging.INFO)

logger = structlog.getLogger()

# Configure rioxarray for cloud-optimized access
import odc.stac

odc.stac.configure_rio(
    cloud_defaults=True,
    verbose=True,
)

app = FastAPI(
    title="COG Statistics Endpoint",
    description="Extract statistical evaluations from COG timeseries in GeoParquet STAC collections",
    version="1.0.0",
)

# Add middleware
app.add_middleware(PrometheusMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_route("/metrics", handle_metrics)


def nan_to_none(a: float) -> float | None:
    """Convert NaN and Inf values to None for JSON serialization."""
    if math.isnan(a) or math.isinf(a):
        return None
    return a


FloatNoNan = Annotated[float | None, pydantic.BeforeValidator(nan_to_none)]


class BandStats(BaseModel):
    """Statistics for an individual band."""
    band: str = Field(..., description="Band name or index")
    index: Optional[int] = Field(None, description="Band index (0-based)")
    min: FloatNoNan
    max: FloatNoNan
    mean: FloatNoNan
    stddev: FloatNoNan
    valid_pixels: Optional[int] = Field(None, description="Count of valid (non-nodata) pixels")


class GeoParquetStatsItem(BaseModel):
    """Response model for COG statistics, including per-band stats."""
    datetime: datetime.datetime
    asset_id: str
    # Detailed band stats
    bands: List[BandStats]


@app.get("/")
async def landing_page(request: Request):
    """API landing page."""
    return {
        "title": "COG Statistics Endpoint",
        "description": "Extract statistical evaluations from COG timeseries",
        "endpoints": {
            "/geoparquet-stats/": "Get statistics from GeoParquet STAC collection",
            "/healthz": "Health check",
            "/metrics": "Prometheus metrics",
        },
    }


@app.get("/healthz")
def healthz():
    """Health check endpoint."""
    return {"message": "All is OK!"}


@app.get("/inspect-geoparquet/")
def inspect_geoparquet(geoparquet_url: str, limit: int = 5):
    """
    Inspect the contents of a GeoParquet file to see what assets it contains.

    Args:
        geoparquet_url: URL to the GeoParquet file
        limit: Number of rows to return (default: 5)

    Returns:
        Sample of the GeoParquet data
    """
    try:
        # Load GeoParquet
        if geoparquet_url.startswith("https://") or geoparquet_url.startswith("http://"):
            with fsspec.open(geoparquet_url, mode='rb') as f:
                gdf = gpd.read_parquet(f)
        else:
            gdf = gpd.read_parquet(geoparquet_url)

        # Get basic info
        result = {
            "total_rows": len(gdf),
            "columns": list(gdf.columns),
            "crs": str(gdf.crs) if hasattr(gdf, 'crs') else None,
            "bounds": list(gdf.total_bounds) if hasattr(gdf, 'total_bounds') else None,
            "sample_rows": []
        }

        # Add sample rows
        for idx, row in gdf.head(limit).iterrows():
            sample = {
                "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "datetime": str(row.get("datetime") or row.get("start_datetime")),
                "geometry": str(row.get("geometry")) if row.get("geometry") else None,
                "assets": row.get("assets") if row.get("assets") else None,
            }
            result["sample_rows"].append(sample)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inspect-cog/")
def inspect_cog(cog_url: str):
    """
    Inspect a COG file to see its CRS, bounds, and other metadata.

    Args:
        cog_url: URL to the COG file

    Returns:
        COG metadata
    """
    try:
        # Open COG
        da = rioxarray.open_rasterio(cog_url, masked=True)

        # Get bounds in the native CRS
        bounds = da.rio.bounds()

        # Handle nodata value (convert nan to None)
        nodata_val = da.rio.nodata
        if nodata_val is not None:
            nodata_val = None if math.isnan(float(nodata_val)) else float(nodata_val)

        result = {
            "crs": str(da.rio.crs),
            "bounds_native": list(bounds),
            "shape": list(da.shape),
            "resolution": [float(da.rio.resolution()[0]), float(da.rio.resolution()[1])],
            "nodata": nodata_val,
        }

        # Transform bounds to WGS84 for easier understanding
        if da.rio.crs and str(da.rio.crs) != "EPSG:4326":
            from shapely.geometry import box as shapely_box
            bounds_geom = shapely_box(*bounds)
            bounds_gdf = gpd.GeoDataFrame([1], geometry=[bounds_geom], crs=da.rio.crs)
            bounds_wgs84 = bounds_gdf.to_crs("EPSG:4326")
            result["bounds_wgs84"] = list(bounds_wgs84.total_bounds)
        else:
            result["bounds_wgs84"] = list(bounds)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _process_cog(item: tuple, bbox: list[float]) -> dict:
    """
    Process a single COG file and extract statistics for the given bbox.
    Includes timing of each step for performance diagnostics.

    Args:
        item: Tuple of (datetime, asset_id, cog_url)
        bbox: Bounding box [minx, miny, maxx, maxy] in EPSG:4326

    Returns:
        Dictionary with datetime, asset_id, statistics, and step timings
    """
    dt, asset_id, cog_url = item
    start_total = time.time()
    timings = {}

    logger.info(f"Processing COG {asset_id}")

    try:
        # Step 1: Open COG
        t0 = time.time()
        with COGReader(cog_url) as cog:
            timings['open_cog'] = time.time() - t0

            # Step 2: Read bbox
            t1 = time.time()
            img = cog.part(bbox=bbox, indexes=1)  # only first band
            timings['read_bbox'] = time.time() - t1

            # Step 3: Get masked array
            t2 = time.time()
            arr = img.array  # masked numpy array
            valid_data = arr.compressed()
            timings['mask_array'] = time.time() - t2

            if valid_data.size == 0:
                raise ValueError("No valid data in bbox")

            # Step 4: Compute statistics
            t3 = time.time()
            result_stats = {
                "min": float(valid_data.min()),
                "max": float(valid_data.max()),
                "mean": float(valid_data.mean()),
                "stddev": float(valid_data.std())
            }
            timings['compute_stats'] = time.time() - t3

        total_duration = time.time() - start_total
        logger.info(f"Finished processing {asset_id} in {total_duration:.2f}s")
        logger.info(f"Step timings: {timings}")

        result = {
            "datetime": dt,
            "asset_id": asset_id,
            **result_stats,
            "timings": timings,
            "total_duration": total_duration
        }

        return result

    except Exception as e:
        logger.error(f"Error processing {asset_id}: {e}")
        raise
class COGProcessor:
    """
    Process a single COG using rioxarray.
    Adds debug info and timing.
    """

    def __init__(self, cog_url: str):
        self.cog_url = cog_url
        self.da = None  # xarray DataArray

    def open(self):
        """Open the COG with rioxarray (lazy loading)."""
        if self.da is None:
            t0 = time.time()
            self.da = rioxarray.open_rasterio(self.cog_url, masked=True, chunks=True)
            t1 = time.time() - t0
            logger.info(f"Opened COG {self.cog_url} in {t1:.2f}s")

    def close(self):
        """Close the dataset."""
        if self.da is not None:
            self.da.close()
            self.da = None

    def process_bbox(self, item: tuple, bbox: list[float]) -> dict:
        """
        Extract statistics for each band in a COG over a given bbox.
        Uses rioxarray/xarray with mask-aware computations.
        """

        dt, asset_id, _ = item
        timings = {}
        start_total = time.time()

        self.open()

        # Step 1: Prepare bbox in raster CRS
        t0 = time.time()
        minx, miny, maxx, maxy = bbox
        if self.da.rio.crs is None:
            self.da.rio.write_crs("EPSG:4326", inplace=True)
        if self.da.rio.crs.to_string() != "EPSG:4326":
            bbox_geom = gpd.GeoDataFrame(
                geometry=[box(minx, miny, maxx, maxy)],
                crs="EPSG:4326"
            ).to_crs(self.da.rio.crs)
            minx, miny, maxx, maxy = bbox_geom.total_bounds
        timings["bbox_reprojection"] = time.time() - t0

        # Step 2: Clip to bbox
        t1 = time.time()
        clipped = self.da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        timings["clip_bbox"] = time.time() - t1

        # Step 3: Compute per-band stats
        t2 = time.time()
        band_stats = []

        # Determine band labels from GDAL metadata
        band_labels = []

        # Check if there is a 'band' dimension
        if "band" in clipped.dims:
            for i in range(clipped.rio.count):
                # Extract metadata for band i
                # rioxarray stores GDAL attributes per band in attrs (long_name, description, etc.)
                try:
                    band_da = clipped.isel(band=i)
                    name = band_da.attrs.get("long_name") or band_da.attrs.get("description") or f"band_{i+1}"
                    band_labels.append(str(name[i]))
                except Exception:
                    band_labels.append(f"band_{i+1}")
        else:
            # Single-band raster fallback
            name = clipped.attrs.get("long_name") or clipped.attrs.get("description") or "band_1"
            band_labels = [name]

        # If multi-variable dataset (rare but possible)
        if isinstance(clipped, xr.Dataset):
            datasets = {k: v for k, v in clipped.data_vars.items()}
        else:
            datasets = {"default": clipped}

        for _, da in datasets.items():
            for i, band_label in enumerate(band_labels):
                try:
                    single_band = da.isel(band=i)
                except Exception:
                    single_band = da  # single-band fallback

                nodata = single_band.rio.nodata
                data = single_band.astype("float64")

                # Mask nodata and invalid values
                if nodata is not None and not np.isnan(nodata):
                    data = data.where(data != nodata)

                if data.count().values == 0:
                    continue  # skip empty bands

                # Compute statistics with skipna
                stats = {
                    "band": band_label,
                    "index": i,
                    "min": float(data.min(skipna=True).values),
                    "max": float(data.max(skipna=True).values),
                    "mean": float(data.mean(skipna=True).values),
                    "stddev": float(data.std(skipna=True).values),
                    "valid_pixels": int(data.count().values),
                }
                band_stats.append(stats)

        timings["compute_per_band"] = time.time() - t2
        timings["total_duration"] = time.time() - start_total

        logger.info(f"Processed {asset_id}: {len(band_stats)} bands, timings={timings}")

        result = {
            "datetime": dt,
            "asset_id": asset_id,
            "bands": band_stats,
            "timings": timings,
        }
        return result

def worker_process_cog(item_bbox_tuple):
    """
    Worker function to process a single COG + bbox using rioxarray.
    Args:
        item_bbox_tuple: ((datetime, asset_id, cog_url), bbox)
    Returns:
        dict with statistics
    """
    item, bbox = item_bbox_tuple
    cog_url = item[2]

    processor = COGProcessor(cog_url)
    try:
        result = processor.process_bbox(item, bbox)
    finally:
        processor.close()

    return result

@app.get("/geoparquet-stats/")
def geoparquet_stats(
    geoparquet_url: str,
    bbox: Annotated[list[float], Query()],
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
    max_workers: int = 5,
) -> list[GeoParquetStatsItem]:
    """
    Extract statistics from COG timeseries in a GeoParquet STAC collection.

    Args:
        geoparquet_url: URL to the GeoParquet file containing STAC items
        bbox: Bounding box as [minx, miny, maxx, maxy] in EPSG:4326
        start_date: Optional start date filter
        end_date: Optional end date filter
        max_workers: Maximum number of parallel workers (default: 5)

    Returns:
        List of statistics for each COG asset
    """
    logger.info(
        "Handling geoparquet-stats request",
        url=geoparquet_url,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
    )

    # Validate bbox
    if len(bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="bbox must contain exactly 4 values: [minx, miny, maxx, maxy]"
        )

    try:

        # Load GeoParquet
        logger.debug("Loading GeoParquet", url=geoparquet_url)

        # Handle HTTPS URLs by explicitly opening with fsspec
        if geoparquet_url.startswith("https://") or geoparquet_url.startswith("http://"):
            with fsspec.open(geoparquet_url, mode='rb') as f:
                gdf = gpd.read_parquet(f)
        else:
            gdf = gpd.read_parquet(geoparquet_url)
        logger.debug(f"Loaded GeoParquet with {len(gdf)} items")

        # Filter by date range if provided
        if start_date or end_date:
            # Try 'datetime' first, fallback to start_datetime
            date_col = None
            if "datetime" in gdf.columns:
                date_col = "datetime"
            elif "start_datetime" in gdf.columns:
                date_col = "start_datetime"

            if date_col:
                mask = True
                if start_date:
                    mask = gdf[date_col] >= start_date
                if end_date:
                    if isinstance(mask, bool):
                        mask = gdf[date_col] <= end_date
                    else:
                        mask &= gdf[date_col] <= end_date
                gdf = gdf[mask]
                logger.debug(f"Filtered to {len(gdf)} items by date range")

        # Extract COG assets
        items_to_process = []
        for idx, row in gdf.iterrows():
            # Get datetime
            dt = row.get("datetime") or row.get("start_datetime")
            if dt is None:
                logger.warning(f"No datetime found for row {idx}, skipping")
                continue

            # Convert to datetime if needed
            if isinstance(dt, str):
                dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))

            # Extract assets
            if "assets" not in row or row["assets"] is None:
                logger.warning(f"No assets found for row {idx}, skipping")
                continue

            assets = row["assets"]
            if isinstance(assets, str):
                import json

                assets = json.loads(assets)

            for asset_id, asset in assets.items():
                # Check if it's a COG
                asset_type = asset.get("type", "")
                roles = asset.get("roles", [])

                # Accept if it's a geotiff or has data role
                if "geotiff" in asset_type.lower() or "data" in roles or "cog" in asset_type.lower():
                    href = asset.get("href")
                    if href:
                        items_to_process.append((dt, asset_id, href))

        if not items_to_process:
            logger.warning("No COG assets found to process")
            return []

        logger.info(f"Processing {len(items_to_process)} COG assets")

        items_to_process = items_to_process[:8]

        logger.info(f"Processing MODIFIED {len(items_to_process)} COG assets")

        # Process COGs in parallel
        # Create a COGReader for each worker process
        items_with_bbox = [(item, bbox) for item in items_to_process]  # one bbox per COG
        max_workers = 4

        # Process in parallel using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            all_results = list(executor.map(worker_process_cog, items_with_bbox))
        results = all_results
 
        # Convert to response models
        response = [GeoParquetStatsItem(**r) for r in results]
        logger.info(f"Successfully processed {len(response)} items")
        return response

    except Exception as e:
        logger.error(f"Error during GeoParquet processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def log_middle(request: Request, call_next):
    """Log all HTTP requests with timing information."""
    start_time = time.time()

    response = await call_next(request)

    ignored_paths = ["/healthz", "/metrics"]
    if request.url.path not in ignored_paths:
        duration = time.time() - start_time
        logger.info(
            "Request finished",
            method=request.method,
            url=str(request.url),
            duration_ms=duration * 1000,
            content_length=response.headers.get("content-length"),
            status=int(response.status_code),
        )

    return response


if __name__ == "__main__":
    # Example usage for testing
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
