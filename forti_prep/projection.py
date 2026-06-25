from dataclasses import dataclass

import cf_units
import numpy as np
import pyproj
import xarray as xr


@dataclass
class LatLon:
    lat: np.ndarray
    lon: np.ndarray


def find_latlon(ds: xr.Dataset) -> LatLon | None:
    """Resolve 2-D lat/lon arrays from a CF-compliant dataset without needing config.

    Tries in order:
    1. Pre-existing latitude/longitude coordinates (most projected datasets).
    2. Reprojection via the grid_mapping variable, discovering x/y coordinates
       by their CF standard_name (e.g. rotated_pole without pre-computed lat/lon).

    Returns None when the dataset lacks enough metadata to resolve lat/lon on
    its own; callers can then fall back to config-provided dimension names.
    """
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return LatLon(
            lat=ds["latitude"].values,
            lon=ds["longitude"].values,
        )

    grid_mapping_var = next(
        (ds[v] for v in ds.data_vars if "grid_mapping_name" in ds[v].attrs), None
    )
    if grid_mapping_var is None:
        return None

    x_coord = next(
        (
            ds[c]
            for c in ds.coords
            if ds[c].attrs.get("standard_name") == "projection_x_coordinate"
        ),
        None,
    )
    y_coord = next(
        (
            ds[c]
            for c in ds.coords
            if ds[c].attrs.get("standard_name") == "projection_y_coordinate"
        ),
        None,
    )
    if x_coord is None or y_coord is None:
        return None

    return _reproject_to_latlon(x_coord, y_coord, grid_mapping_var)


def _reproject_to_latlon(
    x_coord: xr.DataArray,
    y_coord: xr.DataArray,
    grid_mapping_var: xr.DataArray,
) -> LatLon:
    """Reproject x/y coordinate arrays to WGS-84 lat/lon."""
    crs_src = pyproj.CRS.from_cf(grid_mapping_var.attrs)
    transformer = pyproj.Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)

    x_vals = x_coord.values.astype(float)
    y_vals = y_coord.values.astype(float)

    # Scale coordinate values to the CRS native units (e.g. "100 km" → metres).
    crs_unit_name = crs_src.axis_info[0].unit_name
    for vals, coord in [(x_vals, x_coord), (y_vals, y_coord)]:
        unit_str = coord.attrs.get("units", "")
        if unit_str:
            src_unit = cf_units.Unit(unit_str)
            dst_unit = cf_units.Unit(crs_unit_name)
            if src_unit.is_convertible(dst_unit):
                scale = src_unit.convert(1.0, dst_unit)
                if scale != 1.0:
                    vals *= scale

    xx, yy = np.meshgrid(x_vals, y_vals) if x_vals.ndim == 1 else (x_vals, y_vals)
    lon, lat = transformer.transform(xx, yy)
    return LatLon(lat=lat, lon=lon)
