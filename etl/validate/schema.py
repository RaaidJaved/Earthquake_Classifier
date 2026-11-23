"""Pydantic Schema Validation Module for USGS Seismic Telemetry.

Enforces data integrity, type checking, and boundary constraints on raw GeoJSON records.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class USGSGeometry(BaseModel):
    """GeoJSON Point geometry."""
    type: str = "Point"
    coordinates: List[Optional[float]]  # [longitude, latitude, depth]

    @field_validator("coordinates")
    @classmethod
    def check_coordinates(cls, v: List[Optional[float]]) -> List[Optional[float]]:
        if len(v) < 2:
            raise ValueError("Coordinates must have at least [longitude, latitude]")
        lon, lat = v[0], v[1]
        if lon is not None and not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude {lon} out of range [-180, 180]")
        if lat is not None and not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        return v


class USGSProperties(BaseModel):
    """USGS raw earthquake properties."""
    mag: Optional[float] = None
    place: Optional[str] = "Unknown Location"
    time: Optional[int] = None
    updated: Optional[int] = None
    tz: Optional[int] = None
    url: Optional[str] = None
    detail: Optional[str] = None
    felt: Optional[int] = None
    cdi: Optional[float] = None
    mmi: Optional[float] = None
    alert: Optional[str] = None
    status: Optional[str] = None
    tsunami: Optional[int] = 0
    sig: Optional[int] = None
    net: Optional[str] = None
    code: Optional[str] = None
    ids: Optional[str] = None
    sources: Optional[str] = None
    types: Optional[str] = None
    nst: Optional[int] = None  # Number of seismic stations
    dmin: Optional[float] = None  # Distance to nearest station (degrees)
    rms: Optional[float] = None  # Travel time residual (seconds)
    gap: Optional[float] = None  # Azimuthal gap (degrees)
    magType: Optional[str] = None
    type: Optional[str] = "earthquake"
    title: Optional[str] = None

    @field_validator("gap")
    @classmethod
    def check_gap(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 360.0):
            return None  # Replace corrupted gap with None rather than dropping record
        return v


class USGSEarthquakeRaw(BaseModel):
    """Raw GeoJSON Feature representation."""
    type: str = "Feature"
    id: str
    properties: USGSProperties
    geometry: USGSGeometry


class ValidatedEarthquakeRecord(BaseModel):
    """Flattened and validated record ready for Pandas cleaning and transformation."""
    id: str
    magnitude: Optional[float]
    place: Optional[str]
    time_epoch: Optional[int]
    longitude: float
    latitude: float
    depth: float
    nst: Optional[int] = None  # station count
    dmin: Optional[float] = None  # distance to nearest station
    rms: Optional[float] = None  # travel-time error
    gap: Optional[float] = None  # azimuthal gap
    mag_type: Optional[str] = None
    alert: Optional[str] = None
    felt: Optional[int] = None
    tsunami: int = 0
    sig: Optional[int] = None
    net: Optional[str] = None
    status: Optional[str] = None

    @classmethod
    def from_raw_feature(cls, feature: USGSEarthquakeRaw) -> ValidatedEarthquakeRecord:
        props = feature.properties
        geom = feature.geometry
        coords = geom.coordinates
        
        lon = float(coords[0]) if coords[0] is not None else 0.0
        lat = float(coords[1]) if coords[1] is not None else 0.0
        depth = float(coords[2]) if len(coords) > 2 and coords[2] is not None else 0.0

        return cls(
            id=feature.id,
            magnitude=props.mag,
            place=props.place,
            time_epoch=props.time,
            longitude=lon,
            latitude=lat,
            depth=depth,
            nst=props.nst,
            dmin=props.dmin,
            rms=props.rms,
            gap=props.gap,
            mag_type=props.magType,
            alert=props.alert,
            felt=props.felt,
            tsunami=props.tsunami or 0,
            sig=props.sig,
            net=props.net,
            status=props.status,
        )


def validate_raw_records(features: List[Dict[str, Any]]) -> Tuple[List[ValidatedEarthquakeRecord], int]:
    """Validates raw USGS features using Pydantic.
    
    Returns:
        Tuple of (list_of_valid_records, count_of_invalid_records)
    """
    valid_records: List[ValidatedEarthquakeRecord] = []
    invalid_count = 0

    for item in features:
        try:
            raw_model = USGSEarthquakeRaw.model_validate(item)
            validated = ValidatedEarthquakeRecord.from_raw_feature(raw_model)
            valid_records.append(validated)
        except Exception as e:
            invalid_count += 1
            logger.debug(f"Validation discarded record: {e}")

    logger.info(f"Validated {len(valid_records)} records successfully ({invalid_count} rejected)")
    return valid_records, invalid_count
