"""USGS Earthquake Data Extraction Module.

Uses requests to pull real-time or historical seismic telemetry from the public USGS API.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

USGS_FEED_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
USGS_QUERY_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"


class USGSClient:
    """Client for extracting seismic telemetry from the USGS API."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EarthquakeClassifier-ETL/1.0",
            "Accept": "application/json"
        })

    def fetch_feed(self, feed: str = "all_week") -> List[Dict[str, Any]]:
        """Fetch real-time GeoJSON feed.
        
        Args:
            feed: e.g., 'all_day', 'all_week', 'all_month', '2.5_month'
        
        Returns:
            List of raw feature dictionaries
        """
        url = f"{USGS_FEED_BASE}/{feed}.geojson"
        logger.info(f"Extracting USGS feed from: {url}")
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            logger.info(f"Successfully extracted {len(features)} seismic events from USGS feed '{feed}'")
            return features
        except requests.RequestException as e:
            logger.error(f"Failed to fetch USGS feed {feed}: {e}")
            raise

    def fetch_by_query(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        min_magnitude: Optional[float] = None,
        limit: int = 20000
    ) -> List[Dict[str, Any]]:
        """Fetch seismic data using USGS FDSNWS query parameters.
        
        Args:
            start_time: ISO-8601 formatted date (e.g. '2026-01-01')
            end_time: ISO-8601 formatted date (e.g. '2026-01-31')
            min_magnitude: Minimum magnitude filter
            limit: Maximum records to return (USGS max is 20000)
        """
        params: Dict[str, Any] = {"format": "geojson", "limit": limit}
        if start_time:
            params["starttime"] = start_time
        if end_time:
            params["endtime"] = end_time
        if min_magnitude is not None:
            params["minmagnitude"] = min_magnitude

        logger.info(f"Extracting USGS data via query: {params}")
        try:
            response = self.session.get(USGS_QUERY_BASE, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            logger.info(f"Successfully queried {len(features)} seismic events from USGS API")
            return features
        except requests.RequestException as e:
            logger.error(f"Failed to query USGS API: {e}")
            raise


def fetch_usgs_earthquakes(feed_or_query: str = "all_week") -> List[Dict[str, Any]]:
    """Convenience helper to extract raw earthquake records."""
    client = USGSClient()
    return client.fetch_feed(feed=feed_or_query)
