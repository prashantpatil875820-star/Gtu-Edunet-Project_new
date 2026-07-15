"""
Fetches LIVE satellite data for a given location:

1. NASA GIBS  -> true-color satellite image (public, no API key, updates daily)
2. NASA FIRMS -> active-fire hotspot detections for the same area (needs a
                 free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/)

The GIBS image is what we feed into the CNN. The FIRMS hotspots are used
as an independent real-world signal you can compare the CNN's prediction
against, or use as weak labels when building a training set.
"""

import io
import os
import csv
import datetime as dt

import requests
from PIL import Image

import config


def _bbox_from_point(lat, lon, half_size_deg=0.1):
    """Build a small bounding box (lon/lat degrees) around a point."""
    return (lon - half_size_deg, lat - half_size_deg,
            lon + half_size_deg, lat + half_size_deg)


def fetch_gibs_image(lat, lon, date=None, half_size_deg=0.15,
                      pixels=1024, save_path=None):
    """
    Download a true-color satellite image centered on (lat, lon) from
    NASA GIBS for a given date (defaults to yesterday, since same-day
    imagery is often not yet processed).

    Returns a PIL.Image.
    """
    if date is None:
        date = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    min_lon, min_lat, max_lon, max_lat = _bbox_from_point(lat, lon, half_size_deg)

    params = {
      "SERVICE": "WMS",
      "VERSION": "1.1.1",          # 1.3.0 ki jagah
      "REQUEST": "GetMap",
      "LAYERS": config.GIBS_LAYER,
      "SRS": "EPSG:4326",          # CRS ki jagah SRS
      "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}",
      "WIDTH": pixels,
      "HEIGHT": pixels,
      "FORMAT": "image/jpeg",
      "STYLES": "",
      "TRANSPARENT": "FALSE",
      "TIME": date,
  }

    resp = requests.get(config.GIBS_WMS_URL, params=params, timeout=30)
    resp.raise_for_status()

    if "image" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(
            f"GIBS did not return an image (got {resp.headers.get('Content-Type')}). "
            f"Response: {resp.text[:300]}"
        )

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

    return img


def fetch_firms_hotspots(lat, lon, half_size_deg=0.15, day_range=None):
    """
    Query NASA FIRMS for active-fire detections in a bounding box around
    (lat, lon) over the last `day_range` days. Returns a list of dicts
    with at least: latitude, longitude, brightness, confidence, acq_date.
    """
    if config.FIRMS_MAP_KEY in (None, "", "PASTE_YOUR_FIRMS_MAP_KEY_HERE"):
        raise RuntimeError(
            "No FIRMS_MAP_KEY set. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/ and put it "
            "in a .env file as FIRMS_MAP_KEY=..."
        )

    day_range = day_range or config.FIRMS_DAY_RANGE
    min_lon, min_lat, max_lon, max_lat = _bbox_from_point(lat, lon, half_size_deg)
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    url = (
        f"{config.FIRMS_AREA_URL}/{config.FIRMS_MAP_KEY}/"
        f"{config.FIRMS_SOURCE}/{bbox_str}/{day_range}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    hotspots = list(reader)
    return hotspots


def get_live_sample(lat, lon, date=None, save_dir=None):
    """
    Convenience wrapper: fetch both the satellite image and the active-fire
    hotspots for a location, so you can predict + sanity-check in one call.
    """
    save_dir = save_dir or config.LIVE_IMAGE_CACHE
    fname = f"{lat:.3f}_{lon:.3f}_{date or 'latest'}.jpg".replace(":", "-")
    save_path = os.path.join(save_dir, fname)

    image = fetch_gibs_image(lat, lon, date=date, save_path=save_path)

    try:
        hotspots = fetch_firms_hotspots(lat, lon)
    except RuntimeError as e:
        print(f"[warn] Skipping FIRMS hotspot check: {e}")
        hotspots = []

    return {
        "image": image,
        "image_path": save_path,
        "hotspots": hotspots,
        "hotspot_count": len(hotspots),
    }


if __name__ == "__main__":
    # Example: Los Angeles National Forest area
    sample = get_live_sample(lat=34.2, lon=-118.1)
    print(f"Saved live image to: {sample['image_path']}")
    print(f"Active fire hotspots nearby (last {config.FIRMS_DAY_RANGE}d): "
          f"{sample['hotspot_count']}")
