"""
Central configuration for the Forest Fire Risk Prediction project.

Get your free NASA FIRMS MAP_KEY here (takes ~1 minute, no cost):
    https://firms.modaps.eosdis.nasa.gov/api/map_key/

NASA GIBS (satellite basemap imagery) requires NO api key at all -
it's public WMTS/WMS tile service.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- API keys
# Put this in a local .env file (never commit it): FIRMS_MAP_KEY=xxxxxxxx
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "PASTE_YOUR_FIRMS_MAP_KEY_HERE")

# ---------------------------------------------------------------- Image / model
IMAGE_SIZE = (128, 128)      # (height, width) fed to the CNN
CHANNELS = 3                 # RGB true-color satellite composite
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# Classes the CNN predicts. Keep binary for a simple/robust first model;
# switch to the 4-class list if you have graded-risk training data.
CLASS_NAMES_BINARY = ["fire_risk", "no_fire_risk"]
CLASS_NAMES_MULTI = ["low", "moderate", "high", "very_high"]
USE_MULTICLASS = False

# ---------------------------------------------------------------- Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")   # dataset/<class>/*.jpg
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(MODEL_OUT_DIR, "fire_risk_cnn.keras")
LIVE_IMAGE_CACHE = os.path.join(BASE_DIR, "outputs", "live_images")

# ---------------------------------------------------------------- NASA GIBS (live imagery)
# Free, public, no auth. We use the VIIRS/MODIS true-color corrected-reflectance
# layer, which updates daily.
GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
# GIBS_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
GIBS_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
# GIBS_LAYER = "MODIS_Aqua_CorrectedReflectance_TrueColor"

# ---------------------------------------------------------------- NASA FIRMS (active-fire ground truth)
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCE = "VIIRS_SNPP_NRT"   # near-real-time active fire detections
FIRMS_DAY_RANGE = 1               # last N days
