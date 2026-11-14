"""Project-wide settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

PROJECT_NAME = "scouting-liga3"
# Set SCOUTING_CONTACT_EMAIL to your own address before collecting: it goes in
# the User-Agent so the site can reach you. Collection refuses to run without it.
CONTACT_EMAIL = os.environ.get("SCOUTING_CONTACT_EMAIL", "")
USER_AGENT = (
    f"{PROJECT_NAME}/0.1 (academic research project; contact: {CONTACT_EMAIL})"
)

# Politeness: pause between real network requests (never in parallel), plus a
# long rest every LONG_PAUSE_EVERY requests. A second consecutive failure stops
# the run (see http.PoliteClient.get).
MIN_DELAY_S = 5.0
MAX_DELAY_S = 6.0
LONG_PAUSE_EVERY = 80
LONG_PAUSE_S = 180.0
