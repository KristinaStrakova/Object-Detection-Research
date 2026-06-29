"""RDW registry lookups for Dutch plates."""

import json
import re
from urllib import parse, request as urlrequest, error as urlerror

from settings import RDW_API_URL, RDW_TIMEOUT_S


def normalise_plate_for_registry(plate_text: str) -> str:
    """Create an RDW query key: uppercase letters/numbers only."""
    return re.sub(r"[^A-Z0-9]", "", (plate_text or "").upper())


def fetch_rdw_vehicle_info(plate_text: str) -> dict:
    """
    Query Dutch RDW open-data API for one plate.
    Returns a serializable result with validity and basic vehicle fields.
    """
    normalized = normalise_plate_for_registry(plate_text)
    result = {
        "query_plate": plate_text,
        "normalized_plate": normalized,
        "valid": False,
        "found": False,
        "source": "RDW",
        "error": None,
        "vehicle": None,
    }

    if len(normalized) < 6:
        result["error"] = "Plate format too short for RDW lookup"
        return result

    url = f"{RDW_API_URL}?{parse.urlencode({'kenteken': normalized})}"
    req = urlrequest.Request(url, headers={"User-Agent": "plate-scanner-local/1.0"})

    try:
        with urlrequest.urlopen(req, timeout=RDW_TIMEOUT_S) as resp:
            payload = resp.read().decode("utf-8")
        rows = json.loads(payload)
        print(f"[RDW] Raw response for '{normalized}':")
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    except urlerror.HTTPError as e:
        result["error"] = f"RDW HTTP error: {e.code}"
        return result
    except (urlerror.URLError, TimeoutError) as e:
        result["error"] = f"RDW request failed: {e}"
        return result
    except Exception as e:
        result["error"] = f"RDW parsing failed: {e}"
        return result

    if not isinstance(rows, list) or not rows:
        result["valid"] = True
        result["found"] = False
        return result

    row = rows[0]
    result["valid"] = True
    result["found"] = True
    result["vehicle"] = {
        "plate": row.get("kenteken") or normalized,
        "brand": row.get("merk"),
        "model": row.get("handelsbenaming"),
        "vehicle_type": row.get("voertuigsoort"),
        "first_color": row.get("eerste_kleur"),
        "fuel": row.get("brandstof_omschrijving"),
        "first_admission": row.get("datum_eerste_toelating"),
        "apk_expiry": row.get("vervaldatum_apk"),
    }
    return result
