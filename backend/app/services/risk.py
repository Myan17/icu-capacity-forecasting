from app.core.config import settings


def compute_risk(predicted_occupied: float, icu_capacity: int) -> str:
    if icu_capacity <= 0:
        return "UNKNOWN"

    ratio = predicted_occupied / icu_capacity

    if ratio >= settings.ALERT_OCCUPANCY_RED:
        return "RED"
    if ratio >= settings.ALERT_OCCUPANCY_YELLOW:
        return "YELLOW"
    return "GREEN"