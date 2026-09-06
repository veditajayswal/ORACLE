"""
ORACLE ML — risk/recommendation.py
Maps risk level and health state to a concrete recommended action.
These appear in the Android app and ORACLE's explanation output.
"""

from typing import Dict


# Maps (risk_level, health_state) -> recommended action string
_RECOMMENDATION_MAP: Dict[tuple, str] = {
    ("LOW",      "HEALTHY"):   "Machine is operating normally. Continue regular monitoring.",
    ("LOW",      "ATTENTION"): "Minor deviation detected. Monitor closely for changes.",
    ("LOW",      "CRITICAL"):  "Unusual state. Inspect machine at next available opportunity.",

    ("MEDIUM",   "HEALTHY"):   "Trend indicates potential issue. Schedule a check within the week.",
    ("MEDIUM",   "ATTENTION"): "Degradation detected. Schedule inspection within 48 hours.",
    ("MEDIUM",   "CRITICAL"):  "Significant degradation. Inspect mechanical assembly soon.",

    ("HIGH",     "HEALTHY"):   "High risk detected despite current normal readings. Perform diagnostic check.",
    ("HIGH",     "ATTENTION"): "Elevated risk. Inspect mechanical assembly within 24 hours.",
    ("HIGH",     "CRITICAL"):  "High risk of failure. Stop machine and inspect immediately.",

    ("CRITICAL", "HEALTHY"):   "Critical risk pattern detected. Perform immediate diagnostic.",
    ("CRITICAL", "ATTENTION"): "Critical — stop machine and perform immediate inspection.",
    ("CRITICAL", "CRITICAL"):  "STOP MACHINE. Immediate mechanical inspection required.",
}

_DEFAULT_RECOMMENDATION = "Continue monitoring. Consult maintenance team if issues persist."


def get_recommendation(risk_level: str, health_state: str) -> str:
    """
    Return the recommended action string for a given risk level and health state.

    Args:
        risk_level:   "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        health_state: "HEALTHY" | "ATTENTION" | "CRITICAL"
    """
    key = (risk_level.upper(), health_state.upper())
    return _RECOMMENDATION_MAP.get(key, _DEFAULT_RECOMMENDATION)


def should_trigger_alert(risk_level: str, health_state: str) -> bool:
    """
    True if ORACLE should send a push notification / alert to the Android app.
    """
    risk_alert = risk_level.upper() in ("HIGH", "CRITICAL")
    state_alert = health_state.upper() in ("ATTENTION", "CRITICAL")
    return risk_alert or state_alert


def should_command_stop(risk_level: str, health_state: str) -> bool:
    """
    True if ORACLE should issue a relay-STOP command to the ESP32.
    Only for the most critical combination.
    """
    return risk_level.upper() == "CRITICAL" and health_state.upper() == "CRITICAL"
