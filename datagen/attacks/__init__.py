"""
attacks/__init__.py — Attack injection registry.

Each attack module exposes an `inject_*` function with the same signature.
This registry makes it easy to iterate over all attacks in the orchestrator.
"""

from datagen.attacks.brute_force import inject_brute_force
from datagen.attacks.impossible_travel import inject_impossible_travel
from datagen.attacks.credential_stuffing import inject_credential_stuffing
from datagen.attacks.lateral_movement import inject_lateral_movement
from datagen.attacks.device_spoofing import inject_device_spoofing
from datagen.attacks.exfiltration import inject_exfiltration
from datagen.attacks.insider_drift import inject_insider_drift

# Ordered list — orchestrator iterates through these
ATTACK_INJECTORS = [
    ("brute_force", inject_brute_force),
    ("impossible_travel", inject_impossible_travel),
    ("credential_stuffing", inject_credential_stuffing),
    ("lateral_movement", inject_lateral_movement),
    ("device_spoofing", inject_device_spoofing),
    ("exfiltration", inject_exfiltration),
    ("insider_drift", inject_insider_drift),
]
