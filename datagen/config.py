"""
config.py — All tunable constants for the synthetic data generator.

Centralizes every parameter so that experiments are reproducible and
adjustments don't require hunting through multiple files.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GeneratorConfig:
    """Master configuration for the synthetic data generator."""

    # ── Entity counts ──────────────────────────────────────────────
    num_users: int = 50
    num_service_accounts: int = 10
    num_edge_devices: int = 20

    # ── Time window ────────────────────────────────────────────────
    sim_days: int = 60

    # ── Attack injection rates (fraction of total sessions) ────────
    attack_rates: dict = field(default_factory=lambda: {
        "brute_force":        0.005,
        "impossible_travel":  0.003,
        "credential_stuffing": 0.004,
        "lateral_movement":   0.003,
        "device_spoofing":    0.003,
        "exfiltration":       0.004,
        "insider_drift":      0.003,
    })

    # ── Reproducibility ───────────────────────────────────────────
    random_seed: int = 42

    # ── Train / test split ────────────────────────────────────────
    test_days: int = 12            # last N days reserved for test
    concept_drift_frac: float = 0.12  # fraction of entities that drift in test

    # ── Resource pool ─────────────────────────────────────────────
    resource_pool: List[str] = field(default_factory=lambda: [
        # Databases
        "db_customers", "db_orders", "db_inventory", "db_employees",
        "db_financials", "db_logs", "db_analytics", "db_configs",
        # APIs
        "api_auth", "api_payments", "api_notifications", "api_reports",
        "api_users", "api_search", "api_admin", "api_metrics",
        # File shares
        "share_engineering", "share_hr", "share_legal", "share_finance",
        "share_marketing", "share_executive", "share_research",
        # Servers / infra
        "server_web01", "server_web02", "server_app01", "server_app02",
        "server_db_primary", "server_db_replica", "server_cache",
        "server_queue", "server_monitoring", "server_ci_cd",
        # Cloud services
        "s3_data_lake", "s3_backups", "s3_models", "s3_logs",
        "lambda_etl", "lambda_alerts", "lambda_reports",
        # OT / IoT endpoints
        "plc_line1", "plc_line2", "scada_hmi", "sensor_temp_01",
        "sensor_pressure_01", "sensor_flow_01", "gateway_north",
        "gateway_south", "actuator_valve_01", "actuator_pump_01",
    ])

    # ── Command vocabularies per entity type ──────────────────────
    user_commands: List[str] = field(default_factory=lambda: [
        "ssh", "ls", "cd", "cat", "grep", "vim", "git", "docker",
        "kubectl", "curl", "scp", "exit", "sudo", "tail", "head",
        "python", "pip", "make", "cp", "mv", "rm", "mkdir", "chmod",
    ])

    service_account_commands: List[str] = field(default_factory=lambda: [
        "api_call", "db_query", "health_check", "sync", "backup",
        "rotate_keys", "deploy", "fetch_config", "write_log",
        "poll_queue", "send_notification", "aggregate",
    ])

    edge_device_commands: List[str] = field(default_factory=lambda: [
        "read_sensor", "write_register", "heartbeat", "upload_telemetry",
        "firmware_check", "calibrate", "reset", "ack_alarm",
        "set_threshold", "report_status",
    ])

    # ── Auth methods per entity type (method → weight) ────────────
    user_auth_methods: dict = field(default_factory=lambda: {
        "password": 0.3, "MFA": 0.5, "SSO": 0.2,
    })

    service_auth_methods: dict = field(default_factory=lambda: {
        "api_key": 0.6, "certificate": 0.4,
    })

    edge_auth_methods: dict = field(default_factory=lambda: {
        "certificate": 0.5, "token": 0.5,
    })

    # ── Geo locations (city, country, lat, lon) ───────────────────
    geo_locations: List[dict] = field(default_factory=lambda: [
        {"city": "Mumbai", "country": "IN", "lat": 19.076, "lon": 72.878},
        {"city": "Delhi", "country": "IN", "lat": 28.614, "lon": 77.209},
        {"city": "Bangalore", "country": "IN", "lat": 12.972, "lon": 77.595},
        {"city": "Hyderabad", "country": "IN", "lat": 17.385, "lon": 78.487},
        {"city": "Chennai", "country": "IN", "lat": 13.083, "lon": 80.271},
        {"city": "Pune", "country": "IN", "lat": 18.520, "lon": 73.857},
        {"city": "New York", "country": "US", "lat": 40.713, "lon": -74.006},
        {"city": "San Francisco", "country": "US", "lat": 37.775, "lon": -122.419},
        {"city": "London", "country": "GB", "lat": 51.507, "lon": -0.128},
        {"city": "Singapore", "country": "SG", "lat": 1.352, "lon": 103.820},
        {"city": "Tokyo", "country": "JP", "lat": 35.682, "lon": 139.692},
        {"city": "Sydney", "country": "AU", "lat": -33.868, "lon": 151.209},
        {"city": "Frankfurt", "country": "DE", "lat": 50.110, "lon": 8.682},
        {"city": "Dubai", "country": "AE", "lat": 25.205, "lon": 55.271},
        {"city": "Toronto", "country": "CA", "lat": 43.653, "lon": -79.383},
    ])

    @property
    def total_entities(self) -> int:
        return self.num_users + self.num_service_accounts + self.num_edge_devices

    @property
    def total_attack_rate(self) -> float:
        return sum(self.attack_rates.values())
