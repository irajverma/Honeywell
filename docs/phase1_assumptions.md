# Phase 1 — Behavioral Assumptions & Data Generation Notes

## Overview

This document describes the design decisions, statistical distributions, and
injection rates used by the synthetic data generator for the Behavioral Anomaly
Detection system.

---

## Entity Types

| Type | Count | Description |
|---|---|---|
| **User** | 50 | Human employees with work-hour patterns |
| **Service Account** | 10 | Automated processes running 24/7 |
| **Edge Device** | 20 | IoT/OT devices at fixed sites (elevated count for cold-start testing) |
| **Total** | **80** | |

Edge device count is intentionally higher than the minimum needed — the
problem statement explicitly calls out IoT/OT as a target domain, and more
edge device entities provide better signal for cold-start and baseline
profiling tests in Phases 2–3.

---

## Behavioral Profile Distributions

### Login Hours — Beta Distribution

Each entity's login hour is sampled from `Beta(α, β)` mapped to a
`[shift, shift+scale]` hour range:

| Entity Type | α range | β range | Shift | Scale | Effect |
|---|---|---|---|---|---|
| User | 3–8 | 2–6 | 5–8 | 12–16 | Peak during work hours (≈8–18), per-user variance |
| Service Account | 1–2 | 1–2 | 0 | 24 | Near-uniform (24/7 operation) |
| Edge Device | 2–5 | 2–5 | 4–7 | 10–16 | Concentrated around industrial shift times |

**Why Beta?** It's bounded (unlike Normal), unimodal, and the shape is
independently tunable per entity — producing realistic diversity.

### Session Duration — LogNormal Distribution

`LogNormal(μ, σ)` in minutes:

| Entity Type | μ range | σ range | Median |
|---|---|---|---|
| User | 2.7–4.1 | 0.3–0.7 | 15–60 min |
| Service Account | 0.7–2.3 | 0.2–0.4 | 2–10 min |
| Edge Device | 0.0–1.6 | 0.2–0.5 | 1–5 min |

**Why LogNormal?** Session durations are strictly positive, right-skewed,
and heavy-tailed — properties that LogNormal captures well. Real-world
session data consistently fits LogNormal better than Normal or Exponential.

### Resources — Zipf-Weighted Subset

Each entity accesses a subset of the 50-resource global pool:

| Entity Type | Resources | Weighting | Noise |
|---|---|---|---|
| User | 3–8 | Zipf(1.5) | 1% curiosity access outside typical set |
| Service Account | 2–5 | Dirichlet(2.0) | <1% |
| Edge Device | 1–3 (OT/IoT only) | Dirichlet(2.0) | <1% |

### Source IPs and Geo-Locations

- **Users**: 1–2 home cities, 5 IPs per geo pool, 2% chance of IP jitter
- **Service Accounts**: 1 datacenter region, 2 IPs
- **Edge Devices**: 1 fixed site, 2 IPs

### Daily Event Rate — Poisson

| Entity Type | λ range | Expected daily events |
|---|---|---|
| User | 5–20 | ~10 |
| Service Account | 50–200 | ~100 |
| Edge Device | 10–40 | ~20 |

---

## Attack Patterns & Injection Rates

### Three-Way Label Schema

The ground truth uses **three label values**, not the traditional binary:

| Label | Meaning | Used For |
|---|---|---|
| `normal` | Legitimate activity | Negative class |
| `anomaly` | Confirmed attack pattern | Positive class for precision/recall |
| `ambiguous` | Edge case (insider drift) | Excluded from core metrics; used for FP tuning |

**Rationale:** Insider drift is described in the problem statement as an
ambiguous edge case.  Tagging it as `"ambiguous"` rather than `"anomaly"`
keeps precision/recall on the core anomaly class clean.  During evaluation
in Phase 3, the `ambiguous` class should be:
- **Excluded** from the primary PR-AUC calculation
- **Included** in a separate analysis of false-positive behavior
- Used to test the model's calibration on uncertain cases

### Attack Injection Summary

| Attack | Rate | Target | Key Signals |
|---|---|---|---|
| **Brute Force** | 0.5% | Users | 10–50 failed auths in <5 min, single IP |
| **Impossible Travel** | 0.3% | Users | >5000 km geo gap, <1 hour time gap |
| **Credential Stuffing** | 0.4% | Many users | Small botnet IP set, 5–15% success rate |
| **Lateral Movement** | 0.3% | Any | 3–6 unusual resources accessed in sequence |
| **Device Spoofing** | 0.3% | Any | Unknown device fingerprint |
| **Exfiltration** | 0.4% | Users | Elevated session duration over 5–15 days |
| **Insider Drift** | 0.3% | Users | Gradual resource creep + off-hours shift (→ `ambiguous`) |
| **Total** | **~2.5%** | | Within 0.5–3% target range |

### Attack Pattern Details

1. **Brute Force**: Rapid failed password attempts from a single external IP.
   30% chance of eventual success after the burst. Session durations are
   very short (failed auth = seconds).

2. **Impossible Travel**: A legitimate login is followed by a second login
   from >5000 km away within 5–55 minutes. Uses haversine distance.

3. **Credential Stuffing**: A botnet (2–5 IPs) runs a campaign hitting
   8–24 distinct users over 2–8 hours with low success rate.

4. **Lateral Movement**: Compromised entity accesses 3–6 resources outside
   their profile in sequence, using suspicious commands (psexec, mimikatz,
   pass\_the\_hash).

5. **Device Spoofing**: Login from a device fingerprint never seen in the
   entity's known set. 40% also come from an unusual IP.

6. **Low-and-Slow Exfiltration**: 5–15 day campaign with 1–2 events/day.
   Session durations are 1.5–3× normal. Accesses data-heavy resources.
   Hard to detect per-event; anomaly is only visible in aggregate.

7. **Insider Drift** *(ambiguous)*: Gradual resource set expansion (1–2 new
   resources per week) combined with increasing off-hours activity over 2–4
   weeks. Uses the entity's own credentials and devices — intentionally
   hard to distinguish from a legitimate role change.

---

## Train / Test Split & Concept Drift

### Split Strategy

The dataset uses a **time-based split** rather than random — reflecting
real-world deployment where models train on historical data and predict
future events.

| Period | Days | Purpose |
|---|---|---|
| **Train** | 1–48 | Model training and validation |
| **Test** | 49–60 (last 12 days) | Held-out evaluation |

The split is indicated by a `split` column in `events.csv` with values
`"train"` or `"test"`.

### Concept Drift in Test Period

To test model robustness against natural behavioral change, **~12% of
entities** (~10 of 80) receive a legitimate behavioral shift during the
test period.  These shifts are **labeled `"normal"`** — they are NOT
attacks.

Drift types applied (randomly per entity):
- **Hour shift**: Login peak moves by ±1–3 hours (e.g., new schedule)
- **Resource expansion**: 1 new resource added to ~20% of events
  (e.g., assigned to a new project)
- **Session duration change**: Multiplied by a factor of 0.7–1.4
  (e.g., different workflow)

**Purpose**: Models that overfit to rigid behavioral baselines will
false-positive on these drifted entities.  Phase 2's z-score model is
expected to struggle here; Phase 3's sequence model should generalize
better.

---

## Output Schema

### events.csv (model input)

| Column | Type | Description |
|---|---|---|
| `entity_id` | str | Unique entity identifier |
| `entity_type` | str | `user` / `service_account` / `edge_device` |
| `timestamp` | datetime | ISO 8601 timestamp |
| `source_ip` | str | Source IP address |
| `geo_location` | str | `"City, CC"` format |
| `resource_accessed` | str | Target resource name |
| `auth_method` | str | Authentication method used |
| `session_duration` | float | Duration in minutes |
| `command_sequence` | str | Comma-separated command list |
| `device_fingerprint` | str | Device identifier hash |
| `split` | str | `"train"` or `"test"` |

### ground_truth.csv (evaluation only)

| Column | Type | Description |
|---|---|---|
| `entity_id` | str | Matches events.csv |
| `timestamp` | datetime | Matches events.csv (join key) |
| `label` | str | `"normal"` / `"anomaly"` / `"ambiguous"` |
| `attack_subtype` | str | Attack type name or null |
| `split` | str | `"train"` or `"test"` |

---

## Limitations

1. **Synthetic ≠ Real**: Distributions are approximations. Real telemetry
   has seasonality (weekends, holidays), organizational structure effects,
   and correlated entity behaviors that aren't modeled here.

2. **Independent entities**: Entities are generated independently. Real
   attacks like lateral movement involve coordinated multi-entity chains.

3. **Fixed attacker behavior**: Each attack type uses a single pattern
   template. Real adversaries adapt and blend techniques.

4. **No network topology**: The generator doesn't model which resources
   are reachable from which IPs. All entities can access all resources.

5. **Command vocabulary is simplified**: Real command sequences have
   complex dependencies and ordering constraints not captured here.
