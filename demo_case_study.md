# Demo Case Study: Before/After Lateral Movement Surfacing

## Overview
This case study documents a real test-split attack event where the **Standalone Baseline Profiler** missed the threat due to budget crowding, but the **Tiered Monotonic Hybrid Ensemble** successfully surfaced it directly into the SOC analyst triage queue.

---

## The Target Attack Event
- **Entity ID:** `user_018`
- **Timestamp:** `2026-06-29 18:22:41` UTC
- **Attack Subtype:** `lateral_movement`
- **Resource Traversal:** `distinct_resources_recent = 7` (accessed 7 distinct enterprise resources within the recent activity window)

---

## The Before & After Story

### Before (Standalone Baseline Profiler)
- **Baseline Score:** `2.00`
- **Baseline Rank:** **#732** out of 28,758 total test events
- **SOC Alert Queue Status:** **MISSED** (Ranked outside the Top 2.5% budget cutoff of #718 alerts)
- **Why Baseline Missed It:** The baseline rule scored this event as a moderate score (`2.00`), which was crowded out by higher-volume perimeter rule violations.

### After (Tiered Monotonic Hybrid Ensemble)
- **LSTM Sequence Reconstruction Loss:** `2.0667` (Percentile: `99.69%`)
- **Ensemble Boost Condition:** `baseline_score < 3.0` and `res_recent >= 5`
- **Hybrid Score:** `2.9870`
- **Hybrid Rank:** **#705** out of 28,758 total test events
- **Rank Jump:** **+27 spots**
- **SOC Alert Queue Status:** **FLAGGED & TRIAGED** (Surfaced inside the Top 2.5% operational SOC budget queue)

---

## Live Stage Demo Instructions
During your live presentation:
1. Open the **Live SOC Escalation Triage Console** in the dashboard.
2. In the **Search Entity ID** input box, type `user_018`.
3. Highlight the event at timestamp `18:22:41`:
   - Point out that under standard rule profiling, this event was buried at rank **#732** (outside budget).
   - Point out that the sequence model detected a `res_recent = 7` multi-step resource traversal, applying a monotonic percentile boost that elevated the event to rank **#705**—surfacing it directly into the operational triage queue.
