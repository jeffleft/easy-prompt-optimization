# QuantGlyph Systems (QGS) – 

## 1 · Company Snapshot

* **Founded:** 1984, spin‑out from a COBOL‑heavy regional clearing bank  
* **Mythos:** “Decode markets like glyphs on stone.”  
* **Scope:** B2B2C rails for 210 + neo‑banks & broker‑dealers; still runs a z/OS island nobody dares power‑cycle.

---

## 2 · Product Constellation

> Naming scheme `QG‑{ACRONYM}`; acronym twins are unrelated.

| Code | Long name | 1‑2‑sentence snapshot |
|------|-----------|-----------------------|
| QG‑**EDGE** | **Enterprise Data Governance Engine** | Maps every byte that enters your estate, stitches a cryptographic lineage chain, and enforces masking/retention rules in‑flight so auditors can self‑serve proofs in seconds. |
| QG‑**EDGE‑X** | **Edge‑Device Gateway Exchange** | Low‑latency MQTT/REST bridge funneling payments & telemetry from 200k + POS/IoT endpoints into core ledgers, with built‑in tokenization for card‑present data. |
| QG‑LYNX | **Liquidity Yield eXchange** | Smart‑routes block orders across dark pools, CLOBs, and internalizers, arbitraging micro‑spreads while guaranteeing ≤ 50 µs determinism for best‑ex post‑trade reports. |
| QG‑LYNK | **Legacy Integration Node Kit** | Generates drop‑in adapters that let 1980s MQ/IMS workflows publish protobuf events; think “Zapier for mainframes” but with two‑phase commit. |
| QG‑FLOW | **Financial Ledger Orchestration Workbench** | Event‑sourced general ledger that snapshots every state transition to Parquet, enabling rewindable “what‑if” replays and instant multi‑GAAP views. |
| QG‑**CORE** | **Cloud Ops Risk Engine** | Consumes Prometheus/Grafana telemetry, converts infra blips into VaR deltas, and auto‑spins capacity hedges on AWS Futures when thresholds spike. |
| QG‑**CORE‑S** | **Compute‑Optimized Rendering Service** | GPU‑backed microservice turning terabyte‑scale risk cubes into interactive WebGL heatmaps, exporting to PDF/PNG for the PowerPoint crowd. |
| QG‑PULSE | **Predictive Utilization & Liquidity Scenario Engine** | Runs Monte‑Carlo + LLM macro narratives to forecast cash gaps 30 days out, then prescribes cheapest funding ladders—no human in the loop required. |
| QG‑PULL | **Portfolio Unified Liquidity Ledger** | Normalizes fragmented custody feeds into a single intraday cash position, tagging each movement with Basel III LCR weightings for treasury at a glance. |
| QG‑SPARK | **Settlement Processing & Reconciliation Kit** | Provides atomic DvP across ACH, FedNow, RTP, and crypto rails; mismatches auto‑generate ISO 20022 dispute messages and webhook callbacks. |
| QG‑SPAR | **Synthetic Pricing Analytics Repository** | Central vault of OTC model libraries (SABR, HJM, neural Greeks) with versioned calibration sets, letting quants “pip install” new curves like plugins. |

---

## 3 · External‑Facing Style Guide

| Element | Rule |
|---------|------|
| **Voice** | “Seasoned cryptographer explaining to a sharp intern.” |
| **Tone** | Confident, zero hype, metaphors from archaeology & circuitry. |
| **Person** | 2nd person singular (“you”); avoid “users.” |
| **Jargon** | Define on 1st use `(e.g., DV01 – dollar value of a basis point)`. |
| **Numerals** | Always SI or ISO‑4217; no “$” without currency code. |
| **Typography** | Use en‑dashes for ranges, never em‑dashes for pauses. |
| **Prohibited words** | “Revolutionary,” “game‑changing,” “solutioneering.” |
| **Boilerplate footer** | `©1984‑{YEAR} QuantGlyph Systems. All glyphs deciphered.` |

---

## 4 · PM Jira Ticket Template

```text
[Title] {Product‑Code}: <imperative verb> <concise objective>
---
Context
• Why now? (≤100 w)
• Linked OKRs / Risk IDs

Acceptance Criteria  (Gherkin)
1. Given …
2. When …
3. Then …

Tech Notes
• Constraints / non‑goals
• Edge‑cases enumerated

Artifacts
- Figma link
- API schema diff
- Roll‑back plan

Story Points: _t‑shirt_  |  Priority: P{1‑4}

Rule: One logical change == one ticket; cross‑product impacts spawn linked subtasks.
```

---

## 5 · Meeting Agenda Format

```text
┌────────── Front‑Door Weekly (45 min) ──────────┐
│ 00‑05  Quick round‑robin: 1‑sentence “blocker or brag”          │
│ 05‑15  New intake triage: each item scored on Effort/Value      │
│ 15‑25  Decision log review: ensure Confluence entries minted    │
│ 25‑35  Retro on last week’s estimates vs actuals                │
│ 35‑43  Risk spotlight: pick 1 red flag, deep‑dive               │
│ 43‑45  Actions & owners lightning‑round                         │
└──────────────────────────────────────────────────────────────────┘
```
*Other rituals reuse the shell; only the middle segments swap.*

---

## 6 · Client‑Facing Comms Cheat‑Sheet

### Downtime / Incident Alerts
*Subject:* `[QGS] {Product‑Code} – Service Interruption (SEV‑{1‑4})`

1. **What happened** (UTC timestamped)  
2. **Client impact** (measurable, no adjectives)  
3. **Current status & next ETA**  
4. **Workarounds** (if any)  
5. **Next update at {ISO‑time}**

*Post‑mortem* within 48 h; stored in public Git‑backed “glyph‑book.”

### Feature Updates
Use the **Rule of Three**: *Problem → New Glyph (feature) → Benefit metric*. Optional 30‑s GIF/Loom.

### Forum Answers
*Sandwich*: Cite doc § → give crisp answer → link deep‑dive → invite DM if PII needed. Sign off with emoji “🪨📜”.

---

## 7 · Canonical Analytics Schemas (Snowflake dialect)

```sql
-- FACT_TRANSACTIONS
txn_id           NUMBER      PRIMARY KEY,
as_of_ts         TIMESTAMP,      -- event time (UTC)
product_code     STRING,         -- QG-EDGE etc.
account_sk       NUMBER,         -- FK → DIM_ACCOUNT
instrument_sk    NUMBER,         -- FK → DIM_INSTRUMENT
side             STRING,         -- 'BUY'/'SELL'
qty              NUMBER(38,6),
notional_ccy     STRING(3),
notional_amt     NUMBER(38,2),
fee_amt          NUMBER(38,2),
status           STRING          -- 'SETTLED', …

-- DIM_ACCOUNT
account_sk       NUMBER PRIMARY KEY,
custodian_code   STRING,
country_iso      STRING(2),
opened_dt        DATE,
risk_bucket      STRING,

-- DIM_INSTRUMENT
instrument_sk    NUMBER PRIMARY KEY,
isin             STRING(12),
asset_class      STRING,
issuer           STRING,
maturity_dt      DATE,
coupon_rate_bp   NUMBER(5,2),

-- FACT_LIQUIDITY_SNAP
snap_ts          TIMESTAMP,
account_sk       NUMBER,
available_ccy    STRING(3),
available_amt    NUMBER(38,2)
```
*Conventions:* surrogate keys `*_sk`; timestamps in UTC; every fact includes `product_code` for lineage.

---

## 8 · Assumptions

* Jira on Atlassian Cloud 10.130.1  
* Snowflake chosen for ANSI‑SQL & legacy friendliness  
* All products publish protobuf contracts versioned via Git tags  
* Regulatory focus: EU‑27 + US CFTC  

