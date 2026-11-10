import os
import io
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
OUTPUTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("BHARATCODE_API_KEY", "").strip()
API_URL = "https://bharatcode.ai/api/model/v1/chat/completions"
MODEL = "qwen-3.8-27b"
DB_PATH = OUTPUTS / "portiq.db"

st.set_page_config(
    page_title="PortIQ | Logistics Intelligence",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0b1220;
    color: #e5edf7;
}
[data-testid="stSidebar"] {
    background: #101b2c;
    border-right: 1px solid #24334a;
}
[data-testid="stHeader"] {
    background: rgba(11,18,32,0.95);
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #f0f6ff !important;
}
p, label, span {
    color: #c5d2e2;
}
div[data-testid="stMetric"] {
    background: #132238;
    border: 1px solid #253b56;
    padding: 17px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] {
    color: #9fb2c9;
}
div[data-testid="stMetricValue"] {
    color: #f4f8ff;
}
.stButton > button {
    background: #087e8b;
    color: white;
    border: 1px solid #1395a1;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
    background: #0b9aa6;
    color: white;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #263a53;
    border-radius: 10px;
}
section[data-testid="stFileUploader"] {
    background: #132238;
    border: 1px dashed #36516f;
    border-radius: 12px;
    padding: 12px;
}
.piq-hero {
    background: linear-gradient(120deg, #122b43, #123a48 65%, #0b5960);
    border: 1px solid #28566b;
    padding: 27px;
    border-radius: 17px;
    margin-bottom: 20px;
}
.piq-eyebrow {
    color: #61d5d2;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}
.piq-title {
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: 36px;
    font-weight: 800;
    margin-top: 7px;
}
.piq-sub {
    color: #c2d9e5;
    font-size: 15px;
    max-width: 760px;
}
.piq-card {
    background: #132238;
    border: 1px solid #263c57;
    border-radius: 12px;
    padding: 17px;
    min-height: 115px;
}
.piq-card-title {
    color: #8da8c2;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
.piq-card-value {
    color: #f3f7fc;
    font-size: 25px;
    font-weight: 800;
    margin-top: 9px;
}
.piq-muted {
    color: #8ea4bd;
    font-size: 12px;
}
hr {
    border-color: #26384f;
}
</style>
""", unsafe_allow_html=True)


REQUIRED_COLUMNS = [
    "shipment_id",
    "origin_port",
    "destination_port",
    "planned_arrival",
    "estimated_arrival",
    "priority"
]


def make_demo_data(n=180):
    rng = np.random.default_rng(42)
    today = pd.Timestamp.today().normalize()

    ports = [
        "Chennai", "Singapore", "Shanghai", "Jebel Ali",
        "Rotterdam", "Colombo", "Mumbai", "Hamburg"
    ]

    vessels = [
        "MV Ocean Pioneer", "MV Eastern Star",
        "MV Blue Horizon", "MV Global Carrier",
        "MV Pacific Trader", "MV Meridian"
    ]

    customers = [
        "Apex Manufacturing", "Global Retail Group",
        "Northstar Industries", "Vertex Electronics",
        "BlueWave Trading", "Prime Auto Components",
        "Continental Supplies"
    ]

    records = []

    for i in range(n):
        origin, destination = rng.choice(
            ports, size=2, replace=False
        )

        planned = today + pd.Timedelta(
            days=int(rng.integers(-8, 13))
        )

        delay = int(rng.choice(
            [0, 0, 0, 3, 6, 12, 18, 24, 36, 48, 72],
            p=[.20, .12, .10, .10, .10, .10, .08, .07, .06, .04, .03]
        ))

        estimated = planned + pd.Timedelta(hours=delay)

        priority = rng.choice(
            ["Critical", "High", "Medium", "Low"],
            p=[.12, .28, .40, .20]
        )

        transit = int(rng.integers(3, 25))
        planned_departure = planned - pd.Timedelta(days=transit)

        deadline = planned + pd.Timedelta(
            hours=int(rng.integers(12, 72))
        )

        actual = pd.NaT
        if estimated < today:
            actual = estimated + pd.Timedelta(
                hours=int(rng.integers(-3, 12))
            )

        records.append({
            "shipment_id": f"PIQ-{10001+i}",
            "origin_port": origin,
            "destination_port": destination,
            "vessel": rng.choice(vessels),
            "customer": rng.choice(customers),
            "planned_departure": planned_departure,
            "planned_arrival": planned,
            "estimated_arrival": estimated,
            "actual_arrival": actual,
            "delivery_deadline": deadline,
            "priority": priority,
            "transport_mode": rng.choice(
                ["Ocean", "Multimodal", "Road"]
            ),
            "container_type": rng.choice(
                ["20GP", "40GP", "40HC", "Reefer"]
            ),
            "cargo_value_usd": int(
                rng.integers(10000, 500000)
            ),
            "port_congestion_index": round(
                float(rng.uniform(0.1, 0.95)), 2
            ),
            "transport_capacity": int(
                rng.integers(55, 100)
            )
        })

    return pd.DataFrame(records)


def normalize_data(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    aliases = {
        "shipment_no": "shipment_id",
        "container_id": "shipment_id",
        "origin": "origin_port",
        "destination": "destination_port",
        "planned_eta": "planned_arrival",
        "eta": "estimated_arrival",
        "expected_arrival": "estimated_arrival",
        "actual_eta": "actual_arrival",
        "customer_name": "customer",
        "shipment_priority": "priority"
    }

    df = df.rename(columns={
        k: v for k, v in aliases.items()
        if k in df.columns and v not in df.columns
    })

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    optional_defaults = {
        "vessel": "Not specified",
        "customer": "Unspecified customer",
        "planned_departure": pd.NaT,
        "actual_arrival": pd.NaT,
        "delivery_deadline": pd.NaT,
        "transport_mode": "Ocean",
        "container_type": "Unknown",
        "cargo_value_usd": 0,
        "port_congestion_index": 0.5,
        "transport_capacity": 85,
        "warehouse_processing_delay_hours": 0
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in [
        "planned_departure", "planned_arrival",
        "estimated_arrival", "actual_arrival",
        "delivery_deadline"
    ]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in [
        "cargo_value_usd", "port_congestion_index",
        "transport_capacity", "warehouse_processing_delay_hours"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["priority"] = (
        df["priority"].astype(str).str.title()
    )

    df["shipment_id"] = df["shipment_id"].astype(str)

    df = df.drop_duplicates(
        subset=["shipment_id"], keep="last"
    )

    df["origin_port"] = df["origin_port"].fillna("Unknown")
    df["destination_port"] = df["destination_port"].fillna("Unknown")
    df["priority"] = df["priority"].where(
        df["priority"].isin(["Critical", "High", "Medium", "Low"]),
        "Medium"
    )

    return df


def calculate_risk(df, extra_delay=0):
    result = df.copy()
    now = pd.Timestamp.now()

    planned = result["planned_arrival"]
    estimated = result["estimated_arrival"] + pd.to_timedelta(
        extra_delay, unit="h"
    )

    result["simulated_eta"] = estimated

    result["delay_hours"] = (
        (estimated - planned).dt.total_seconds() / 3600
    ).fillna(0).clip(lower=0)

    if "actual_arrival" in result.columns:
        actual_mask = result["actual_arrival"].notna()
        result.loc[actual_mask, "delay_hours"] = (
            (
                result.loc[actual_mask, "actual_arrival"]
                - result.loc[actual_mask, "planned_arrival"]
            ).dt.total_seconds() / 3600
        ).clip(lower=0)

    priority_points = {
        "Critical": 35,
        "High": 25,
        "Medium": 15,
        "Low": 5
    }

    result["priority_points"] = (
        result["priority"].map(priority_points).fillna(15)
    )

    result["delay_points"] = np.minimum(
        result["delay_hours"].fillna(0) * 0.8, 35
    )

    congestion = pd.to_numeric(
        result["port_congestion_index"], errors="coerce"
    ).fillna(0.5).clip(0, 1)

    result["congestion_points"] = congestion * 20

    capacity = pd.to_numeric(
        result["transport_capacity"], errors="coerce"
    ).fillna(85).clip(0, 100)

    result["capacity_points"] = (
        (100 - capacity) / 100
    ) * 10

    result["risk_score"] = (
        result["priority_points"]
        + result["delay_points"]
        + result["congestion_points"]
        + result["capacity_points"]
    ).round().clip(0, 100)

    result["risk_level"] = pd.cut(
        result["risk_score"],
        bins=[-1, 29, 54, 74, 100],
        labels=["Low", "Moderate", "High", "Critical"]
    ).astype(str)

    result["operational_status"] = np.select(
        [
            result["actual_arrival"].notna(),
            result["delay_hours"] >= 24,
            result["delay_hours"] > 0,
            result["simulated_eta"] < now
        ],
        [
            "Arrived",
            "Delayed",
            "At Risk",
            "Arrival Overdue"
        ],
        default="On Schedule"
    )

    warehouse_delay = pd.to_numeric(
        result["warehouse_processing_delay_hours"], errors="coerce"
    ).fillna(0).clip(lower=0)

    result["deadline_buffer_hours"] = (
        (
            result["delivery_deadline"]
            - result["simulated_eta"]
        ).dt.total_seconds() / 3600
    ) - warehouse_delay

    result["deadline_buffer_hours"] = (
        result["deadline_buffer_hours"].fillna(999)
    )

    result["deadline_points"] = (
        ((24 - result["deadline_buffer_hours"]) / 24) * 10
    ).clip(lower=0, upper=10)

    result["risk_score"] = (
        result["risk_score"] + result["deadline_points"]
    ).round().clip(0, 100)

    result["risk_level"] = pd.cut(
        result["risk_score"],
        bins=[-1, 29, 54, 74, 100],
        labels=["Low", "Moderate", "High", "Critical"]
    ).astype(str)

    return result


def run_anomaly_detection(df):
    result = df.copy()

    features = pd.DataFrame({
        "delay_hours": result["delay_hours"].fillna(0),
        "cargo_value": result["cargo_value_usd"].fillna(0),
        "congestion": result["port_congestion_index"].fillna(0.5),
        "capacity": result["transport_capacity"].fillna(85),
        "risk_score": result["risk_score"].fillna(0)
    })

    if len(features) < 10:
        result["anomaly_flag"] = "Insufficient data"
        return result

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=120,
        contamination=0.08,
        random_state=42
    )

    predictions = model.fit_predict(scaled)

    result["anomaly_flag"] = np.where(
        predictions == -1, "Unusual pattern", "Normal pattern"
    )

    return result


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                shipment_id TEXT,
                event_type TEXT,
                decision TEXT,
                details TEXT
            )
        """)


def save_decision(shipment_id, event_type, decision, details):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO decisions
            (created_at, shipment_id, event_type, decision, details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                str(shipment_id),
                event_type,
                decision,
                json.dumps(details, default=str)
            )
        )


def read_decisions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM decisions ORDER BY id DESC",
            conn
        )


def build_recovery_options(row):
    delay = float(row["delay_hours"])
    cargo_value = float(row["cargo_value_usd"])
    congestion = float(row["port_congestion_index"])
    priority = row["priority"]

    urgency = (
        "Immediate"
        if priority in ["Critical", "High"] or delay >= 24
        else "Planned"
    )

    options = [
        {
            "action": "Priority handling request",
            "estimated_time_saved_hours": min(8, delay * 0.20),
            "estimated_cost_usd": round(
                max(100, cargo_value * 0.0015), 2
            ),
            "feasibility": "Medium",
            "reason": "Request priority processing at the destination terminal."
        },
        {
            "action": "Alternative inland transport",
            "estimated_time_saved_hours": min(14, delay * 0.35),
            "estimated_cost_usd": round(
                max(250, cargo_value * 0.003), 2
            ),
            "feasibility": "Medium",
            "reason": "Evaluate an alternative onward transport arrangement."
        },
        {
            "action": "Customer deadline coordination",
            "estimated_time_saved_hours": 0,
            "estimated_cost_usd": 0,
            "feasibility": "High",
            "reason": "Coordinate a revised delivery window with the receiving party."
        },
        {
            "action": "Monitor and retain current route",
            "estimated_time_saved_hours": 0,
            "estimated_cost_usd": 0,
            "feasibility": "High",
            "reason": "Avoid unnecessary recovery cost when the impact is manageable."
        }
    ]

    if congestion > 0.75:
        options.insert(1, {
            "action": "Evaluate alternate terminal",
            "estimated_time_saved_hours": min(18, delay * 0.30),
            "estimated_cost_usd": round(
                max(500, cargo_value * 0.004), 2
            ),
            "feasibility": "Requires validation",
            "reason": "High congestion may justify evaluating an alternative terminal."
        })

    for option in options:
        option["urgency"] = urgency
        option["assumption"] = (
            "Illustrative estimate, not a live carrier quote."
        )

    return pd.DataFrame(options)


def call_bharatcode(prompt):
    if not API_KEY:
        return None, "API key is not configured in the local .env file."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are PortIQ's logistics operations analyst. "
                    "Use only the supplied operational evidence. "
                    "Do not invent shipment facts, live tracking, costs, "
                    "or carrier information. Distinguish assumptions "
                    "from confirmed data. Give concise, practical "
                    "recommendations and explain uncertainty."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=75
        )

        if response.status_code == 402:
            return None, (
                "BharatCode returned HTTP 402: payment or account entitlement "
                "is required for chat-completion access. Your model catalogue "
                "test can still succeed while inference access is inactive. "
                "PortIQ's local analysis mode is available below."
            )

        if response.status_code == 401:
            return None, "BharatCode authentication failed (HTTP 401). Check the local API key."
        if response.status_code == 403:
            return None, "BharatCode access is not enabled for this account (HTTP 403)."
        if response.status_code == 429:
            return None, "BharatCode rate limit reached (HTTP 429). Please retry later."
        if response.status_code == 503:
            return None, "BharatCode is temporarily unavailable (HTTP 503). Please retry."

        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        if not answer:
            return None, "The model returned an empty response."

        return answer, None

    except requests.exceptions.Timeout:
        return None, "The AI request timed out. Try again."
    except requests.exceptions.RequestException as exc:
        return None, f"AI service request failed: {exc}"
    except (KeyError, IndexError, ValueError, TypeError):
        return None, "The AI service returned an unexpected response."


def build_local_operations_analysis(df, question=""):
    ranked = df.sort_values(
        ["risk_score", "delay_hours", "deadline_buffer_hours"],
        ascending=[False, False, True]
    ).head(5)

    lines = [
        "### PortIQ local evidence-based analysis",
        "",
        "BharatCode inference is currently unavailable. This analysis was generated "
        "locally from PortIQ's rule-based risk engine; it is not an LLM response.",
        "",
        f"**Question:** {question or 'Which shipments need attention first?'}",
        "",
        "#### Operational overview",
        f"- Shipments analysed: **{len(df)}**",
        f"- High/critical risk: **{int(df['risk_level'].isin(['High', 'Critical']).sum())}**",
        f"- Delayed shipments: **{int((df['delay_hours'] >= 24).sum())}**",
        f"- Average rule-based risk score: **{df['risk_score'].mean():.1f}/100**",
        "",
        "#### Five shipments to investigate first"
    ]

    for _, row in ranked.iterrows():
        lines.append(
            f"- **{row['shipment_id']}** ({row['origin_port']} → "
            f"{row['destination_port']}) — {row['risk_level']} risk, "
            f"score {int(row['risk_score'])}/100, "
            f"priority {row['priority']}, "
            f"delay {row['delay_hours']:.1f} hours, "
            f"deadline buffer {row['deadline_buffer_hours']:.1f} hours."
        )

    lines.extend([
        "",
        "#### Recommended investigation order",
        "1. Confirm the latest ETA and shipment milestone with the responsible operator.",
        "2. Check destination-port congestion and available onward transport capacity.",
        "3. Review delivery-deadline exposure for Critical and High priority shipments.",
        "4. Compare recovery options and validate feasibility, cost and time estimates with actual providers.",
        "",
        "**Limitations:** Findings are based only on the currently loaded dataset and PortIQ's transparent rules. "
        "Risk scores are indicators, not calibrated disruption probabilities. Recovery estimates are illustrative."
    ])

    return "\n".join(lines)


def build_local_recovery_comparison(row, options):
    ranked = options.sort_values(
        ["estimated_time_saved_hours", "feasibility"],
        ascending=[False, True]
    )

    best = ranked.iloc[0]
    return (
        "### Local recovery comparison\n\n"
        "BharatCode inference is unavailable, so this comparison uses the "
        "illustrative estimates already calculated by PortIQ.\n\n"
        f"**Shipment:** {row['shipment_id']}  \n"
        f"**Current risk:** {row['risk_level']} ({int(row['risk_score'])}/100)  \n"
        f"**Current delay:** {row['delay_hours']:.1f} hours\n\n"
        f"**Highest estimated time-saving option:** {best['action']}  \n"
        f"- Estimated time saved: {best['estimated_time_saved_hours']:.1f} hours  \n"
        f"- Illustrative cost: USD {best['estimated_cost_usd']:,.2f}  \n"
        f"- Feasibility: {best['feasibility']}  \n"
        f"- Reason: {best['reason']}\n\n"
        "Validate all options with the carrier, terminal, warehouse and transport provider "
        "before making an operational decision. These are not live quotes or confirmed capacity."
    )


def make_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def page_header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="piq-hero">
            <div class="piq-eyebrow">{eyebrow}</div>
            <div class="piq-title">{title}</div>
            <div class="piq-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value, caption):
    st.markdown(
        f"""
        <div class="piq-card">
            <div class="piq-card-title">{label}</div>
            <div class="piq-card-value">{value}</div>
            <div class="piq-muted">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


init_db()

if "source_df" not in st.session_state:
    st.session_state.source_df = make_demo_data()

if "scenario_delay" not in st.session_state:
    st.session_state.scenario_delay = 0

if "source_name" not in st.session_state:
    st.session_state.source_name = "Built-in demonstration dataset"

if "last_simulation" not in st.session_state:
    st.session_state.last_simulation = None

try:
    raw_df = normalize_data(st.session_state.source_df)
    base_df = calculate_risk(raw_df)
    base_df = run_anomaly_detection(base_df)
except Exception as exc:
    st.error(f"Dataset processing error: {exc}")
    st.stop()

risk_df = calculate_risk(
    raw_df,
    extra_delay=st.session_state.scenario_delay
)

risk_df = run_anomaly_detection(risk_df)

nav_options = [
    "Overview",
    "Shipment Intelligence",
    "Risk Intelligence",
    "Disruption Simulator",
    "AI Operations Analyst",
    "Recovery Planner",
    "Decision Reports",
    "Data & Model Health"
]

with st.sidebar:
    st.markdown("## ⚓ PortIQ")
    st.caption("LOGISTICS INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        nav_options,
        label_visibility="visible"
    )

    st.divider()
    st.caption("DATA SOURCE")
    st.write(st.session_state.source_name)

    if API_KEY:
        st.success("AI key configured")
    else:
        st.warning("AI key not configured")

    st.caption("PortIQ prototype | Build League 2026")
    st.caption("Demonstration data is simulated.")

if page == "Overview":
    page_header(
        "OPERATIONS CONTROL TOWER",
        "Logistics at a glance.",
        "A unified view of shipment performance, disruption exposure and operational priorities."
    )

    total = len(risk_df)
    delayed = int((risk_df["operational_status"] == "Delayed").sum())
    at_risk = int(
        risk_df["risk_level"].isin(["High", "Critical"]).sum()
    )
    anomalies = int(
        (risk_df["anomaly_flag"] == "Unusual pattern").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total shipments", f"{total:,}")
    with c2:
        st.metric("Delayed shipments", f"{delayed:,}")
    with c3:
        st.metric("High / critical risk", f"{at_risk:,}")
    with c4:
        st.metric("Unusual patterns", f"{anomalies:,}")

    st.divider()

    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Shipment risk distribution")
        counts = (
            risk_df["risk_level"]
            .value_counts()
            .reindex(["Low", "Moderate", "High", "Critical"])
            .fillna(0)
            .reset_index()
        )
        counts.columns = ["Risk level", "Shipments"]

        fig = px.bar(
            counts,
            x="Risk level",
            y="Shipments",
            color="Risk level",
            color_discrete_map={
                "Low": "#16a085",
                "Moderate": "#d6a84f",
                "High": "#e07b39",
                "Critical": "#db5361"
            }
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Operational status")
        status_counts = (
            risk_df["operational_status"]
            .value_counts()
            .reset_index()
        )
        status_counts.columns = ["Status", "Shipments"]

        fig = px.pie(
            status_counts,
            names="Status",
            values="Shipments",
            hole=0.62
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Priority shipment watchlist")

    watchlist = risk_df.sort_values(
        ["risk_score", "delay_hours"],
        ascending=False
    ).head(10)

    display_cols = [
        "shipment_id", "origin_port", "destination_port",
        "priority", "delay_hours", "risk_score",
        "risk_level", "operational_status"
    ]

    st.dataframe(
        watchlist[display_cols],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Risk scores are rule-based operational indicators, "
        "not externally validated probabilities."
    )

elif page == "Shipment Intelligence":
    page_header(
        "MODULE 01",
        "Shipment intelligence.",
        "Inspect shipment records, validate data and explore operational milestones."
    )

    st.subheader("Import shipment records")

    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"]
    )

    a, b = st.columns([1, 1])

    with a:
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    incoming = pd.read_csv(uploaded)
                else:
                    incoming = pd.read_excel(uploaded)

                normalized = normalize_data(incoming)
                st.session_state.source_df = normalized
                st.session_state.source_name = uploaded.name
                st.session_state.scenario_delay = 0
                st.success(
                    f"Successfully loaded {len(normalized)} shipment records."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Upload could not be processed: {exc}")

    with b:
        if st.button("Restore demonstration dataset"):
            st.session_state.source_df = make_demo_data()
            st.session_state.source_name = "Built-in demonstration dataset"
            st.session_state.scenario_delay = 0
            st.rerun()

    st.divider()

    search = st.text_input(
        "Search shipment ID, port, vessel or customer"
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        priorities = st.multiselect(
            "Priority",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"]
        )

    with f2:
        risk_levels = st.multiselect(
            "Risk level",
            ["Critical", "High", "Moderate", "Low"],
            default=["Critical", "High", "Moderate", "Low"]
        )

    with f3:
        statuses = st.multiselect(
            "Operational status",
            sorted(risk_df["operational_status"].unique()),
            default=sorted(risk_df["operational_status"].unique())
        )

    filtered = risk_df[
        risk_df["priority"].isin(priorities)
        & risk_df["risk_level"].isin(risk_levels)
        & risk_df["operational_status"].isin(statuses)
    ].copy()

    if search.strip():
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)
        filtered = filtered[mask]

    columns = [
        "shipment_id", "customer", "origin_port",
        "destination_port", "vessel", "priority",
        "planned_arrival", "estimated_arrival",
        "delay_hours", "operational_status",
        "risk_score", "risk_level"
    ]

    st.dataframe(
        filtered[columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered shipments",
        make_csv_download(filtered),
        file_name="portiq_shipments.csv",
        mime="text/csv"
    )

    st.subheader("Shipment detail")

    selected = st.selectbox(
        "Choose a shipment",
        risk_df["shipment_id"].tolist()
    )

    row = risk_df[
        risk_df["shipment_id"] == selected
    ].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Risk score", int(row["risk_score"]))
    d2.metric("Delay hours", round(row["delay_hours"], 1))
    d3.metric("Priority", row["priority"])
    d4.metric("Risk level", row["risk_level"])

    detail = pd.DataFrame({
        "Field": [
            "Shipment ID", "Customer", "Vessel",
            "Origin", "Destination", "Planned arrival",
            "Estimated arrival", "Delivery deadline",
            "Cargo value (USD)", "Transport mode"
        ],
        "Value": [
            row["shipment_id"], row["customer"], row["vessel"],
            row["origin_port"], row["destination_port"],
            row["planned_arrival"], row["estimated_arrival"],
            row["delivery_deadline"], row["cargo_value_usd"],
            row["transport_mode"]
        ]
    })

    st.dataframe(detail, use_container_width=True, hide_index=True)

elif page == "Risk Intelligence":
    page_header(
        "MODULE 02",
        "Risk intelligence.",
        "Identify exceptions, quantify operational exposure and prioritise shipments requiring attention."
    )

    high_risk = risk_df[
        risk_df["risk_level"].isin(["High", "Critical"])
    ].copy()

    critical = risk_df[
        risk_df["risk_level"] == "Critical"
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk shipments", len(high_risk))
    c2.metric("Critical shipments", len(critical))
    c3.metric(
        "Average risk score",
        round(risk_df["risk_score"].mean(), 1)
    )

    st.subheader("Risk ranking")

    ranked = risk_df.sort_values(
        ["risk_score", "delay_hours"],
        ascending=False
    )

    st.dataframe(
        ranked[[
            "shipment_id", "origin_port", "destination_port",
            "priority", "delay_hours", "deadline_buffer_hours",
            "risk_score", "risk_level", "anomaly_flag"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Why is a shipment at risk?")

    selected = st.selectbox(
        "Select a shipment to inspect",
        ranked["shipment_id"].tolist(),
        key="risk_detail"
    )

    row = ranked[
        ranked["shipment_id"] == selected
    ].iloc[0]

    reasons = []

    if row["delay_hours"] > 0:
        reasons.append(
            f"Estimated delay is {row['delay_hours']:.1f} hours."
        )

    if row["priority"] in ["Critical", "High"]:
        reasons.append(
            f"Shipment priority is {row['priority']}."
        )

    if row["port_congestion_index"] >= 0.75:
        reasons.append(
            "Port congestion index is elevated."
        )

    if row["transport_capacity"] < 70:
        reasons.append(
            "Available transport capacity indicator is low."
        )

    if row["deadline_buffer_hours"] < 24:
        reasons.append(
            "Estimated arrival is within 24 hours of the delivery deadline."
        )

    if not reasons:
        reasons.append(
            "No major threshold breach was identified by the current rules."
        )

    for reason in reasons:
        st.write("• " + reason)

    st.caption(
        "The risk score combines priority, delay, congestion "
        "and transport capacity using transparent weighted rules."
    )

    st.download_button(
        "Export risk register",
        make_csv_download(ranked),
        file_name="portiq_risk_register.csv",
        mime="text/csv"
    )

elif page == "Disruption Simulator":
    page_header(
        "MODULE 04",
        "Disruption simulator.",
        "Model hypothetical disruptions and compare the resulting shipment risk exposure."
    )

    st.info(
        "Scenarios are hypothetical planning exercises. "
        "They do not change the original shipment dataset."
    )

    scenario_type = st.selectbox(
        "Disruption type",
        [
            "Port congestion",
            "Vessel schedule delay",
            "Transport capacity shortage",
            "Warehouse processing delay"
        ]
    )

    extra_hours = st.slider(
        "Additional delay (hours)",
        min_value=0,
        max_value=120,
        value=24,
        step=6
    )

    affected_port = st.selectbox(
        "Affected port",
        ["All ports"] + sorted(
            set(risk_df["origin_port"].dropna().astype(str))
            | set(risk_df["destination_port"].dropna().astype(str))
        )
    )

    if st.button("Run disruption simulation"):
        simulated = raw_df.copy()

        if affected_port != "All ports":
            mask = (
                (simulated["origin_port"] == affected_port)
                | (simulated["destination_port"] == affected_port)
            )
        else:
            mask = pd.Series(True, index=simulated.index)

        affected_count = int(mask.sum())

        if scenario_type == "Port congestion":
            congestion_increase = min(0.35, extra_hours / 120 * 0.35)
            simulated.loc[mask, "port_congestion_index"] = (
                pd.to_numeric(
                    simulated.loc[mask, "port_congestion_index"],
                    errors="coerce"
                ).fillna(0.5) + congestion_increase
            ).clip(0, 1)
            simulated.loc[mask, "estimated_arrival"] += pd.to_timedelta(
                extra_hours * 0.35, unit="h"
            )
            impact_description = (
                f"Congestion index increased by up to {congestion_increase:.2f}; "
                f"ETA extended by {extra_hours * 0.35:.1f} hours."
            )

        elif scenario_type == "Vessel schedule delay":
            simulated.loc[mask, "estimated_arrival"] += pd.to_timedelta(
                extra_hours, unit="h"
            )
            impact_description = (
                f"Estimated arrival extended by {extra_hours} hours."
            )

        elif scenario_type == "Transport capacity shortage":
            capacity_reduction = min(60, extra_hours * 0.5)
            simulated.loc[mask, "transport_capacity"] = (
                pd.to_numeric(
                    simulated.loc[mask, "transport_capacity"],
                    errors="coerce"
                ).fillna(85) - capacity_reduction
            ).clip(0, 100)
            simulated.loc[mask, "estimated_arrival"] += pd.to_timedelta(
                extra_hours * 0.25, unit="h"
            )
            impact_description = (
                f"Transport capacity reduced by {capacity_reduction:.1f} points; "
                f"ETA extended by {extra_hours * 0.25:.1f} hours."
            )

        else:
            if "warehouse_processing_delay_hours" not in simulated.columns:
                simulated["warehouse_processing_delay_hours"] = 0.0
            simulated.loc[mask, "warehouse_processing_delay_hours"] = (
                pd.to_numeric(
                    simulated.loc[mask, "warehouse_processing_delay_hours"],
                    errors="coerce"
                ).fillna(0) + extra_hours
            )
            impact_description = (
                f"Downstream warehouse processing delay increased by {extra_hours} hours."
            )

        before = calculate_risk(raw_df)
        after = calculate_risk(simulated)

        st.session_state.last_simulation = {
            "type": scenario_type,
            "hours": extra_hours,
            "port": affected_port,
            "before": before,
            "after": after,
            "affected": affected_count,
            "impact": impact_description
        }

        save_decision(
            "ALL",
            "Disruption simulation",
            f"{scenario_type}: {extra_hours} hour scenario",
            {
                "affected_port": affected_port,
                "affected_shipments": affected_count,
                "impact": impact_description
            }
        )

    sim = st.session_state.last_simulation

    if sim is not None:
        before = sim["before"]
        after = sim["after"]

        st.divider()
        st.subheader("Scenario results")
        st.write(sim.get("impact", "Scenario impact calculated."))
        st.caption(
            "Scenario impacts are simplified planning assumptions, not calibrated forecasts."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Affected shipments",
            sim["affected"]
        )

        before_high = int(
            before["risk_level"].isin(["High", "Critical"]).sum()
        )
        after_high = int(
            after["risk_level"].isin(["High", "Critical"]).sum()
        )

        c2.metric(
            "High / critical risk",
            after_high,
            delta=after_high - before_high
        )

        c3.metric(
            "Average risk score",
            round(after["risk_score"].mean(), 1),
            delta=round(
                after["risk_score"].mean()
                - before["risk_score"].mean(), 1
            )
        )

        comparison = pd.DataFrame({
            "Risk category": ["Low", "Moderate", "High", "Critical"],
            "Before": [
                (before["risk_level"] == x).sum()
                for x in ["Low", "Moderate", "High", "Critical"]
            ],
            "After": [
                (after["risk_level"] == x).sum()
                for x in ["Low", "Moderate", "High", "Critical"]
            ]
        })

        chart_data = comparison.melt(
            id_vars="Risk category",
            var_name="Scenario",
            value_name="Shipments"
        )

        fig = px.bar(
            chart_data,
            x="Risk category",
            y="Shipments",
            color="Scenario",
            barmode="group"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Most affected shipments")

        delta_df = before[
            ["shipment_id", "risk_score", "risk_level"]
        ].merge(
            after[["shipment_id", "risk_score", "risk_level"]],
            on="shipment_id",
            suffixes=("_before", "_after")
        )

        delta_df["risk_increase"] = (
            delta_df["risk_score_after"]
            - delta_df["risk_score_before"]
        )

        st.dataframe(
            delta_df.sort_values(
                "risk_increase", ascending=False
            ).head(20),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download scenario results",
            make_csv_download(delta_df),
            file_name="portiq_scenario_results.csv",
            mime="text/csv"
        )

elif page == "AI Operations Analyst":
    page_header(
        "MODULE 03",
        "AI operations analyst.",
        "Ask questions about shipment risks, operational evidence and disruption scenarios."
    )

    st.info(
        "The AI analyst receives a compact summary of your current dataset. "
        "It does not independently access live carrier or port systems."
    )

    if not API_KEY:
        st.warning(
            "No API key is configured. Add your key to the local .env "
            "file after BharatCode activates your account."
        )
    else:
        st.success("A local API key is configured. Account access is checked when you send a request.")

    if "ai_question" not in st.session_state:
        st.session_state["ai_question"] = ""

    question = st.text_area(
        "Ask PortIQ",
        placeholder=(
            "Which shipments should the operations team prioritise "
            "today, and why?"
        ),
        height=120,
        key="ai_question"
    )

    if st.button("Analyse with BharatCode Qwen"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            context_df = risk_df.sort_values(
                "risk_score", ascending=False
            ).head(30)

            context = {
                "dataset_name": st.session_state.source_name,
                "total_shipments": len(risk_df),
                "risk_distribution": risk_df[
                    "risk_level"
                ].value_counts().to_dict(),
                "status_distribution": risk_df[
                    "operational_status"
                ].value_counts().to_dict(),
                "top_shipments": context_df[[
                    "shipment_id", "origin_port",
                    "destination_port", "priority",
                    "delay_hours", "risk_score",
                    "risk_level", "deadline_buffer_hours"
                ]].to_dict(orient="records"),
                "question": question
            }

            with st.spinner("Analysing operational evidence..."):
                answer, error = call_bharatcode(
                    json.dumps(context, default=str)
                )

            if error:
                st.warning(error)
                local_answer = build_local_operations_analysis(risk_df, question)
                st.markdown(local_answer)
                save_decision(
                    "DATASET",
                    "Local fallback analysis",
                    question,
                    {"response": local_answer, "provider_status": error}
                )
            else:
                st.markdown("### BharatCode AI analysis")
                st.markdown(answer)
                save_decision(
                    "DATASET",
                    "AI analysis",
                    question,
                    {"response": answer, "provider": "BharatCode", "model": MODEL}
                )

    st.divider()
    st.subheader("Suggested questions")

    suggestions = [
        "Which shipments should be prioritised and why?",
        "Explain the biggest operational risks in this dataset.",
        "What should the operations team investigate first?",
        "What are the trade-offs between the recovery options?"
    ]

    for item in suggestions:
        if st.button(item, key="suggest_" + item):
            st.session_state["ai_question"] = item
            st.rerun()

elif page == "Recovery Planner":
    page_header(
        "MODULE 05",
        "Recovery & action planner.",
        "Compare possible responses to shipment disruptions and understand their estimated trade-offs."
    )

    candidates = risk_df.sort_values(
        "risk_score", ascending=False
    )

    selected = st.selectbox(
        "Select a shipment",
        candidates["shipment_id"].tolist()
    )

    row = candidates[
        candidates["shipment_id"] == selected
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", int(row["risk_score"]))
    c2.metric("Delay hours", round(row["delay_hours"], 1))
    c3.metric("Priority", row["priority"])

    options = build_recovery_options(row)

    st.subheader("Available recovery options")
    st.dataframe(
        options,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Costs and time savings are illustrative planning assumptions. "
        "Validate with actual carrier, terminal and transport quotations."
    )

    chosen = st.selectbox(
        "Record a proposed action",
        options["action"].tolist()
    )

    if st.button("Record proposed recovery action"):
        selected_option = options[
            options["action"] == chosen
        ].iloc[0]

        save_decision(
            selected,
            "Recovery action",
            chosen,
            selected_option.to_dict()
        )

        st.success("Proposed action recorded in the decision audit trail.")

    if st.button("Ask AI to compare recovery options"):
        context = {
            "shipment": {
                "shipment_id": selected,
                "priority": row["priority"],
                "delay_hours": row["delay_hours"],
                "risk_score": row["risk_score"],
                "origin": row["origin_port"],
                "destination": row["destination_port"]
            },
            "options": options.to_dict(orient="records"),
            "instruction": (
                "Compare these options. Explain cost, time, feasibility "
                "and uncertainty. Do not claim estimates are live quotes."
            )
        }

        with st.spinner("Comparing recovery options..."):
            answer, error = call_bharatcode(
                json.dumps(context, default=str)
            )

        if error:
            st.warning(error)
            st.markdown(build_local_recovery_comparison(row, options))
            save_decision(
                selected,
                "Local recovery comparison",
                "Compare recovery options",
                {"provider_status": error, "options": options.to_dict(orient="records")}
            )
        else:
            st.markdown("### BharatCode AI recovery analysis")
            st.markdown(answer)
            save_decision(
                selected,
                "AI recovery comparison",
                "Compare recovery options",
                {"response": answer, "provider": "BharatCode", "model": MODEL}
            )

elif page == "Decision Reports":
    page_header(
        "MODULE 06",
        "Decision reports.",
        "Review recorded operational decisions and export a structured snapshot of current shipment exposure."
    )

    decisions = read_decisions()

    st.subheader("Decision audit trail")

    if decisions.empty:
        st.info(
            "No decisions have been recorded yet. Run a simulation "
            "or record a recovery action to populate the audit trail."
        )
    else:
        st.dataframe(
            decisions,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download decision audit trail",
            make_csv_download(decisions),
            file_name="portiq_decision_audit.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Current operational snapshot")

    report = risk_df.copy()

    st.metric(
        "Shipments in report",
        len(report)
    )

    st.metric(
        "High and critical risk",
        int(report["risk_level"].isin(["High", "Critical"]).sum())
    )

    st.download_button(
        "Export operational report",
        make_csv_download(report),
        file_name="portiq_operational_report.csv",
        mime="text/csv"
    )

    report_text = f"""
PORTIQ OPERATIONAL REPORT
Generated: {datetime.now().isoformat(timespec="seconds")}

Dataset: {st.session_state.source_name}
Total shipments: {len(report)}
Delayed shipments: {(report["operational_status"] == "Delayed").sum()}
High and critical risk: {report["risk_level"].isin(["High", "Critical"]).sum()}
Average risk score: {report["risk_score"].mean():.2f}

This report is based on the currently loaded dataset.
Demonstration data, where used, is simulated.
Risk scores are rule-based indicators, not validated probabilities.
"""

    st.download_button(
        "Download text summary",
        report_text,
        file_name="portiq_summary.txt",
        mime="text/plain"
    )

elif page == "Data & Model Health":
    page_header(
        "SYSTEM INTELLIGENCE",
        "Data & model health.",
        "Inspect data quality, anomaly detection and AI connectivity."
    )

    st.subheader("Data quality")

    quality = pd.DataFrame({
        "Column": raw_df.columns,
        "Missing values": [
            int(raw_df[c].isna().sum())
            for c in raw_df.columns
        ],
        "Missing percentage": [
            round(raw_df[c].isna().mean() * 100, 2)
            for c in raw_df.columns
        ],
        "Data type": [
            str(raw_df[c].dtype)
            for c in raw_df.columns
        ]
    })

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Unique shipments",
        raw_df["shipment_id"].nunique()
    )

    st.metric(
        "Duplicate shipment IDs",
        int(raw_df["shipment_id"].duplicated().sum())
    )

    st.divider()
    st.subheader("Machine learning anomaly detection")

    anomaly_count = int(
        (risk_df["anomaly_flag"] == "Unusual pattern").sum()
    )

    st.metric(
        "Unusual patterns detected",
        anomaly_count
    )

    st.dataframe(
        risk_df[[
            "shipment_id", "delay_hours",
            "risk_score", "port_congestion_index",
            "transport_capacity", "anomaly_flag"
        ]].sort_values(
            "risk_score", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Isolation Forest is fitted to the currently loaded dataset. "
        "An anomaly is an unusual pattern, not proof of a disruption."
    )

    st.divider()
    st.subheader("BharatCode API connectivity")

    if not API_KEY:
        st.warning(
            "API key is missing. Configure BHARATCODE_API_KEY in .env."
        )
    else:
        st.success("API key is present in the local environment.")

        if st.button("Test BharatCode model access"):
            try:
                response = requests.get(
                    "https://bharatcode.ai/api/model/v1/models",
                    headers={
                        "Authorization": f"Bearer {API_KEY}"
                    },
                    timeout=25
                )

                if response.status_code == 200:
                    st.success("Model catalogue retrieved successfully.")
                    st.json(response.json())
                elif response.status_code in [401, 403]:
                    st.error(
                        f"Access is not authorised yet (HTTP {response.status_code})."
                    )
                else:
                    st.error(
                        f"Model check returned HTTP {response.status_code}."
                    )
            except requests.exceptions.RequestException as exc:
                st.error(f"Connection error: {exc}")

    st.divider()
    st.subheader("Application information")

    st.write("Application: PortIQ")
    st.write("Model: Isolation Forest")
    st.write("AI provider: BharatCode")
    st.write("AI model: " + MODEL)
    st.write("Storage: SQLite")
    st.write("Current records: " + str(len(raw_df)))

st.sidebar.divider()
st.sidebar.caption("PortIQ | Build League 2026")