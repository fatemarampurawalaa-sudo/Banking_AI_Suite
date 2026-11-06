import math
import random
import time
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pydantic import BaseModel
from xgboost import XGBClassifier
from google import genai
from google.genai import types

st.set_page_config(page_title="Banking AI Suite", page_icon="🏦", layout="wide")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
CHAT_MODEL = "gemini-flash-lite-latest"


def call_ai(contents, config=None, tries=3):
    for attempt in range(tries):
        try:
            return client.models.generate_content(model=CHAT_MODEL, contents=contents, config=config)
        except Exception:
            time.sleep(5)
    return None


# =============== PART 1: STATEMENT DATA ===============
STATEMENT = pd.DataFrame({
    "date": pd.date_range("2026-10-01", periods=30, freq="D").strftime("%Y-%m-%d"),
    "description": [
        "SALARY CREDIT - ABC TRADING LLC", "CARREFOUR MALL OF EMIRATES", "DEWA BILL PAYMENT", "NETFLIX.COM",
        "UBER *TRIP DUBAI", "TALABAT ORDER", "ENOC FUEL STATION", "ZARA DUBAI MALL",
        "RENT TRANSFER - LANDLORD", "STARBUCKS JBR", "ETISALAT POSTPAID", "NOON.COM ORDER",
        "CAREEM RIDE", "LULU HYPERMARKET", "AMAZON.AE", "GYM MEMBERSHIP - FITNESS FIRST",
        "DELIVEROO", "ATM WITHDRAWAL", "EMIRATES AIRLINE TICKET", "ASTER PHARMACY",
        "SPINNEYS", "VOX CINEMAS", "SALIK TOLL", "KFC", "IKEA DUBAI",
        "SCHOOL FEES TRANSFER", "SPOTIFY", "DU MOBILE RECHARGE", "TALABAT ORDER", "CARREFOUR CITY CENTRE",
    ],
    "amount": [
        25000, -420, -650, -45, -38, -85, -180, -560, -7500, -28,
        -250, -310, -42, -390, -199, -300, -72, -1000, -2100, -95,
        -270, -120, -16, -48, -880, -3500, -22, -100, -64, -230,
    ],
})

# Backup categories, used only if the AI service is busy
BACKUP_CATEGORIES = {
    "SALARY CREDIT - ABC TRADING LLC": "Income", "CARREFOUR MALL OF EMIRATES": "Groceries",
    "DEWA BILL PAYMENT": "Bills & Utilities", "NETFLIX.COM": "Entertainment",
    "UBER *TRIP DUBAI": "Transport", "TALABAT ORDER": "Food & Dining", "ENOC FUEL STATION": "Transport",
    "ZARA DUBAI MALL": "Shopping", "RENT TRANSFER - LANDLORD": "Rent & Housing", "STARBUCKS JBR": "Food & Dining",
    "ETISALAT POSTPAID": "Bills & Utilities", "NOON.COM ORDER": "Shopping", "CAREEM RIDE": "Transport",
    "LULU HYPERMARKET": "Groceries", "AMAZON.AE": "Shopping", "GYM MEMBERSHIP - FITNESS FIRST": "Health & Fitness",
    "DELIVEROO": "Food & Dining", "ATM WITHDRAWAL": "Cash", "EMIRATES AIRLINE TICKET": "Travel",
    "ASTER PHARMACY": "Health & Fitness", "SPINNEYS": "Groceries", "VOX CINEMAS": "Entertainment",
    "SALIK TOLL": "Transport", "KFC": "Food & Dining", "IKEA DUBAI": "Shopping",
    "SCHOOL FEES TRANSFER": "Education", "SPOTIFY": "Entertainment", "DU MOBILE RECHARGE": "Bills & Utilities",
    "CARREFOUR CITY CENTRE": "Groceries",
}

Category = Literal["Income", "Groceries", "Food & Dining", "Transport", "Bills & Utilities",
                   "Shopping", "Rent & Housing", "Entertainment", "Health & Fitness",
                   "Travel", "Education", "Cash"]


class Tagged(BaseModel):
    description: str
    category: Category


@st.cache_data(show_spinner=False)
def categorize(descriptions: tuple):
    response = call_ai(
        "Categorize each UAE bank transaction:\n" + "\n".join(descriptions),
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=list[Tagged]),
        tries=1,
    )
    if response is not None and response.parsed:
        mapping = {t.description: t.category for t in response.parsed}
        if all(d in mapping for d in descriptions):
            return mapping, "Gemini AI"
    return BACKUP_CATEGORIES, "backup rules (AI busy)"


# =============== PART 2: LOAN MODEL ===============
LOAN_FEATURES = ["age", "monthly_income", "loan_amount", "credit_score",
                 "years_employed", "existing_debt", "late_payments"]


@st.cache_resource
def train_loan_model():
    r = np.random.RandomState(42)
    n = 1000
    df = pd.DataFrame({
        "age": r.randint(21, 65, n),
        "monthly_income": r.randint(5000, 60000, n),
        "loan_amount": r.randint(20000, 500000, n),
        "credit_score": r.randint(300, 850, n),
        "years_employed": r.randint(0, 30, n),
        "existing_debt": r.randint(0, 200000, n),
        "late_payments": r.poisson(1, n),
    })
    risk = (-0.01 * (df["credit_score"] - 600) + 0.8 * df["late_payments"]
            + 3 * df["loan_amount"] / (df["monthly_income"] * 12)
            + 2 * df["existing_debt"] / (df["monthly_income"] * 12)
            - 0.05 * df["years_employed"] - 2)
    df["default"] = (r.rand(n) < 1 / (1 + np.exp(-risk))).astype(int)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1)
    model.fit(df[LOAN_FEATURES], df["default"])
    return model


def loan_risk(profile: dict):
    prob = float(loan_model.predict_proba(pd.DataFrame([profile])[LOAN_FEATURES])[0][1])
    if prob > 0.6:
        label = "🔴 High risk – review carefully"
    elif prob > 0.3:
        label = "🟡 Medium risk"
    else:
        label = "🟢 Low risk"
    return prob, label


# =============== PART 3: FRAUD MODEL ===============
FRAUD_FEATURES = ["amount", "hour", "is_foreign", "new_device", "txns_last_hour", "distance_km"]


@st.cache_resource
def train_fraud_model():
    r = np.random.RandomState(7)
    nn, nf = 4900, 100
    travelling = r.choice([0, 1], nn, p=[0.88, 0.12])
    normal = pd.DataFrame({
        "amount": r.gamma(2, 150, nn).round(0), "hour": r.choice(range(7, 24), nn),
        "is_foreign": travelling, "new_device": r.choice([0, 1], nn, p=[0.97, 0.03]),
        "txns_last_hour": r.poisson(1, nn),
        "distance_km": np.where(travelling == 1, r.exponential(2500, nn), r.exponential(10, nn)).round(1),
        "fraud": 0,
    })
    fraud = pd.DataFrame({
        "amount": r.gamma(3, 900, nf).round(0), "hour": r.choice(range(0, 6), nf),
        "is_foreign": r.choice([0, 1], nf, p=[0.3, 0.7]), "new_device": r.choice([0, 1], nf, p=[0.4, 0.6]),
        "txns_last_hour": r.poisson(5, nf), "distance_km": r.exponential(800, nf).round(1), "fraud": 1,
    })
    # "Grey zone": daytime, abroad, fairly large amounts, sometimes a new device.
    # In real life most of these are genuine, but some are fraud, so the model learns to be unsure (→ OTP).
    ng = 500
    grey = pd.DataFrame({
        "amount": r.gamma(2, 900, ng).round(0), "hour": r.choice(range(6, 24), ng),
        "is_foreign": 1, "new_device": r.choice([0, 1], ng, p=[0.7, 0.3]),
        "txns_last_hour": r.poisson(2, ng), "distance_km": r.exponential(3000, ng).round(1),
        "fraud": r.choice([0, 1], ng, p=[0.8, 0.2]),
    })
    df = pd.concat([normal, fraud, grey])
    model = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, min_child_weight=5,
                          scale_pos_weight=5)
    model.fit(df[FRAUD_FEATURES], df["fraud"])
    return model


def check_transaction(t: dict):
    score = float(fraud_model.predict_proba(pd.DataFrame([t])[FRAUD_FEATURES])[0][1])
    reasons = []
    if t["amount"] > 2000: reasons.append("large amount")
    if t["hour"] < 6: reasons.append("unusual time (night)")
    if t["is_foreign"]: reasons.append("foreign transaction")
    if t["new_device"]: reasons.append("new device")
    if t["txns_last_hour"] >= 4: reasons.append("many transactions in 1 hour")
    if t["distance_km"] > 200: reasons.append("far from home")
    if score > 0.8:
        action = "🔴 BLOCK and call the customer"
    elif score > 0.4:
        action = "🟡 Send OTP to confirm"
    else:
        action = "🟢 Approve"
    return score, action, reasons


# =============== LOAD EVERYTHING (fast) ===============
loan_model = train_loan_model()
fraud_model = train_fraud_model()

with st.spinner("Categorising transactions..."):
    mapping, category_source = categorize(tuple(STATEMENT["description"]))
statement = STATEMENT.copy()
statement["category"] = statement["description"].map(mapping)
income = float(statement[statement["amount"] > 0]["amount"].sum())
spent = float(-statement[statement["amount"] < 0]["amount"].sum())
by_cat = (-statement[statement["amount"] < 0].groupby("category")["amount"].sum()).sort_values(ascending=False)


# =============== PAGE ===============
st.title("🏦 Banking AI Suite")
st.caption("Four AI tools for retail banking: live fraud monitoring, statement insights, loan risk and fraud checks.")
st.warning("⚠️ Demo only: all customers and transactions are made-up data. Not financial advice.")
st.info("👋 **How to use:** start with **🗺️ Live Fraud Monitor**: switch on **Live monitoring** and watch the AI "
        "score every card payment on the world map. Then explore the other tabs: 💳 spending · 🏦 loan risk · 🚨 test one transaction.")

tab_live, tab1, tab2, tab3 = st.tabs(["🗺️ Live Fraud Monitor", "💳 Statement Analyzer", "🏦 Loan Risk", "🚨 Fraud Check"])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("October 2026 statement · Sara Ali")
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Income", f"AED {income:,.0f}")
    c2.metric("💸 Spent", f"AED {spent:,.0f}")
    c3.metric("🏦 Saved", f"AED {income - spent:,.0f}", f"{(income - spent) / income:.0%} of income")
    st.write("**Spending by category (AED)**")
    st.bar_chart(by_cat, horizontal=True)
    st.caption(f"Categories by: {category_source}")
    st.dataframe(statement, hide_index=True)
    if st.button("💡 Get AI money-saving tips"):
        prompt = (f"Monthly income: AED {income:,.0f}\nTotal spent: AED {spent:,.0f}\n"
                  f"Spending by category:\n{by_cat.to_string()}\n\n"
                  "Give 3 short, practical money-saving tips for this customer, based on the numbers. "
                  "Be friendly. No investment advice.")
        with st.spinner("Thinking..."):
            tips = call_ai(prompt, types.GenerateContentConfig(max_output_tokens=250))
        st.info(tips.text if tips else "The AI service is busy. Please try again in a minute.")

# ---------- Tab 2 ----------
with tab2:
    st.subheader("Personal loan risk check")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 21, 65, 34)
        monthly_income = st.number_input("Monthly income (AED)", 5000, 60000, 25000, step=1000)
        loan_amount = st.number_input("Loan amount (AED)", 20000, 500000, 300000, step=10000)
        credit_score = st.slider("Credit score", 300, 850, 710)
    with c2:
        years_employed = st.slider("Years employed", 0, 30, 8)
        existing_debt = st.number_input("Existing debt (AED)", 0, 200000, 40000, step=5000)
        late_payments = st.slider("Late payments (last 2 years)", 0, 10, 0)

    prob, label = loan_risk({"age": age, "monthly_income": monthly_income, "loan_amount": loan_amount,
                             "credit_score": credit_score, "years_employed": years_employed,
                             "existing_debt": existing_debt, "late_payments": late_payments})
    monthly_payment = loan_amount / 48
    r1, r2 = st.columns(2)
    r1.metric("Default probability", f"{prob:.0%}", label, delta_color="off")
    r2.metric("Est. monthly payment (4 yrs)", f"AED {monthly_payment:,.0f}",
              "Within 50% salary rule ✅" if monthly_payment <= 0.5 * monthly_income else "Above 50% salary rule ❌",
              delta_color="off")
    st.write("**What drives the model's decision?**")
    st.bar_chart(pd.Series(loan_model.feature_importances_, index=LOAN_FEATURES).sort_values(ascending=False),
                 horizontal=True)
    st.caption("Indicative only. A loan officer makes the final decision.")

# ---------- Tab 3 ----------
with tab3:
    st.subheader("Card transaction fraud check")
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount (AED)", 1, 50000, 4800)
        hour = st.slider("Hour of day (0–23)", 0, 23, 3)
        txns_last_hour = st.slider("Transactions in the last hour", 0, 15, 6)
    with c2:
        distance_km = st.number_input("Distance from home (km)", 0, 15000, 2500)
        is_foreign = st.checkbox("Foreign transaction", value=True)
        new_device = st.checkbox("New device", value=True)

    score, action, reasons = check_transaction({
        "amount": amount, "hour": hour, "is_foreign": int(is_foreign), "new_device": int(new_device),
        "txns_last_hour": txns_last_hour, "distance_km": distance_km})
    st.metric("Fraud score", f"{score:.0%}")
    st.subheader(action)
    st.write("**Reasons:** " + (", ".join(reasons) if reasons else "nothing unusual"))
    st.caption("Model trained on imbalanced data (2% fraud) using class weights; optimised for recall.")

# ---------- Live Fraud Monitor ----------
HOME = (25.20, 55.27)  # customers' home city: Dubai
CITIES = {
    "Dubai": (25.20, 55.27, "UAE"), "Abu Dhabi": (24.45, 54.38, "UAE"), "Sharjah": (25.35, 55.42, "UAE"),
    "Al Ain": (24.21, 55.74, "UAE"), "Riyadh": (24.71, 46.68, "Saudi Arabia"), "Muscat": (23.59, 58.41, "Oman"),
    "Doha": (25.29, 51.53, "Qatar"), "Mumbai": (19.08, 72.88, "India"), "London": (51.51, -0.13, "UK"),
    "Paris": (48.86, 2.35, "France"), "Cairo": (30.04, 31.24, "Egypt"), "Karachi": (24.86, 67.01, "Pakistan"),
    "Singapore": (1.35, 103.82, "Singapore"), "Istanbul": (41.01, 28.98, "Turkey"), "Lagos": (6.52, 3.38, "Nigeria"),
    "Moscow": (55.76, 37.62, "Russia"), "New York": (40.71, -74.01, "USA"), "Sao Paulo": (-23.55, -46.63, "Brazil"),
    "Sydney": (-33.87, 151.21, "Australia"), "Bangkok": (13.76, 100.50, "Thailand"),
}
LOCAL = ["Dubai"] * 8 + ["Abu Dhabi"] * 4 + ["Sharjah"] * 3 + ["Al Ain"]
TRAVEL = ["Riyadh", "Muscat", "Doha", "Mumbai", "London", "Paris", "Cairo", "Karachi", "Singapore", "Istanbul"]
RISKY = ["Lagos", "Moscow", "New York", "Sao Paulo", "Sydney", "Bangkok", "London", "Istanbul"]
STATUS_COLORS = {"Approved": "#2ecc71", "OTP": "#f5b041", "Blocked": "#eb4034"}
SPEEDS = {"Slow": 1, "Normal": 3, "Fast": 6}


def km_from_home(lat, lon):
    lat1, lon1, lat2, lon2 = map(math.radians, [HOME[0], HOME[1], lat, lon])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def make_transactions(n):
    """Simulate n new card payments and score each one with the fraud model."""
    rows = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.06:      # suspicious pattern
            city = random.choice(RISKY)
            amount, hour = round(random.gammavariate(3, 900)), random.randint(0, 5)
            new_device, txns = int(random.random() < 0.6), random.randint(3, 8)
        elif roll < 0.13:    # grey zone: abroad, daytime, bigger amount (model is unsure → OTP)
            city = random.choice(TRAVEL + RISKY)
            amount, hour = round(random.gammavariate(2, 900)), random.randint(9, 20)
            new_device, txns = int(random.random() < 0.3), random.randint(1, 3)
        elif roll < 0.25:    # customer travelling abroad (normal)
            city = random.choice(TRAVEL)
            amount, hour = round(random.gammavariate(2, 250)), random.randint(8, 22)
            new_device, txns = int(random.random() < 0.05), random.randint(0, 2)
        else:                # everyday local payment
            city = random.choice(LOCAL)
            amount, hour = round(random.gammavariate(2, 150)), random.randint(7, 23)
            new_device, txns = int(random.random() < 0.03), random.randint(0, 2)
        lat, lon, country = CITIES[city]
        rows.append({
            "city": city, "country": country,
            "lat": lat + random.uniform(-0.7, 0.7), "lon": lon + random.uniform(-0.7, 0.7),
            "amount": max(amount, 5), "hour": hour, "is_foreign": int(country != "UAE"),
            "new_device": new_device, "txns_last_hour": txns,
            "distance_km": round(km_from_home(lat, lon), 1),
            "time": f"{hour:02d}:{random.randint(0, 59):02d}",
        })
    df = pd.DataFrame(rows)
    df["score"] = fraud_model.predict_proba(df[FRAUD_FEATURES])[:, 1]
    df["status"] = np.where(df["score"] > 0.8, "Blocked", np.where(df["score"] > 0.4, "OTP", "Approved"))
    return df


def reasons_for(t):
    reasons = []
    if t["amount"] > 2000: reasons.append("large amount")
    if t["hour"] < 6: reasons.append("night-time")
    if t["new_device"]: reasons.append("new device")
    if t["txns_last_hour"] >= 4: reasons.append(f"{t['txns_last_hour']} payments in 1 hr")
    if t["distance_km"] > 200: reasons.append(f"{t['distance_km']:,.0f} km from home")
    return " · ".join(reasons) if reasons else "unusual pattern"


def add_to_feed(df):
    s = st.session_state.live_stats
    s["checked"] += len(df)
    for status in STATUS_COLORS:
        s[status] += int((df["status"] == status).sum())
    s["saved"] += float(df.loc[df["status"] == "Blocked", "amount"].sum())
    st.session_state.live_feed = pd.concat([st.session_state.live_feed, df], ignore_index=True).tail(400)


def reset_feed():
    st.session_state.live_feed = pd.DataFrame()
    st.session_state.live_stats = {"checked": 0, "Approved": 0, "OTP": 0, "Blocked": 0, "saved": 0.0}
    add_to_feed(make_transactions(150))


if "live_feed" not in st.session_state:
    reset_feed()


def stat_card(label, value, color):
    return (f"<div style='background:#111a33;border-radius:12px;padding:10px 16px'>"
            f"<div style='color:#8c9bb9;font-size:0.85rem'>{label}</div>"
            f"<div style='color:{color};font-size:1.7rem;font-weight:700'>{value}</div></div>")


def live_view():
    if st.session_state.get("live_on"):
        add_to_feed(make_transactions(SPEEDS[st.session_state.get("live_speed", "Normal")]))
    feed, s = st.session_state.live_feed, st.session_state.live_stats

    cards = [("Payments checked", f"{s['checked']:,}", "#e1e8f5"), ("Approved", f"{s['Approved']:,}", "#2ecc71"),
             ("OTP sent", f"{s['OTP']:,}", "#f5b041"), ("Blocked", f"{s['Blocked']:,}", "#eb4034"),
             ("Fraud stopped", f"AED {s['saved']:,.0f}", "#2ecc71")]
    for col, (label, value, color) in zip(st.columns(5), cards):
        col.markdown(stat_card(label, value, color), unsafe_allow_html=True)

    left, right = st.columns([2.2, 1])
    with left:
        fig = go.Figure()
        for status, size in [("Approved", 6), ("OTP", 11), ("Blocked", 15)]:
            d = feed[feed["status"] == status]
            fig.add_trace(go.Scattergeo(
                lat=d["lat"], lon=d["lon"], mode="markers", name=status,
                marker=dict(size=size, color=STATUS_COLORS[status], opacity=0.85,
                            line=dict(width=1 if status == "Blocked" else 0, color="white")),
                text=[f"{r.city} · AED {r.amount:,} · {r.time} · {r.status} ({r.score:.0%})" for r in d.itertuples()],
                hoverinfo="text"))
        fig.update_geos(projection_type="natural earth", showland=True, landcolor="#1c2a4a",
                        showocean=True, oceancolor="#0b1224", showcountries=True, countrycolor="#33476e",
                        showcoastlines=True, coastlinecolor="#33476e", showframe=False, bgcolor="#0b1224",
                        lataxis_range=[-45, 72])
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0b1224",
                          legend=dict(orientation="h", x=0.01, y=0.02, font=dict(color="#e1e8f5"),
                                      bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, key="live_map", config={"displayModeBar": False})
    with right:
        st.markdown("<div style='font-weight:700;font-size:1.1rem;margin-bottom:6px'>🚨 Latest alerts</div>",
                    unsafe_allow_html=True)
        alerts = feed[feed["status"] != "Approved"].iloc[::-1].head(5)
        for _, a in alerts.iterrows():
            color = STATUS_COLORS[a["status"]]
            label = "BLOCKED" if a["status"] == "Blocked" else "OTP SENT"
            st.markdown(
                f"<div style='border-left:6px solid {color};background:#111a33;color:#e1e8f5;padding:8px 12px;"
                f"border-radius:8px;margin-bottom:8px'>"
                f"<span style='color:{color};font-weight:700;font-size:0.8rem'>{label}</span>"
                f"<span style='color:#8c9bb9;font-size:0.8rem'> · {a['time']} · score {a['score']:.0%}</span><br>"
                f"<b>AED {a['amount']:,} · {a['city']}</b><br>"
                f"<span style='color:#8c9bb9;font-size:0.8rem'>{reasons_for(a)}</span></div>",
                unsafe_allow_html=True)


with tab_live:
    st.subheader("Live card-payment monitoring")
    st.caption("Every payment is scored by the XGBoost fraud model in real time → 🟢 approve · 🟡 send OTP · 🔴 block. "
               "Simulated transaction stream for customers based in Dubai.")
    c1, c2, c3 = st.columns([1.2, 1.5, 1])
    c1.toggle("▶️ Live monitoring", key="live_on")
    c2.select_slider("Speed", options=list(SPEEDS), value="Normal", key="live_speed")
    if c3.button("🔄 Reset"):
        reset_feed()
    st.fragment(run_every=1.5 if st.session_state.get("live_on") else None)(live_view)()
