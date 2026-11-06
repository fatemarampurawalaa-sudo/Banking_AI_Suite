# 🏦 Banking AI Suite

🌐 **Live app:** https://bankingaisuite-eqteknkbkx7m8grxexqtgk.streamlit.app

One app, four AI tools for retail banking: live fraud monitoring, statement insights, loan risk and fraud checks.

> ⚠️ **Demo project only.** All customers, transactions and loans are made-up data. Not financial advice.

## The 4 tools
| Tab | What it does | Tech |
|---|---|---|
| 🗺️ Live Fraud Monitor | Simulates a real-time stream of card payments. Every payment is scored by the fraud model and shown live on a world map: 🟢 approve · 🟡 OTP · 🔴 block, with live counters and alerts that explain why | XGBoost, Plotly maps, Streamlit fragments (auto-refresh) |
| 💳 Statement Analyzer | Sorts transactions into categories in one batched AI call, shows spending charts and AI money-saving tips | Gemini structured output, pandas |
| 🏦 Loan Risk Predictor | Predicts loan default risk, explains the key drivers, checks the UAE 50% salary rule | XGBoost, feature importance |
| 🚨 Fraud Check | Test any single card transaction and see its fraud score, decision and reasons | XGBoost on imbalanced data (2% fraud) |

## Key ML lessons
- **Fraud:** accuracy is misleading (a "never fraud" model scores 98%), so I focused on **recall** and used class weights
- **Realistic training data:** normal customers also travel abroad, so the model learns that "far from home" alone is not fraud; night-time, large amounts, new devices and bursts of payments matter more
- **Loans:** risk score + feature importance, so every decision can be **explained**; a loan officer makes the final call

## Engineering
- Models train once and are cached, so the app starts fast
- The live monitor updates only its own section every 1.5 seconds (Streamlit fragment), not the whole page
- One batched AI call for categorisation (faster and cheaper), with retry logic and a rule-based backup if the AI service is busy

## Tech stack
Python · pandas · XGBoost · Plotly · Gemini API · Streamlit
