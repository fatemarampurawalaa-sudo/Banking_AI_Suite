# Banking_AI_Suite
# 🏦 Banking AI Suite

One app, four AI tools for retail banking: built with machine learning, LLMs and an AI agent.

> ⚠️ **Demo project only.** All customers, transactions and loans are made-up data. Not financial advice.

## The 4 tools
| Tab | What it does | Tech |
|---|---|---|
| 💳 Statement Analyzer | Sorts transactions into categories, shows spending charts and AI money-saving tips | Gemini structured output, pandas |
| 🏦 Loan Risk Predictor | Predicts loan default risk, explains the key drivers, checks the UAE 50% salary rule | XGBoost, feature importance |
| 🚨 Fraud Detector | Scores a card transaction, gives reasons, and decides: approve / OTP / block | XGBoost on imbalanced data (2% fraud) |
| 🤖 AI Support Agent | Answers customer questions by choosing the right tool: balance, fees, spending, loans, fraud | Gemini tool calling, SQLite, policy RAG |

## Security & compliance
- The agent only sees the **logged-in customer's** data; it can't pick another customer ID
- **Read-only** database access and parameterised SQL (no SQL injection)
- Account numbers **masked**; IBAN/card numbers hidden in every answer
- Cannot move money, change limits or unblock cards
- **No investment advice**; loan answers are indicative, and a loan officer decides
- Policy text treated as data, not instructions (prompt-injection protection)
- Retry logic for temporary API errors (503)

## Key ML lessons
- **Fraud:** accuracy is misleading (a "never fraud" model scores 98%), so I focused on **recall** and used class weights
- **Loans:** risk score + feature importance, so every decision can be **explained**

## Tech stack
Python · pandas · XGBoost · Gemini API · Function calling · SQLite · Streamlit
