# 📈 AI Financial Advisor

An AI-powered financial analysis assistant built with **Python, Flask, yFinance, and LLM technology**. The application combines real-time market data with AI-generated insights to help users understand stocks, market trends, and financial metrics through a simple web interface.

> ⚠️ **Disclaimer:** This application is designed for educational and informational purposes only. It does not provide professional financial, investment, or trading advice.
Live Link: https://ai-financial-advisor-3iiw.onrender.com/
---

## 🚀 Features

- 📊 **Real-Time Stock Data**
  - Fetches current and historical market data using `yFinance`
  - Supports analysis of popular stocks and financial assets

- 🤖 **AI-Powered Financial Analysis**
  - Uses an LLM to interpret financial data
  - Generates natural-language explanations and insights

- 📈 **Market Trend Analysis**
  - Analyzes historical price movements
  - Provides basic trend information based on available market data

- 💬 **Natural-Language Interaction**
  - Ask financial questions in plain English
  - Receive AI-generated responses based on available financial context

- 🧮 **Financial Calculations**
  - Average prices
  - Latest prices
  - Price changes
  - Basic market statistics

- 🔄 **Fallback Analysis**
  - Provides rule-based analysis when the AI service is unavailable

- 🌐 **Web-Based Interface**
  - Clean and responsive interface
  - Built using HTML, CSS, and Flask

---

## 🧠 How It Works

The application follows a simple data-to-insight pipeline:

```text
                 User
                   │
                   ▼
          ┌─────────────────┐
          │   Flask Web App │
          └────────┬────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
   ┌──────────────┐    ┌──────────────┐
   │   yFinance   │    │    LLM API   │
   │ Market Data  │    │ AI Analysis  │
   └──────┬───────┘    └──────┬───────┘
          │                   │
          └─────────┬─────────┘
                    ▼
          ┌──────────────────┐
          │ Financial Insight│
          └──────────────────┘
                    │
                    ▼
                  User
