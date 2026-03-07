from flask import Flask, render_template, request, jsonify
import yfinance as yf
app = Flask(__name__)
def ai_advisor(stock, avg_price, latest_price):
    if latest_price > avg_price:
        advice = f"""
The stock {stock} is currently trading ABOVE its yearly average.

This suggests positive momentum in the market.

Risk Level: Medium

Suggestion:
Consider HOLDING the stock or buying on dips if you are a long-term investor.
"""
    else:
        advice = f"""
The stock {stock} is currently trading BELOW its yearly average.

This may indicate bearish sentiment.

Risk Level: Medium to High

Suggestion:
Be cautious and wait for stronger signals before investing.
"""

    return advice


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json()
        stock = data["stock"].upper()

        df = yf.download(stock, period="1y", progress=False)

        if df.empty:
            return jsonify({"error": "Invalid stock symbol"})

        close = df["Close"].squeeze()

        avg_price = float(close.mean())
        latest_price = float(close.iloc[-1])

        trend = "Uptrend 📈" if latest_price > avg_price else "Downtrend 📉"

        price_change = latest_price - avg_price

        advice = ai_advisor(stock, avg_price, latest_price)

        return jsonify({
            "stock": stock,
            "avg": round(avg_price,2),
            "latest": round(latest_price,2),
            "trend": trend,
            "change": round(price_change,2),
            "advice": advice,
            "chart": close.tolist(),
            "dates": df.index.strftime("%Y-%m-%d").tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)