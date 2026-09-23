from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import json
import os
import urllib.request
import urllib.error

app = Flask(__name__)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"


def groq_chat(system_prompt, user_prompt, max_tokens=1024):
    """Call Groq API (OpenAI-compatible). Falls back to rule-based if no key."""
    if not GROQ_API_KEY:
        return None

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + GROQ_API_KEY,
            "User-Agent": "QuantView/1.0"
            
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        print("Groq error:", e)
        print("Groq response:", e.read().decode("utf-8", errors="replace"))
        return None
    except Exception as e:
        print("Groq error:", e)
        return None


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    rs_vals  = []
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rs_vals.append(100 - (100 / (1 + rs)))
    return float(rs_vals[-1]) if rs_vals else 50.0


def compute_macd(prices):
    prices = np.array(prices)
    def ema(data, span):
        k = 2 / (span + 1)
        out = [data[0]]
        for p in data[1:]:
            out.append(p * k + out[-1] * (1 - k))
        return np.array(out)
    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    macd_line = ema12 - ema26
    signal    = ema(macd_line, 9)
    histogram = macd_line - signal
    return {
        "macd":      round(float(macd_line[-1]), 4),
        "signal":    round(float(signal[-1]),    4),
        "histogram": round(float(histogram[-1]), 4),
        "crossover": "bullish" if macd_line[-1] > signal[-1] else "bearish"
    }


def compute_bollinger(prices, period=20):
    arr = np.array(prices[-period:])
    mid = float(np.mean(arr))
    std = float(np.std(arr))
    upper = mid + 2 * std
    lower = mid - 2 * std
    pct_b = (float(arr[-1]) - lower) / (upper - lower) if (upper - lower) else 0.5
    return {"upper": round(upper, 2), "middle": round(mid, 2),
            "lower": round(lower, 2), "pct_b": round(pct_b * 100, 1)}


def compute_atr(df, period=14):
    try:
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        close = df["Close"].squeeze()
        prev  = close.shift(1)
        tr    = np.maximum(high - low, np.maximum(abs(high - prev), abs(low - prev)))
        return round(float(tr.rolling(period).mean().iloc[-1]), 2)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# RULE-BASED FALLBACK ANALYSIS
# (used when no API key is set)
# ─────────────────────────────────────────────

def rule_based_analysis(stock, metrics, fundamentals):
    rsi  = metrics["rsi"]
    macd = metrics["macd"]
    boll = metrics["bollinger"]
    chg  = metrics["change_pct"]
    dist = metrics["dist_from_high"]

    # Score system
    score = 0
    if rsi < 30:  score += 2
    elif rsi < 45: score += 1
    elif rsi > 70: score -= 2
    elif rsi > 60: score -= 1

    if macd["crossover"] == "bullish": score += 2
    else: score -= 1

    if boll["pct_b"] < 20:  score += 1
    elif boll["pct_b"] > 80: score -= 1

    if chg > 5:  score += 1
    elif chg < -10: score -= 1

    if score >= 4:   signal, conf, risk = "STRONG BUY",  "HIGH",   "MEDIUM"
    elif score >= 2: signal, conf, risk = "BUY",         "MEDIUM", "MEDIUM"
    elif score >= 0: signal, conf, risk = "HOLD",        "MEDIUM", "MEDIUM"
    elif score >= -2:signal, conf, risk = "SELL",        "MEDIUM", "HIGH"
    else:            signal, conf, risk = "STRONG SELL", "HIGH",   "VERY HIGH"

    rsi_note = ("oversold — potential bounce" if rsi < 30
                else "overbought — caution" if rsi > 70
                else "neutral momentum")

    macd_note = ("bullish crossover — upward momentum" if macd["crossover"] == "bullish"
                 else "bearish crossover — downward momentum")

    pe  = fundamentals.get("pe_fwd", "N/A")
    beta = fundamentals.get("beta", "N/A")

    bear = round(metrics["latest"] * 0.85, 2)
    base = round(metrics["latest"] * 1.08, 2)
    bull = round(metrics["latest"] * 1.20, 2)

    if fundamentals.get("analyst_target") and fundamentals["analyst_target"] != "N/A":
        base = fundamentals["analyst_target"]
        bull = round(float(base) * 1.10, 2)
        bear = round(metrics["latest"] * 0.88, 2)

    return {
        "summary": (
            f"{stock} is showing {macd_note}. RSI at {rsi:.1f} indicates {rsi_note}. "
            f"The stock is {abs(dist):.1f}% below its 52-week high, "
            f"suggesting {'limited' if dist < 10 else 'significant'} room to recover."
        ),
        "signal":    signal,
        "confidence": conf,
        "risk_level": risk,
        "key_insights": [
            f"RSI {rsi:.1f} — {rsi_note}",
            f"MACD {macd_note}",
            f"Bollinger %B at {boll['pct_b']}% — {'near upper band' if boll['pct_b'] > 70 else 'near lower band' if boll['pct_b'] < 30 else 'mid-range'}",
            f"Price {chg:+.1f}% vs 52-week average"
        ],
        "technical_analysis": (
            f"MACD shows a {macd['crossover']} crossover with histogram at {macd['histogram']:.4f}. "
            f"RSI at {rsi:.1f} and Bollinger %B at {boll['pct_b']}% "
            f"{'suggest buying pressure' if score > 0 else 'suggest selling pressure'}."
        ),
        "fundamental_analysis": (
            f"Forward P/E of {pe} and beta of {beta}. "
            f"Analyst consensus: {fundamentals.get('analyst_rec','N/A')} "
            f"with target ${fundamentals.get('analyst_target','N/A')}."
        ),
        "catalysts": [
            "Upcoming earnings release" if fundamentals.get("earnings_date","N/A") != "N/A" else "Sector rotation opportunities",
            "Potential MACD crossover signal",
            "Analyst price target upside"
        ],
        "risks": [
            "High RSI overbought risk" if rsi > 65 else "Low volume momentum risk",
            f"Beta {beta} — {'high' if str(beta) != 'N/A' and float(str(beta)) > 1.5 else 'moderate'} market sensitivity",
            "Macro / interest rate headwinds"
        ],
        "price_targets": {"bear": bear, "base": base, "bull": bull},
        "time_horizon": "MEDIUM-TERM",
        "sector_outlook": f"Monitor sector trends and broader market conditions for {stock}."
    }


# ─────────────────────────────────────────────
# GROQ-POWERED ANALYSIS
# ─────────────────────────────────────────────

def get_ai_analysis(stock, metrics, fundamentals):
    system = "You are a senior CFA charterholder. Respond ONLY with valid JSON, no markdown, no preamble."
    prompt = f"""Analyze {stock}:
Price ${metrics['latest']:.2f} | 52W Avg ${metrics['avg']:.2f} | vs Avg {metrics['change_pct']:+.1f}%
RSI {metrics['rsi']:.1f} | MACD {metrics['macd']['macd']:.4f} ({metrics['macd']['crossover']}) | Bollinger %B {metrics['bollinger']['pct_b']}%
52W High ${metrics['high_52w']:.2f} (-{metrics['dist_from_high']:.1f}%) | SMA20 ${metrics['sma20']:.2f} | SMA50 ${metrics['sma50']:.2f} | ATR ${metrics['atr']}
Sector {fundamentals['sector']} | Cap {fundamentals['market_cap']} | Fwd P/E {fundamentals['pe_fwd']} | Beta {fundamentals['beta']}
Earnings {fundamentals['earnings_date']} | Analyst {fundamentals['analyst_rec']} Target ${fundamentals['analyst_target']}

Return JSON:
{{"summary":"3 sentences","signal":"STRONG BUY|BUY|HOLD|SELL|STRONG SELL","confidence":"HIGH|MEDIUM|LOW","risk_level":"LOW|MEDIUM|HIGH|VERY HIGH","key_insights":["i1","i2","i3","i4"],"technical_analysis":"2 sentences","fundamental_analysis":"2 sentences","catalysts":["c1","c2","c3"],"risks":["r1","r2","r3"],"price_targets":{{"bear":0,"base":0,"bull":0}},"time_horizon":"SHORT-TERM|MEDIUM-TERM|LONG-TERM","sector_outlook":"1 sentence"}}"""

    raw = groq_chat(system, prompt, max_tokens=1024)
    if not raw:
        return rule_based_analysis(stock, metrics, fundamentals)

    # strip any accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return rule_based_analysis(stock, metrics, fundamentals)


def get_news_sentiment(stock, news_items):
    empty = {"score": 0, "label": "NEUTRAL", "summary": "No recent news found.",
             "headlines": [], "key_themes": []}
    if not news_items:
        return empty

    headlines_text = "\n".join([f"- {n['title']}" for n in news_items[:10]])

    system = "You are a financial sentiment analyst. Respond ONLY with valid JSON, no markdown."
    prompt = f"""Analyze sentiment of these {stock} headlines:
{headlines_text}

Return JSON:
{{"score":<-100 to 100>,"label":"VERY BULLISH|BULLISH|NEUTRAL|BEARISH|VERY BEARISH","summary":"2 sentences","key_themes":["t1","t2","t3"],"notable_headline":"paraphrase most impactful headline"}}"""

    raw = groq_chat(system, prompt, max_tokens=400)
    if not raw:
        # simple rule-based sentiment
        positive_words = ["surge", "beat", "record", "growth", "buy", "upgrade", "profit", "gain", "strong", "bullish"]
        negative_words = ["fall", "miss", "loss", "cut", "downgrade", "sell", "decline", "weak", "bearish", "drop"]
        all_text = " ".join([n["title"].lower() for n in news_items])
        pos = sum(1 for w in positive_words if w in all_text)
        neg = sum(1 for w in negative_words if w in all_text)
        score = min(100, max(-100, (pos - neg) * 15))
        label = "BULLISH" if score > 20 else ("BEARISH" if score < -20 else "NEUTRAL")
        result = {"score": score, "label": label,
                  "summary": f"News sentiment appears {label.lower()} based on recent headlines.",
                  "key_themes": ["earnings", "market trends", "analyst coverage"],
                  "notable_headline": news_items[0]["title"] if news_items else ""}
        result["headlines"] = [{"title": n["title"], "publisher": n.get("publisher",""), "link": n.get("link","")} for n in news_items[:6]]
        return result

    raw = raw.replace("```json","").replace("```","").strip()
    try:
        result = json.loads(raw)
        result["headlines"] = [{"title": n["title"], "publisher": n.get("publisher",""), "link": n.get("link","")} for n in news_items[:6]]
        return result
    except Exception:
        empty["headlines"] = [{"title": n["title"], "publisher": n.get("publisher",""), "link": n.get("link","")} for n in news_items[:6]]
        return empty


# ─────────────────────────────────────────────
# FUNDAMENTALS
# ─────────────────────────────────────────────

def get_fundamentals(ticker_obj):
    empty = {"earnings_date":"N/A","eps_fwd":"N/A","pe_fwd":"N/A","pe_trail":"N/A",
             "peg":"N/A","div_yield":"N/A","market_cap":"N/A","beta":"N/A",
             "short_ratio":"N/A","analyst_target":"N/A","analyst_low":"N/A",
             "analyst_high":"N/A","analyst_rec":"N/A","name":"","sector":"N/A","industry":"N/A"}
    try:
        def fmt_cap(v):
            if not v: return "N/A"
            if v >= 1e12: return f"${v/1e12:.2f}T"
            if v >= 1e9:  return f"${v/1e9:.2f}B"
            if v >= 1e6:  return f"${v/1e6:.2f}M"
            return f"${v:,.0f}"

        # ── Try fast_info first (most reliable in yfinance 0.2.x+) ──
        fi = None
        try:
            fi = ticker_obj.fast_info
        except Exception:
            pass

        # ── Then try full info dict ──
        info = {}
        try:
            info = ticker_obj.info or {}
            # yfinance sometimes returns {trailingPegRatio: null} junk — check it's real
            if not info.get("symbol") and not info.get("shortName") and not info.get("regularMarketPrice"):
                info = {}
        except Exception:
            pass

        # ── Merge: prefer info, fall back to fast_info ──
        def g(key, fi_key=None, default=None):
            v = info.get(key)
            if v is None and fi and fi_key:
                try:
                    v = getattr(fi, fi_key, None)
                except Exception:
                    pass
            return v if v is not None else default

        # Market cap
        market_cap = g("marketCap", "market_cap")
        if not market_cap and fi:
            try: market_cap = fi.market_cap
            except Exception: pass

        # Name / sector
        name     = g("longName") or g("shortName") or ""
        sector   = g("sector", default="N/A")
        industry = g("industry", default="N/A")

        # Valuation
        pe_fwd  = g("forwardPE")
        pe_tr   = g("trailingPE", "pe_ratio")
        peg     = g("pegRatio")
        beta    = g("beta", "beta3_year")
        div     = g("dividendYield")
        short   = g("shortRatio")
        eps_fwd = g("forwardEps")

        # Analyst targets
        at = g("targetMeanPrice")
        al = g("targetLowPrice")
        ah = g("targetHighPrice")

        # Recommendation
        rec     = g("recommendationMean")
        rec_key = (g("recommendationKey") or "").upper().replace("-", " ")
        rec_map = {1: "STRONG BUY", 2: "BUY", 3: "HOLD", 4: "SELL", 5: "STRONG SELL"}
        if rec_key:
            rec_label = rec_key
        elif rec:
            try: rec_label = rec_map.get(round(float(rec)), "N/A")
            except Exception: rec_label = "N/A"
        else:
            rec_label = "N/A"

        # Earnings date — try calendar, then earningsTimestamp
        earnings_date = "N/A"
        try:
            cal = ticker_obj.calendar
            if cal is not None and not cal.empty and "Earnings Date" in cal.index:
                ed  = cal.loc["Earnings Date"]
                val = list(ed)[0] if hasattr(ed, "__iter__") else ed
                earnings_date = str(val)[:10]
        except Exception:
            pass
        if earnings_date == "N/A":
            try:
                et = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
                if et:
                    import datetime
                    earnings_date = datetime.datetime.utcfromtimestamp(et).strftime("%Y-%m-%d")
            except Exception:
                pass

        return {
            "earnings_date":  earnings_date,
            "eps_fwd":        round(float(eps_fwd), 2) if eps_fwd else "N/A",
            "pe_fwd":         round(float(pe_fwd),  1) if pe_fwd  else "N/A",
            "pe_trail":       round(float(pe_tr),   1) if pe_tr   else "N/A",
            "peg":            round(float(peg),     2) if peg     else "N/A",
            "div_yield":      f"{float(div)*100:.2f}%" if div     else "N/A",
            "market_cap":     fmt_cap(market_cap),
            "beta":           round(float(beta),    2) if beta    else "N/A",
            "short_ratio":    round(float(short),   1) if short   else "N/A",
            "analyst_target": round(float(at), 2)      if at      else "N/A",
            "analyst_low":    round(float(al), 2)      if al      else "N/A",
            "analyst_high":   round(float(ah), 2)      if ah      else "N/A",
            "analyst_rec":    rec_label,
            "name":           name,
            "sector":         sector   or "N/A",
            "industry":       industry or "N/A",
        }
    except Exception as e:
        print(f"get_fundamentals error: {e}")
        return empty


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

def clean_for_json(obj):
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None

    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]

    return obj


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search_stocks():
    """Search for stock symbols using yfinance + Yahoo Finance search API."""
    query = request.args.get("q", "").strip()
    if len(query) < 1:
        return jsonify([])
    try:
        import urllib.parse
        # Yahoo Finance search endpoint
        url = "https://query2.finance.yahoo.com/v1/finance/search?q=" + \
              urllib.parse.quote(query) + \
              "&quotesCount=8&newsCount=0&listsCount=0&enableFuzzyQuery=false"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible)"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for q in data.get("quotes", [])[:8]:
            symbol   = q.get("symbol", "")
            name     = q.get("longname") or q.get("shortname") or ""
            q_type   = q.get("quoteType", "")
            exchange = q.get("exchange", "")
            # Filter to equities and ETFs on major exchanges
            if q_type in ("EQUITY", "ETF", "INDEX") and symbol:
                results.append({
                    "symbol":   symbol,
                    "name":     name,
                    "type":     q_type,
                    "exchange": exchange
                })
        return jsonify(results)
    except Exception as e:
        return jsonify([])


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data  = request.get_json()
        stock = data["stock"].upper().strip()
        ticker = yf.Ticker(stock)
        df = yf.download(stock, period="1y", progress=False, auto_adjust=True)
        if df.empty:
            return jsonify({"error": f"Invalid symbol: {stock}"})

        close  = df["Close"].squeeze().dropna()
        volume = df["Volume"].squeeze().dropna()
        prices = close.tolist()

        avg_price    = float(close.mean())
        latest_price = float(close.iloc[-1])
        high_52w     = float(close.max())
        low_52w      = float(close.min())
        price_change = latest_price - avg_price
        change_pct   = (price_change / avg_price) * 100
        dist_from_high = ((high_52w - latest_price) / high_52w) * 100

        sma20  = float(close.rolling(20).mean().iloc[-1])
        sma50  = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        rsi    = compute_rsi(prices)
        macd   = compute_macd(prices)
        boll   = compute_bollinger(prices)
        atr    = compute_atr(df)

        avg_vol    = float(volume.mean())
        recent_vol = float(volume.iloc[-5:].mean())
        vol_trend  = "High" if recent_vol > avg_vol * 1.2 else ("Low" if recent_vol < avg_vol * 0.8 else "Normal")

        metrics = {
            "latest": latest_price, "avg": avg_price,
            "change": price_change, "change_pct": change_pct,
            "high_52w": high_52w, "low_52w": low_52w,
            "dist_from_high": dist_from_high,
            "rsi": rsi, "macd": macd, "bollinger": boll, "atr": atr,
            "sma20": sma20, "sma50": sma50, "volume_trend": vol_trend
        }

        fundamentals = get_fundamentals(ticker)

        # News
        try:
            raw_news = ticker.news or []
            news_items = []
            for n in raw_news[:12]:
                title = n.get("title","")
                if not title:
                    c = n.get("content",{})
                    title = c.get("title","") if isinstance(c, dict) else ""
                publisher = n.get("publisher","")
                if not publisher:
                    c = n.get("content",{})
                    if isinstance(c, dict):
                        p = c.get("provider",{})
                        publisher = p.get("displayName","") if isinstance(p, dict) else ""
                link = n.get("link","")
                if not link:
                    c = n.get("content",{})
                    if isinstance(c, dict):
                        cu = c.get("canonicalUrl",{})
                        link = cu.get("url","") if isinstance(cu, dict) else str(cu)
                if title:
                    news_items.append({"title": title, "publisher": publisher, "link": link})
            sentiment = get_news_sentiment(stock, news_items)
        except Exception:
            sentiment = {"score":0,"label":"NEUTRAL","summary":"News unavailable.","headlines":[],"key_themes":[]}

        ai = get_ai_analysis(stock, metrics, fundamentals)

        def safe_list(series):
            return [round(float(v), 2) if v == v else None for v in series.tolist()]

        chart = {
            "prices":          [round(p, 2) for p in prices[-90:]],
            "sma20":           safe_list(close.rolling(20).mean().iloc[-90:]),
            "sma50":           safe_list(close.rolling(50).mean().iloc[-90:]),
            "dates":           df.index.strftime("%b %d").tolist()[-90:],
            "bollinger_upper": safe_list(close.rolling(20).mean().add(close.rolling(20).std() * 2).iloc[-90:]),
            "bollinger_lower": safe_list(close.rolling(20).mean().sub(close.rolling(20).std() * 2).iloc[-90:]),
            "volumes":         [int(v) for v in volume.iloc[-90:].tolist()],
        }

        return jsonify(clean_for_json({
            "stock": stock,
            "name":  fundamentals.get("name", stock),
            "ai_powered": bool(GROQ_API_KEY),
            "metrics": {
                "avg": round(avg_price, 2), "latest": round(latest_price, 2),
                "change": round(price_change, 2), "change_pct": round(change_pct, 2),
                "high_52w": round(high_52w, 2), "low_52w": round(low_52w, 2),
                "dist_from_high": round(dist_from_high, 1),
                "rsi": round(rsi, 1), "macd": macd, "bollinger": boll, "atr": atr,
                "sma20": round(sma20, 2), "sma50": round(sma50, 2),
                "sma200": round(sma200, 2) if sma200 else None,
                "volume_trend": vol_trend,
                "trend": "Uptrend" if latest_price > avg_price else "Downtrend"
            },
            "fundamentals": fundamentals,
            "sentiment":    sentiment,
            "ai":           ai,
            "chart":        chart
        }))

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        body     = request.get_json()
        messages = body.get("messages", [])
        ctx      = body.get("context", {})

        system = (
            f"You are an expert financial analyst AI in QuantView dashboard. "
            f"Stock: {ctx.get('stock','N/A')} | Price: ${ctx.get('price','N/A')} | "
            f"Signal: {ctx.get('signal','N/A')} | RSI: {ctx.get('rsi','N/A')} | "
            f"Sector: {ctx.get('sector','N/A')} | Earnings: {ctx.get('earnings_date','N/A')}. "
            f"Be direct and data-driven. Keep answers to 3-5 sentences."
        )

        # Build conversation for Groq
        last_user = messages[-1]["content"] if messages else ""
        reply = groq_chat(system, last_user, max_tokens=600)

        if not reply:
            # Rule-based chat fallback
            q = last_user.lower()
            stock = ctx.get("stock","the stock")
            rsi   = ctx.get("rsi", 50)
            sig   = ctx.get("signal","HOLD")
            if "entry" in q or "buy" in q:
                reply = f"Based on the current signal of {sig}, RSI at {rsi}, consider entering {stock} on a pullback to support. Always use a stop-loss."
            elif "target" in q or "price" in q:
                reply = f"Analyst consensus and technical targets suggest monitoring the base case. Signal is {sig} with RSI at {rsi}."
            elif "risk" in q:
                reply = f"Key risks include earnings volatility, macro headwinds, and technical resistance. RSI at {rsi} — {'overbought caution' if float(str(rsi)) > 65 else 'monitor momentum'}."
            elif "technical" in q:
                reply = f"RSI at {rsi} ({'overbought' if float(str(rsi)) > 70 else 'oversold' if float(str(rsi)) < 30 else 'neutral'}). Current signal: {sig}. Check MACD and Bollinger Bands in the Analysis tab for more detail."
            else:
                reply = f"Current signal for {stock} is {sig} with RSI at {rsi}. Check the Analysis and Fundamentals tabs for a complete picture. Add your GROQ_API_KEY for full AI chat."

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/portfolio/quote", methods=["POST"])
def portfolio_quote():
    try:
        symbols = request.get_json().get("symbols", [])
        result  = {}
        for sym in symbols:
            try:
                df = yf.download(sym, period="2d", progress=False, auto_adjust=True)
                if not df.empty:
                    c    = df["Close"].squeeze()
                    cur  = float(c.iloc[-1])
                    prev = float(c.iloc[-2]) if len(c) > 1 else cur
                    result[sym] = {
                        "price":      round(cur, 2),
                        "change":     round(cur - prev, 2),
                        "change_pct": round((cur - prev) / prev * 100, 2)
                    }
            except Exception:
                result[sym] = {"price": 0, "change": 0, "change_pct": 0}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/portfolio/analyze", methods=["POST"])
def portfolio_analyze():
    try:
        body        = request.get_json()
        holdings    = body.get("holdings", [])
        total_value = body.get("total_value", 0)
        total_pnl   = body.get("total_pnl", 0)

        lines = "\n".join([
            f"- {h['symbol']}: {h['shares']} shares @ ${h['avg_cost']} | "
            f"Now ${h['current_price']} | P&L ${h['pnl']:+.2f} ({h['pnl_pct']:+.1f}%) | Weight {h['weight']:.1f}%"
            for h in holdings
        ])

        system = "You are a portfolio manager. Respond ONLY with valid JSON, no markdown."
        prompt = (
            f"Analyze portfolio. Value: ${total_value:,.2f} | P&L: ${total_pnl:+,.2f}\n{lines}\n"
            f'Return JSON: {{"assessment":"2-3 sentences","diversification":"POOR|FAIR|GOOD|EXCELLENT",'
            f'"risk_score":<1-10>,"top_concern":"biggest risk","recommendations":["r1","r2","r3"],'
            f'"rebalance_suggestions":["s1","s2"]}}'
        )

        raw = groq_chat(system, prompt, max_tokens=500)

        if not raw:
            # Rule-based portfolio assessment
            n = len(holdings)
            diversification = "POOR" if n < 3 else ("FAIR" if n < 5 else ("GOOD" if n < 8 else "EXCELLENT"))
            pnl_pct = (total_pnl / (total_value - total_pnl) * 100) if (total_value - total_pnl) else 0
            return jsonify({
                "assessment": f"Portfolio of {n} positions with total P&L of ${total_pnl:+,.2f} ({pnl_pct:+.1f}%). {'Diversification is limited — consider adding more positions.' if n < 4 else 'Portfolio has reasonable diversification.'}",
                "diversification": diversification,
                "risk_score": max(1, min(10, 10 - n)),
                "top_concern": "Concentration risk" if n < 4 else "Monitor individual position sizing",
                "recommendations": [
                    "Review stop-loss levels for all positions",
                    "Consider trimming positions with >20% gains",
                    "Maintain cash reserve for new opportunities"
                ],
                "rebalance_suggestions": [
                    "Ensure no single position exceeds 25% of portfolio",
                    "Add sector diversification if over-concentrated"
                ]
            })

        raw = raw.replace("```json","").replace("```","").strip()
        return jsonify(json.loads(raw))
    except Exception as e:
        return jsonify({"error": str(e)})


# ─────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────

def backtest_strategy(prices, dates, strategy="combined"):
    """
    Simulate a signal-based trading strategy over historical data.
    Returns directional accuracy, total return, win rate, Sharpe ratio,
    max drawdown, and a full trade log.

    Strategies:
      rsi       — buy <30, sell >70
      macd      — buy on bullish crossover, sell on bearish
      combined  — both RSI + MACD must agree (higher precision)
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)

    # ── Pre-compute indicators for every bar ──
    # RSI series
    def rsi_series(px, period=14):
        out = [np.nan] * period
        delta = np.diff(px)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        ag = np.mean(gain[:period])
        al = np.mean(loss[:period])
        for i in range(period, len(delta)):
            ag = (ag * (period - 1) + gain[i]) / period
            al = (al * (period - 1) + loss[i]) / period
            rs = ag / al if al != 0 else 100
            out.append(100 - (100 / (1 + rs)))
        return out

    def macd_series(px):
        k12 = 2 / 13; k26 = 2 / 27; k9 = 2 / 10
        ema12 = [px[0]]; ema26 = [px[0]]
        for p in px[1:]:
            ema12.append(p * k12 + ema12[-1] * (1 - k12))
            ema26.append(p * k26 + ema26[-1] * (1 - k26))
        macd_line = [a - b for a, b in zip(ema12, ema26)]
        sig = [macd_line[0]]
        for m in macd_line[1:]:
            sig.append(m * k9 + sig[-1] * (1 - k9))
        return macd_line, sig

    rsi_vals = rsi_series(prices)
    macd_line, macd_sig = macd_series(prices)

    # ── Signal generation ──
    position   = 0       # 1 = long, 0 = flat
    entry_price = 0.0
    trades     = []      # list of closed trades
    equity     = [1.0]   # normalised equity curve
    capital    = 1.0

    for i in range(26, n - 1):          # need enough bars for MACD
        rsi   = rsi_vals[i]
        m     = macd_line[i];   s = macd_sig[i]
        m_prev = macd_line[i-1]; s_prev = macd_sig[i-1]
        bullish_cross = (m > s) and (m_prev <= s_prev)
        bearish_cross = (m < s) and (m_prev >= s_prev)

        if strategy == "rsi":
            buy_sig  = (rsi is not None and rsi < 30)
            sell_sig = (rsi is not None and rsi > 70)
        elif strategy == "macd":
            buy_sig  = bullish_cross
            sell_sig = bearish_cross
        else:  # combined — both must agree
            buy_sig  = bullish_cross and rsi is not None and rsi < 50
            sell_sig = bearish_cross and rsi is not None and rsi > 50

        next_price = prices[i + 1]

        if buy_sig and position == 0:
            position    = 1
            entry_price = prices[i]

        elif sell_sig and position == 1:
            ret = (next_price - entry_price) / entry_price
            capital *= (1 + ret)
            trades.append({
                "entry_date":  dates[i - 1],
                "exit_date":   dates[i],
                "entry_price": round(float(entry_price), 2),
                "exit_price":  round(float(next_price), 2),
                "return_pct":  round(float(ret * 100), 2),
                "win":         bool(ret > 0)
            })
            position = 0

        equity.append(round(capital, 4))

    # Close any open position at last bar
    if position == 1:
        ret = (prices[-1] - entry_price) / entry_price
        capital *= (1 + ret)
        trades.append({
            "entry_date":  dates[-2],
            "exit_date":   dates[-1],
            "entry_price": round(float(entry_price), 2),
            "exit_price":  round(float(prices[-1]), 2),
            "return_pct":  round(float(ret * 100), 2),
            "win":         bool(ret > 0)
        })

    if not trades:
        return None

    # ── Metrics ──
    wins      = [t for t in trades if t["win"]]
    losses    = [t for t in trades if not t["win"]]
    win_rate  = len(wins) / len(trades) * 100

    # Directional accuracy: did signal correctly predict next-day direction?
    correct = 0; total_signals = 0
    for i in range(26, n - 1):
        rsi = rsi_vals[i]
        m   = macd_line[i]; s = macd_sig[i]
        m_p = macd_line[i-1]; s_p = macd_sig[i-1]
        bc  = (m > s) and (m_p <= s_p)
        sc  = (m < s) and (m_p >= s_p)
        actual_up = prices[i + 1] > prices[i]
        if strategy == "rsi":
            if rsi is not None and rsi < 30:
                total_signals += 1
                if actual_up: correct += 1
            elif rsi is not None and rsi > 70:
                total_signals += 1
                if not actual_up: correct += 1
        elif strategy == "macd":
            if bc:
                total_signals += 1
                if actual_up: correct += 1
            elif sc:
                total_signals += 1
                if not actual_up: correct += 1
        else:
            if bc and rsi is not None and rsi < 50:
                total_signals += 1
                if actual_up: correct += 1
            elif sc and rsi is not None and rsi > 50:
                total_signals += 1
                if not actual_up: correct += 1

    dir_accuracy = round(correct / total_signals * 100, 1) if total_signals else 0

    # Sharpe ratio (annualised, assume 252 trading days)
    rets = np.diff(equity)
    sharpe = 0.0
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = round(float(np.mean(rets) / np.std(rets) * np.sqrt(252)), 2)

    # Max drawdown
    eq_arr  = np.array(equity)
    peak    = np.maximum.accumulate(eq_arr)
    dd      = (eq_arr - peak) / peak
    max_dd  = round(float(dd.min()) * 100, 2)

    # Buy-and-hold benchmark — cast everything to plain Python float/int
    bh_return    = round(float((prices[-1] - prices[26]) / prices[26] * 100), 2)
    total_return = round(float((capital - 1) * 100), 2)
    avg_win      = round(float(np.mean([t["return_pct"] for t in wins])),   2) if wins   else 0.0
    avg_loss     = round(float(np.mean([t["return_pct"] for t in losses])), 2) if losses else 0.0

    return {
        "strategy":             strategy,
        "total_trades":         int(len(trades)),
        "total_signals":        int(total_signals),
        "directional_accuracy": float(dir_accuracy),
        "win_rate":             round(float(win_rate), 1),
        "total_return":         total_return,
        "bh_return":            bh_return,
        "sharpe":               float(sharpe),
        "max_drawdown":         float(max_dd),
        "avg_win":              avg_win,
        "avg_loss":             avg_loss,
        "trades":               trades[-20:],
        "equity_curve":         [round(float(e), 4) for e in equity[-90:]],
        "equity_dates":         dates[-90:]
    }


@app.route("/backtest", methods=["POST"])
def backtest():
    try:
        body     = request.get_json()
        stock    = body.get("stock", "").upper().strip()
        strategy = body.get("strategy", "combined")
        period   = body.get("period", "1y")

        if not stock:
            return jsonify({"error": "No symbol provided"})

        df = yf.download(stock, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return jsonify({"error": f"Invalid symbol: {stock}"})

        close  = df["Close"].squeeze().dropna()
        prices = close.tolist()
        dates  = df.index.strftime("%b %d '%y").tolist()

        if len(prices) < 60:
            return jsonify({"error": "Not enough data — try a longer period"})

        result = backtest_strategy(prices, dates, strategy)
        if not result:
            return jsonify({"error": "No trades generated — try a different strategy or period"})

        result["stock"]  = stock
        result["period"] = period
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    mode = "Groq AI (llama3-70b)" if GROQ_API_KEY else "Rule-Based (no API key)"
    print(f"QuantView starting — Analysis mode: {mode}")
    print("Get free Groq API key at: https://console.groq.com")
    app.run(debug=True)