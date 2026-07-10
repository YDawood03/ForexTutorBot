"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# We import the 'app' object initialized inside __init__.py
from ForexTutorBot import app

# Comprehensive Forex Knowledge Base
knowledge_base = [
    # SECTION 1: FOREX BASICS & OVERVIEW
    {"q": "What is forex?", "a": "Forex (Foreign Exchange) is the global decentralized market where international currencies are traded. It is the largest and most liquid financial market in the world, with over $5.3 trillion traded daily."},
    {"q": "What is the U.S. Dollar Index?", "a": "The U.S. Dollar Index (DXY) measures the value of the USD against a basket of six major currencies: Euro, Japanese Yen, British Pound, Canadian Dollar, Swedish Krona, and Swiss Franc. It's a key indicator of the dollar's strength in global markets."},
    {"q": "What is intermarket analysis?", "a": "Intermarket analysis examines the relationships between different asset classes - currencies, commodities, bonds, and equities. Institutional traders watch gold, oil, bonds, and global liquidity because each one moves currencies in ways most retail traders never see."},
    {"q": "How do equities affect FX trading?", "a": "Equity markets and currencies are closely linked. Strong stock markets often attract foreign investment, strengthening the local currency. Conversely, weak equities can signal economic problems, potentially weakening the currency. Traders watch the S&P 500, Nikkei, and other major indices for FX signals."},
    {"q": "What is the daily trading volume of the forex market?", "a": "The forex market trades approximately $5.3 to $5.5 TRILLION daily. In comparison, the NYSE trades about $22.4 billion daily, and the London Stock Exchange trades about $7.2 billion daily."},
    {"q": "What are the trading hours of the forex market?", "a": "The forex market is open 24 hours a day, 5 days a week. Trading shifts between financial centers: Sydney, Tokyo, London, Frankfurt, and New York, creating a continuous global trading cycle."},
    {"q": "What are the major forex trading sessions?", "a": "The major sessions are: Sydney (9 PM - 5 AM GMT), Tokyo/Asian (11 PM - 7 AM GMT), Frankfurt/European (7 AM - 3 PM GMT), London (8 AM - 4 PM GMT), and New York (1 PM - 9 PM GMT)."},
    {"q": "What are session overlaps and why are they important?", "a": "Session overlaps occur when two major financial centers are open simultaneously. These offer highly liquid market conditions with good quality, sustained moves. The London-New York overlap (1-4 PM GMT) is particularly liquid."},
    {"q": "When should I trade specific currency pairs?", "a": "Trade currency pairs during their corresponding session. For example, AUD/USD moves during Sydney session due to news releases, while EUR/GBP is better traded during European hours."},

    # SECTION 2: CURRENCY PAIRS
    {"q": "What are the most actively traded currencies?", "a": "The most popular currencies are: US Dollar (USD), Euro (EUR), Japanese Yen (JPY), Pound Sterling (GBP), Australian Dollar (AUD), Swiss Franc (CHF), and Canadian Dollar (CAD)."},
    {"q": "What is the ranking order for currency pairs?", "a": "The universally accepted ranking order: 1. EUR, 2. GBP, 3. AUD, 4. NZD, 5. USD, 6. CAD, 7. CHF, 8. JPY. Higher-ranked currencies become the base currency when paired with lower-ranked ones."},
    {"q": "What is the base currency and quote currency?", "a": "The base currency is the first currency listed (e.g., EUR in EUR/USD). The quote currency is the second (e.g., USD in EUR/USD). The price shows how much of the quote currency is required to buy 1 unit of the base currency."},
    {"q": "How do I read a forex quote?", "a": "A forex quote like EUR/USD at 1.2000 means 1 Euro buys 1.2000 US Dollars. The bid price is what you can buy at, the ask price is what you can sell at. The spread is the difference between bid and ask."},
    {"q": "What are the most commonly traded currency pairs?", "a": "The most common pairs are: EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, and USD/CAD. These are the 'Majors' and offer the tightest spreads."},
    {"q": "What are the 'Majors' in Forex?", "a": "The Majors are the most traded, most liquid currency pairs paired with USD. Average spreads: EUR/USD 0.1 pips, USD/JPY 0.5 pips, GBP/USD 0.6 pips, AUD/USD 0.4 pips."},
    {"q": "What are 'Minors' in Forex?", "a": "Minors are non-USD currency pairs. They experience wider swings and spreads. Examples include: EUR/JPY (2.1 pips), EUR/GBP (3.5 pips), GBP/CHF (3.3 pips), AUD/NZD (1.4 pips)."},
    {"q": "What are 'Crosses' or 'Currency Crosses'?", "a": "Crosses are pairs where USD is not required, bypassing the historical step of converting to USD first. Examples include GBP/JPY and EUR/AUD."},
    {"q": "What are 'Exotics' in Forex?", "a": "Exotics are rarely traded currency pairs pairing a major with a developing market currency, such as EUR/TRY or CAD/SGD. They typically have wide spreads and lower liquidity."},
    {"q": "What are the nicknames for currency pairs?", "a": "Common nicknames: Cable (GBP/USD - from submarine cable), Fiber (EUR/USD - modern upgrade), Aussie (AUD/USD), Kiwi (NZD/USD), Swissy (USD/CHF), Loonie (USD/CAD), Guppy (GBP/JPY), Yuppy (EUR/JPY), Chunnel (EUR/GBP)."},
    {"q": "What are the different types of currency pairs?", "a": "Currency pairs are categorized into three groups: Majors (pairs containing the USD paired with another heavily traded economy, e.g., EUR/USD, GBP/USD), Minors/Crosses (major currencies excluding the USD, e.g., EUR/GBP, AUD/JPY), and Exotics (one major currency paired with a developing economy, e.g., USD/TRY, EUR/ZAR)."},

    # SECTION 3: GOING LONG & SHORT
    {"q": "What is a 'long' position in Forex?", "a": "Going long means buying a currency pair, expecting the base currency to rise against the quote currency. You profit if EUR rises against USD when going long EUR/USD."},
    {"q": "What is a 'short' position in Forex?", "a": "Going short means selling a currency pair, expecting the base currency to fall against the quote currency. You profit if EUR falls against USD when going short EUR/USD."},

    # SECTION 4: LOT SIZES & LEVERAGE
    {"q": "What are standard lot sizes in Forex?", "a": "Standard lot = 100,000 units, Mini lot = 10,000 units, Micro lot = 1,000 units. Most brokers allow trading 0.01 of a standard lot, effectively allowing micro lot trading."},
    {"q": "What is leverage in Forex?", "a": "Leverage allows you to control more money than your account balance. With 100:1 leverage, you can control $100,000 with only $1,000 in your account. It amplifies both profits and losses."},
    {"q": "What is margin in Forex?", "a": "Margin is the amount required in your account to open a leveraged position - a 'good faith deposit.' For example, $1,000 margin for a $100,000 position with 100:1 leverage."},
    {"q": "What happens during a margin call?", "a": "If your equity falls below 100% of margin requirements, you get a notification. If equity falls to 50%, your positions are automatically closed ('blown account'), leaving a fraction of your original balance."},
    {"q": "How can I avoid a margin call?", "a": "Risk no more than 1-2% per trade, use proper position sizing, don't open too many positions, monitor correlated positions, and keep your account sufficiently funded."},

    # SECTION 5: PIPS
    {"q": "What is a pip in Forex?", "a": "A pip (Percentage in Point) is the smallest unit of price movement. For most pairs (EUR/USD, GBP/USD), a pip is 0.0001. For JPY pairs, a pip is 0.01."},
    {"q": "How do I calculate pip value for EUR/USD with a USD account?", "a": "Pip Value = 0.0001 × Units. Example: 25,000 EUR/USD (0.25 lots) = 0.0001 × 25,000 = $2.50 per pip."},
    {"q": "How do I calculate pip value for a non-USD account with EUR/USD?", "a": "Pip Value = (0.0001 × Units) / AUDUSD rate. Example: AUD/USD at 0.7150, 25,000 units = ($2.50) / 0.7150 = $3.50 AUD per pip."},
    {"q": "How do I calculate pip value when USD is the base currency (USD/CHF)?", "a": "Pip Value = 0.0001 × Units / Quote price. Example: 25,000 USD/CHF at 0.9915 = ($2.50) / 0.9915 = $2.52 per pip."},
    {"q": "How do I calculate pip value for crosses with no USD (EUR/GBP)?", "a": "Pip Value = 0.0001 × Units × Quote currency rate (GBP/USD). Example: 25,000 EUR/GBP with GBP/USD at 1.4350 = ($2.50) × 1.4350 = $3.59."},
    {"q": "What multiplier do I use for JPY pairs?", "a": "For pairs where JPY is the quote currency (USD/JPY, EUR/JPY), use a multiplier of 0.01 instead of 0.0001."},

    # SECTION 6: ORDER TYPES
    {"q": "What is a Market Order?", "a": "A market order executes instantly at the current market price. Click buy/sell at market and you're in the trade immediately at the best available price."},
    {"q": "What is a Stop Order?", "a": "A stop order is used to buy above the market (Buy Stop) or sell below the market (Sell Stop). Best for trading breakouts or trend continuation strategies."},
    {"q": "What is a Limit Order?", "a": "A limit order is used to buy below the market (Buy Limit) or sell above the market (Sell Limit). Best for trading expected reversals at specific price levels."},

    # SECTION 7: SWAP TRADING
    {"q": "What is swap in Forex?", "a": "Swap is an interest fee paid or charged at the end of each trading day. You receive interest on long positions and pay interest on short positions. The net difference is called the carry."},
    {"q": "What is positive carry?", "a": "Positive carry occurs when you receive more interest than you pay. The amount is added directly to your account. Positive carry is typically earned when holding high-yielding currencies."},
    {"q": "What is negative carry?", "a": "Negative carry occurs when you pay more interest than you receive. The amount is subtracted from your account. Negative carry occurs when holding low-yielding currencies."},
    {"q": "Can I make money from swap trading?", "a": "Yes, through carry trades - buying high-yielding currencies and selling low-yielding ones to earn the interest differential. However, the pair must be trending in your direction to be profitable overall."},
    {"q": "What are examples of low-yielding (funding) currencies?", "a": "Low-yielding currencies include: Japanese Yen (JPY), Swiss Franc (CHF), and Euro (EUR). These are typically used as funding currencies in carry trades."},
    {"q": "What are examples of high-yielding currencies?", "a": "High-yielding currencies include: Australian Dollar (AUD), New Zealand Dollar (NZD), South African Rand (ZAR), and other exotic currencies with higher interest rates."},

    # SECTION 8: FUNDAMENTAL ANALYSIS
    {"q": "What is fundamental analysis in Forex?", "a": "Fundamental analysis studies economic fundamentals of a currency, country, or economy. It involves analyzing economic data releases, central bank decisions, and news headlines."},
    {"q": "Why do economic news releases affect currency value?", "a": "A healthy economic outlook attracts investment, increasing currency demand. This can force central banks to raise rates, further attracting foreign investment and strengthening the currency."},
    {"q": "What matters more - actual numbers or market expectations?", "a": "Market EXPECTATIONS matter more than actual numbers. It's not uncommon for good news to send a currency falling because the market had already priced in an even better result."},
    {"q": "What are the three main types of fundamental analysis events?", "a": "1. Economic data releases, 2. Central Bank Decisions, 3. News Headlines."},
    {"q": "What is the US Non-Farm Payroll (NFP) report?", "a": "NFP is the monthly change in employed people in the US economy (excluding farming). It's a major barometer for the health of the US economy and the globe."},
    {"q": "What is hawkish monetary policy?", "a": "Hawkish policy indicates rising interest rates (tightening). The central bank slows the economy in response to higher inflation. Borrowing becomes more expensive, reducing spending."},
    {"q": "What is dovish monetary policy?", "a": "Dovish policy indicates falling interest rates (loosening). The central bank cuts rates to stimulate a stagnating economy. Money becomes cheaper, encouraging investment and spending."},
    {"q": "What are the three main goals of monetary policy?", "a": "1. Economic growth targets, 2. Inflation in the target band, 3. Low unemployment."},
    {"q": "What is event risk in fundamental analysis?", "a": "Event risk refers to anything that will move markets but can't be predicted: natural disasters, terrorist attacks, declarations of war, and political tensions."},
    {"q": "What is a country profile in forex?", "a": "A country profile analyzes a nation's economic fundamentals: GDP growth, inflation rates, employment data, political stability, central bank policies, and trade balances. Understanding these helps predict currency strength or weakness."},
    {"q": "What is global liquidity?", "a": "Global liquidity refers to the availability of credit and money in the international financial system. Central bank policies, interbank lending, and capital flows affect liquidity. More liquidity typically weakens currencies, while tightening liquidity can strengthen them."},
    {"q": "What is narrative analysis?", "a": "Narrative analysis in forex involves understanding market stories and sentiment - why traders believe currencies will move in certain directions. It combines fundamental analysis with market psychology to identify trading opportunities."},
    {"q": "What news affects the forex markets the most?", "a": "High-impact economic news releases create the most volatility in Forex. The most important reports include Interest Rate Decisions from Central Banks (like the Fed or ECB), Non-Farm Payrolls (NFP) job data from the US, Consumer Price Index (CPI) inflation data, and Gross Domestic Product (GDP) growth numbers."},

    # SECTION 9: TECHNICAL ANALYSIS - PRICE ACTION
    {"q": "What is technical analysis?", "a": "Technical analysis uses charts to study historical price movement to determine future price direction. The principle is that everything you need to know has been reflected in price."},
    {"q": "What are the three types of charts used in Forex?", "a": "1. Line Chart - shows only closing price; 2. Bar Chart - shows high, low, open, and close; 3. Candlestick Chart - like bar charts but adds dimension and color to depict the open-close range."},
    {"q": "What are the components of a candlestick?", "a": "Body (difference between open and close), Wicks/Shadows (high and low), Up/Bullish candle (close higher than open), Down/Bearish candle (close lower than open), Doji (open equals close)."},
    {"q": "What does a large wick relative to the body indicate?", "a": "A large wick relative to the body indicates a potential turning point (support/resistance), suggesting price was rejected from that level."},
    {"q": "What is a Doji candlestick?", "a": "A Doji occurs when the open and close prices are equal (or very close), creating a candle with no body - just a line. It represents indecision in the market."},
    {"q": "What is a Shooting Star candlestick pattern?", "a": "A Shooting Star is a single-candle bearish reversal pattern at the end of an uptrend. Price moves higher but closes near the open, leaving a long wick with a short body (wick at least 1.5x body length)."},
    {"q": "What is a Bearish Engulfing pattern?", "a": "A Bearish Engulfing has a candle that closes lower with a body that completely engulfs the previous candle's body, signalling a major shift in sentiment."},
    {"q": "What is a Hanging Man candlestick pattern?", "a": "A Hanging Man is a bearish reversal candle at peaks. Price moves significantly lower but finishes near the open, leaving a long wick below and small body."},
    {"q": "What is the difference between a Hanging Man and a Hammer?", "a": "Both have the same shape (long lower wick, small body at top). A Hanging Man occurs at peaks (bearish reversal). A Hammer occurs at bottoms (bullish reversal). The context determines which it is."},
    {"q": "What is a Bullish Hammer?", "a": "A Bullish Hammer is identical to a Hanging Man but occurs at the bottom of downtrends. Price moves lower but finishes near the open, leaving a long wick below."},
    {"q": "What is a Bullish Engulfing pattern?", "a": "A Bullish Engulfing is an up candle at the end of a downtrend where the body completely engulfs the previous candle's body, signalling a major shift in sentiment."},
    {"q": "What is a Head and Shoulders pattern?", "a": "A bearish reversal with four components: Left Shoulder (small top), Head (break above left shoulder), Right Shoulder (lower high), Neckline (connects shoulder lows). Pattern confirms when price breaks below the neckline."},
    {"q": "What is an Inverse Head and Shoulders pattern?", "a": "The Inverse Head and Shoulders is identical but upside down, signifying a potential bottom (bullish reversal). Confirmed when price breaks above the neckline."},
    {"q": "What is a Double Top pattern?", "a": "A Double Top (or 'M') is a bearish reversal characterized by two tops of similar magnitudes. Entry trigger is when price breaks below the 'Confirmation Line' connecting the two origin points."},
    {"q": "What is a Double Bottom pattern?", "a": "A Double Bottom (or 'W') is the inverse of a Double Top, signaling a potential bottom (bullish reversal). Entry trigger is when price breaks above the Confirmation Line."},
    {"q": "What is an Ascending Triangle?", "a": "A bullish continuation pattern characterized by a series of higher lows failing at a flat top. It's a 'terminal' pattern - eventually the top must break. Entry trigger is when price breaks above the top."},
    {"q": "What is a Descending Triangle?", "a": "A bearish continuation pattern characterized by a series of lower highs meeting a flat bottom. Traders enter short when the flat bottom is taken out."},
    {"q": "What is a Bull Flag (Pennant)?", "a": "A bullish continuation pattern with a series of parallel lower highs and lower lows within a dominant uptrend. A buy signal is triggered when the upper parallel is breached."},
    {"q": "What is a Bear Flag?", "a": "A bearish continuation pattern with a series of parallel higher lows and higher highs within a dominant downtrend. Traders enter short when the lower parallel breaks."},

    # SECTION 10: SUPPORT, RESISTANCE & FIBONACCI
    {"q": "What is support in Forex trading?", "a": "Support is a lower price point or zone where a currency pair is considered 'cheap,' spurring buying interest. It's a level where price is likely to bounce or reverse upward. Think of support as the floor."},
    {"q": "What is resistance in Forex trading?", "a": "Resistance is an upper price point or zone where a currency pair is considered 'expensive,' encountering selling interest. It's a level where price is likely to bounce or reverse downward. Think of resistance as the ceiling."},
    {"q": "What happens when a support level breaks?", "a": "Former support often becomes resistance. Levels that previously encouraged buying interest will nearly always encourage selling interest after they break down."},
    {"q": "What is trend support?", "a": "Trend support is a dynamic upward-sloping line connecting higher lows in an uptrend. Price tends to find buying interest whenever it nears trend support."},
    {"q": "What is trend resistance?", "a": "Trend resistance is a dynamic downward-sloping line connecting lower highs in a downtrend. Price tends to find selling interest whenever it nears trend resistance."},
    {"q": "What is the basic support/resistance trading strategy?", "a": "Buy low, sell high - buy into support (when price reaches support levels) and sell into resistance (when price reaches resistance levels). This offers high probability setups with tight stops."},
    {"q": "What are Fibonacci retracement levels?", "a": "Fibonacci retracements are horizontal lines that indicate where potential support and resistance levels are likely to occur. They are based on Fibonacci numbers (23.6%, 38.2%, 50%, 61.8%, 78.6%) and are used to find entry points during market pullbacks."},
    {"q": "What are the standard Fibonacci retracement levels?", "a": "The standard levels are 23.6%, 38.2%, 50%, 61.8%, and 78.6%. Many traders add the 50% level even though it's not derived from the Fibonacci sequence."},
    {"q": "What does the 50% Fibonacci level represent?", "a": "The 50% level is not technically a Fibonacci ratio but is included because 50% retracements are very common in markets. It represents a halfway point of the move."},
    {"q": "What does the 78.6% Fibonacci level indicate?", "a": "The 78.6% level is the 'be all and end all.' If a pair retraces more than 78.6% of the prior move, chances are it's heading straight back to the origin (100%)."},
    {"q": "How do I use the Fibonacci tool in an uptrend?", "a": "In an uptrend, drag the Fibonacci tool from the low to the high. This identifies potential support levels where price might correct before continuing higher."},
    {"q": "How do I use the Fibonacci tool in a downtrend?", "a": "In a downtrend, drag the Fibonacci tool from the high to the low. This identifies potential resistance levels where price might correct before continuing lower."},
    {"q": "What are supply and demand zones?", "a": "Supply and demand zones are broad price areas on a chart where institutional buying or selling occurred heavily in the past. A demand zone is where buyers flooded the market (causing price to rocket up), while a supply zone is where sellers overwhelmed buyers (causing price to drop rapidly). Traders look for retests of these zones to enter trades."},
    {"q": "What are trendlines and how do I use them?", "a": "Trendlines are diagonal lines drawn across pivot highs or pivot lows to illustrate the prevailing direction of price. An uptrend line connects higher lows, acting as support, while a downtrend line connects lower highs, acting as resistance."},

    # SECTION 11: INDICATOR TRADING
    {"q": "What is a Moving Average (MA)?", "a": "A Moving Average is the average price over a given period. It's the most widely used indicator, displayed on top of your chart. SMA100 calculates the average close price for the last 100 bars."},
    {"q": "What is the simplest form of MA analysis?", "a": "Price above the MA → Look to buy. Price below the MA → Look to sell. This provides a simple trend-following strategy."},
    {"q": "What is a 'fast' Moving Average?", "a": "A fast MA uses a shorter period (e.g., 20 periods) and reacts more quickly to recent price changes, showing the short-term trend."},
    {"q": "What is a 'slow' Moving Average?", "a": "A slow MA uses a longer period (e.g., 100 periods) and responds more slowly, showing the longer-term trend."},
    {"q": "What is a Moving Average crossover?", "a": "A MA crossover occurs when a fast MA crosses above or below a slow MA. Fast MA above slow MA → Buy signal; Fast MA below slow MA → Sell signal."},
    {"q": "What is the Relative Strength Index (RSI)?", "a": "RSI is an oscillator that looks at an instrument's ability to close higher or lower over a given period. It appears below the chart in a separate window."},
    {"q": "What are the RSI levels for overbought and oversold?", "a": "RSI above 70 indicates 'overbought.' RSI below 30 indicates 'oversold.' However, securities can remain overbought/oversold for extended periods."},
    {"q": "What is RSI divergence?", "a": "Divergence occurs when price makes a new high or low but the indicator does NOT confirm it. Bullish divergence: price new lows, RSI higher lows. Bearish divergence: price new highs, RSI lower highs."},
    {"q": "What is bullish divergence?", "a": "Bullish divergence occurs when price moves to new lows while RSI carves a higher low, suggesting weakening downward momentum and a potential bottom."},
    {"q": "What is bearish divergence?", "a": "Bearish divergence occurs when price moves to new highs but RSI stops short of the previous peak, suggesting weakening upward momentum and a potential top."},

    # SECTION 12: TIMEFRAMES & MARKETS
    {"q": "What are timeframes in trading?", "a": "Timeframes represent the duration of a single price candle or bar on a chart. They range from Scalping charts (1-minute, 5-minute) and Day Trading charts (15-minute, 1-hour) to Swing and Position trading charts (4-hour, Daily, Weekly). Multiple timeframe analysis helps traders see both the bigger trend and optimal entries."},
    {"q": "What are market indexes?", "a": "Market indexes measure the price performance of a basket of shares from a stock exchange. Examples include the S&P 500, Nasdaq, and Dow Jones (US), the DAX (Germany), and the FTSE 100 (UK). Forex traders watch indexes to judge overall risk sentiment (Risk-On vs. Risk-Off)."},
    {"q": "What are commodities and how do they affect Forex?", "a": "Commodities are raw materials like Gold, Oil, Silver, and Natural Gas. They have strong ties to commodity currencies. For example, the Canadian Dollar (CAD) correlates heavily with crude oil prices, while the Australian Dollar (AUD) often moves closely with gold and iron ore exports."},

    # SECTION 13: TRADING PSYCHOLOGY
    {"q": "Why is psychology important in Forex trading?", "a": "The psychological difference between losing real money and demo money is significant. Fear of losing can cause you to lose; overconfidence is equally damaging. Successful traders must be 'cold as ice.'"},
    {"q": "What is the biggest obstacle traders must overcome?", "a": "Greed is probably the biggest obstacle. If you try to get rich on every trade, you'll likely blow your account. The paradox: if you want to get rich quickly, you must do it slowly."},
    {"q": "How do successful traders handle losses?", "a": "They don't get emotional. They trade according to their system, accept losses as unavoidable, lose pre-determined amounts they're comfortable with, move on to the next trade, and trust their system's long-term profitability."},
    {"q": "Why is 'slow and steady' better than trying to get rich quickly?", "a": "Risking 1% per trade with a 1:2 risk-reward ratio and 50% win rate creates consistent returns. A $5,000 account growing 5% weekly compounds to $63,214 in a year (1,260% return)."},
    {"q": "What is the psychology of losing in trading?", "a": "Losing is a part of trading. Successful traders don't get angry or sad - they risk pre-determined amounts they're comfortable with and accept losses as unavoidable. They move on to the next trade knowing their system works over the long term."},

    # SECTION 14: RISK MANAGEMENT & POSITION SIZING
    {"q": "What is risk management in forex?", "a": "Risk management involves protecting your account through position sizing, stop losses, leverage limits, and diversification. The golden rule: never risk more than 1-2% of your account on a single trade."},
    {"q": "How to trade with leverage safely?", "a": "Leverage amplifies both profits and losses. Safe leverage usage means: never use maximum available leverage, consider your account size, use stop losses, and understand that higher leverage = higher risk. Start with low leverage (1:10 or 1:20)."},
    {"q": "What is position sizing?", "a": "Position sizing determines how many units of currency to trade. It's calculated based on account size, risk per trade, and stop loss distance. Formula: Position Size = (Account Risk) / (Stop Loss in Pips x Pip Value)."},
    {"q": "How to place proper stop losses?", "a": "Stop losses should be placed at technical levels (support/resistance), based on volatility (ATR), or at logical price points. Don't set stops too tight (market noise) or too loose (excessive risk). Trail stops to protect profits."},
    {"q": "What is scaling in and out of trades?", "a": "Scaling in means entering a position gradually, scaling out means taking partial profits. This reduces risk and improves average entry/exit prices. Example: Enter 50% position at first signal, add 50% on confirmation."},
    {"q": "What are currency correlations?", "a": "Currency correlations measure how different currency pairs move in relation to each other. Positive correlation = pairs move together (EUR/USD and GBP/USD). Negative correlation = pairs move opposite (EUR/USD and USD/CHF). Monitor correlations to avoid stacking risk."},
    {"q": "How many positions should I have open at once?", "a": "You should never have more than two or three positions open at the same time. Even with 2% risk per trade, 10 simultaneous positions is a sure-fire way to get a margin call."},
    {"q": "Why is correlation important for risk management?", "a": "If you have positions in highly correlated markets, you're essentially risking more than your individual limits. Example: 2% on AUD/USD and 2% on NZD/USD means if USD surges, you lose 4%."},
    {"q": "Can I offset risk by taking opposite trades in correlated markets?", "a": "No. Going long AUD/USD and short NZD/USD means you still have long AUD and short NZD exposure. Correlated markets don't always move in lockstep."},
    {"q": "What is the golden rule of risk management?", "a": "Never risk more than 1% to 2% of your total account equity on any individual trade setup. 1% is more appropriate for beginners."},
    {"q": "How much risk do most professionals recommend per trade?", "a": "Most professionals recommend no more than 1-2% per trade. 5% is considered too much for the majority of strategies."},
    {"q": "What is drawdown?", "a": "Drawdown is the peak-to-trough decline during a specific period. The more an account draws down, the harder it is to build back up. A 50% loss requires a 100% gain to recover."},

    # SECTION 15: RISK-REWARD & WIN RATE
    {"q": "What is the risk-reward ratio?", "a": "Risk-reward ratio compares potential profit to potential loss. If stop loss is 10 pips and take profit is 20 pips, you have a 1:2 risk-reward ratio."},
    {"q": "How does the risk-reward ratio affect my breakeven point?", "a": "1:1 ratio = 50% win rate needed; 1:2 ratio = 33.33% win rate needed; 1:3 ratio = 25% win rate needed."},
    {"q": "What is the formula for expected value?", "a": "Expected Value = (Win Rate × Average Win) - (Loss Rate × Average Loss). A positive expected value means your system is profitable over time."},
    {"q": "What is the breakeven win rate formula?", "a": "Win Rate Needed = 1 / (1 + R:R Ratio). Example with 1:2 ratio: 1 / (1 + 2) = 1/3 = 33.33%."},
    {"q": "What is the position sizing formula?", "a": "Position Size = (Account Risk Amount) / (Stop Loss in Pips × Pip Value). Example: $100 risk / (50 pips × $1/pip) = 2 mini lots (0.2 standard lots)."},

    # SECTION 16: TRADING STRATEGIES
    {"q": "What are the main types of trading strategies?", "a": "The main strategies include: Trend Trading (following the direction), Breakout Trading (trading through levels), Reversal Trading (catching turns), and News Trading (trading economic releases)."},
    {"q": "What is the basic trend trading strategy?", "a": "Check if price is above MA → Look to buy; below → Look to sell. Use fast (20) and slow (100) MAs. Enter on crossover. Set stops above/below the slow MA."},
    {"q": "What is the basic breakout trading strategy?", "a": "Identify clear support and resistance. Wait for price to break through. Enter on the breakout. Set stop loss on the opposite side of the breakout level. Target the next support/resistance level."},
    {"q": "What is a 'false breakout' or 'fakeout'?", "a": "A false breakout occurs when price moves beyond a level but quickly reverses. Avoid by waiting for a close beyond the level (not just a spike) and trading during high liquidity sessions."},
    {"q": "What is the basic reversal trading strategy?", "a": "Identify the trend. Wait for a reversal pattern (Head and Shoulders, Double Top, Shooting Star). Enter after the pattern is confirmed. Place stop loss above the pattern. Target the previous support level."},
    {"q": "How can I confirm a reversal signal?", "a": "Confirmation methods include: break of the neckline (Head and Shoulders), break of the confirmation line (Double Top), RSI divergence, price closing beyond the reversal candle's high/low, and volume confirmation."},
    {"q": "What is news trading?", "a": "News trading involves trading the volatility created by economic data releases. Approaches include: Straddle strategy (orders both sides), Breakout strategy (follow the direction), Fade strategy (trade the reversal), and Reaction strategy (wait for volatility to settle)."},
    {"q": "What are the risks of news trading?", "a": "Risks include: extreme volatility and slippage, widening spreads, gapping (price jumps without trading), and being caught on the wrong side of a sudden move."},
    {"q": "How can I stay updated on economic data releases?", "a": "Use an economic calendar (like Vantage's Forex Economic Calendar) to filter releases by country and expected market impact, mark important releases, and manage risk around high-impact events."},

    # SECTION 17: BROKERS & ACCOUNTS
    {"q": "What is an ECN broker?", "a": "ECN (Electronic Communications Network) brokers provide direct access to other market participants. They don't trade against clients - they connect buyers and sellers and charge a commission."},
    {"q": "What is a Market Maker?", "a": "Market Makers create their own market for clients. They take the opposite side of client trades and profit from the spread. They may offer fixed spreads."},
    {"q": "What is a RAW ECN account?", "a": "A RAW ECN account offers the raw interbank spreads with a commission charged separately. Preferred by scalpers and high-volume traders for tight spreads and fast execution."},
    {"q": "What should I look for in a Forex broker?", "a": "Key factors: Regulation and security (FCA, ASIC), trading platform (MT4, MT5), spreads and commissions, leverage offered, minimum deposit, execution speed, customer service, educational resources, and product range."},
    {"q": "How do I test if a broker is right for me?", "a": "Open a demo account to test the platform and execution, test customer service responsiveness, check deposit and withdrawal processes, verify regulation, test spreads during your trading hours, and start with a small live account."},

    # SECTION 18: TRADING PLAN & SYSTEMS
    {"q": "How do I develop a trading plan?", "a": "Include: trading goals (specific, measurable, realistic), risk tolerance thresholds, specific entry rules (mechanical criteria), specific exit rules (take profit and stop loss levels), position sizing parameters, trading journal workflow, daily routine, and performance review schedule."},
    {"q": "How do I build a trading system?", "a": "Include: mechanical entry criteria, strict exit rules (take profit/stop loss), position sizing calculation variables, performance metrics, validation via forward and backtesting, and regular system review and optimization."},
    {"q": "What are the different types of traders?", "a": "Types include: Scalpers (seconds/minutes on 1m-5m charts), Day Traders (intraday on 5m-1h charts), Swing Traders (days/weeks on 1h-4h charts), Position Traders (weeks/months on Daily-Weekly charts), and Carry Traders (long-term)."},
    {"q": "Why is a trading journal important?", "a": "A trading journal tracks data parameters, execution variables, and psychological conditions to systematically review edge consistency and eliminate behavior errors. It helps identify patterns in winning and losing trades."},
    {"q": "What is the daily routine for a forex trader?", "a": "A typical routine includes: checking the economic calendar, reviewing overnight movements, analyzing higher timeframe charts, identifying key levels, monitoring open positions, executing trades according to the plan, and journaling all trades."},

    # SECTION 19: PROP TRADING & SCAMS
    {"q": "What are prop trading firms?", "a": "Prop trading firms provide capital to traders in exchange for a share of profits. They often require passing evaluation challenges. Benefits include larger capital access, professional infrastructure, and community support."},
    {"q": "How to avoid forex trading scams?", "a": "Avoid: guaranteed profit promises, unregulated brokers, signal sellers with unrealistic claims, and firms asking for large upfront fees. Always check regulation (FCA, CFTC, ASIC), read reviews, and start with small deposits."},
    {"q": "What are common trading mistakes?", "a": "Common mistakes: overtrading, revenge trading, no stop losses, using too much leverage, ignoring economic calendars, trading without a plan, emotional trading, and not keeping a journal."},
    {"q": "What is the most common mistake beginners make?", "a": "The most common mistakes include: over-leveraging, risking too much per trade, not using stop losses, revenge trading after losses, over-trading, and neglecting fundamental analysis events."},

    # SECTION 20: KEY TAKEAWAYS & GRADUATION
    {"q": "What is the key takeaway about chart patterns?", "a": "Chart patterns are often high probability, high reward trades that offer clear entry and stop loss levels. Patterns are confirmed when the relevant line breaks - wait for the breakout."},
    {"q": "What is 'Risk-On' and 'Risk-Off' sentiment?", "a": "Risk-On: investors are confident and seeking higher returns, benefiting higher-yielding currencies (AUD, NZD, CAD). Risk-Off: investors are fearful and seeking safety, benefiting safe havens (USD, JPY, CHF)."},
    {"q": "What are the types of Forex analysis?", "a": "The three main types are: Fundamental Analysis (economic factors), Technical Analysis (price action and indicators), and Sentiment Analysis (market psychology and positioning)."},
    {"q": "Any final graduation advice?", "a": "Successful trading requires discipline, continuous learning, and risk management. Start small, be patient, learn from losses, and never stop educating yourself. The market is always changing, so adapt your strategies accordingly."}
]

# Build NLP Matrix
questions = [item["q"] for item in knowledge_base]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(questions)

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='FX Academy AI Tutor',
        year=datetime.now().year,
        history=[],
        raw_history=""
    )

@app.route('/ask', methods=['POST'])
def ask_tutor():
    """Handles the query submission and performs vector lookups."""
    user_input = request.form.get("user_input", "")
    historical_data = request.form.get("historical_data", "")
    
    query_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(scores)
    
    history = [line.split("|||") for line in historical_data.split("%%%") if line] if historical_data else []
    
    if scores[best_idx] > 0.25:
        bot_response = knowledge_base[best_idx]["a"]
        conf = f"{round(float(scores[best_idx]) * 100)}% Match"
    else:
        bot_response = "I couldn't locate a precise match. Please clarify if you are asking about Fibonacci, timeframes, support/resistance, or risk rules."
        conf = "Low Confidence"
        
    history.append([user_input, bot_response, conf])
    raw_history = "%%%".join([f"{h[0]}|||{h[1]}|||{h[2]}" for h in history])
    
    return render_template(
        'index.html',
        title='FX Academy AI Tutor',
        year=datetime.now().year,
        history=history,
        raw_history=raw_history
    )