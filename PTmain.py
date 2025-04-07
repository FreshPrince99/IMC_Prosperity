import json
import jsonpickle
import statistics
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        # Load previous trader data if available; otherwise, initialize our store.
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {
                "price_history": {},       
                "macd": {},                
                "consolidation": {},       
                "daily_history": {},       
                "last_day": None          
            }

        current_day = state.timestamp

        # If a new day has started, update the daily history.
        if data.get("last_day") is None or data["last_day"] != current_day:
            for product, prices in data["price_history"].items():
                if prices:
                    daily_close = prices[-1]
                    daily_hist = data["daily_history"].get(product, [])
                    daily_hist.append(daily_close)
                    if len(daily_hist) > 3:
                        daily_hist.pop(0)
                    data["daily_history"][product] = daily_hist
            data["last_day"] = current_day

        # Process each product in the order depths.
        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[product]

            if order_depth.buy_orders and order_depth.sell_orders:
                highest_bid = max(order_depth.buy_orders.keys())
                lowest_ask = min(order_depth.sell_orders.keys())
                mid_price = (highest_bid + lowest_ask) / 2.0
            else:
                continue

            orders: List[Order] = []

            # Update intraday price history.
            price_history = data["price_history"].get(product, [])
            price_history.append(mid_price)
            if len(price_history) > 50:
                price_history.pop(0)
            data["price_history"][product] = price_history

            # --- Strategy for Squid ink and Kelp: MACD with Daily Trend and RSI ---
            if product in ["Squid ink", "Kelp"]:
                short_period = 10
                long_period = 30
                signal_period = 8

                multiplier_short = 2 / (short_period + 1)
                multiplier_long = 2 / (long_period + 1)
                multiplier_signal = 2 / (signal_period + 1)

                macd_data = data["macd"].get(product, {
                    "ema_short": mid_price,
                    "ema_long": mid_price,
                    "signal": 0,
                })

                ema_short = mid_price * multiplier_short + macd_data["ema_short"] * (1 - multiplier_short)
                ema_long = mid_price * multiplier_long + macd_data["ema_long"] * (1 - multiplier_long)
                macd_value = ema_short - ema_long

                signal_line = macd_value * multiplier_signal + macd_data["signal"] * (1 - multiplier_signal)

                data["macd"][product] = {
                    "ema_short": ema_short,
                    "ema_long": ema_long,
                    "signal": signal_line,
                }

                daily_hist = data["daily_history"].get(product, [])
                daily_trend = 0
                if len(daily_hist) >= 2:
                    daily_trend = daily_hist[-1] - daily_hist[-2]

                # Calculate RSI if we have enough data, else default to 50.
                def calculate_rsi(prices, period=14):
                    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                    gains = [delta if delta > 0 else 0 for delta in deltas]
                    losses = [-delta if delta < 0 else 0 for delta in deltas]
                    
                    if len(prices) < period + 1:
                        return None

                    avg_gain = sum(gains[:period]) / period
                    avg_loss = sum(losses[:period]) / period
                    
                    rsi_values = []
                    for i in range(period, len(prices)):
                        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
                        
                        rs = avg_gain / avg_loss if avg_loss != 0 else 0
                        rsi = 100 - (100 / (1 + rs))
                        rsi_values.append(rsi)
                    
                    return rsi_values

                current_rsi = 50  # Default RSI value
                rsi_values = calculate_rsi(price_history)
                if rsi_values:
                    current_rsi = rsi_values[-1]
                logger.print(f"{product}: RSI={current_rsi:.2f}")

                logger.print(f"{product}: mid={mid_price:.2f}, EMA_short={ema_short:.2f}, EMA_long={ema_long:.2f}, MACD={macd_value:.2f}, Signal={signal_line:.2f}, DailyTrend={daily_trend:.2f}")

                order_size = 5
                if abs(daily_trend) > 0.005 * mid_price:
                    order_size = 10

                # Adjusted conditions: trade if MACD crosses signal AND RSI is in a neutral zone.
                if macd_value > signal_line and current_rsi < 70:
                    logger.print(f"{product}: Bullish MACD signal, buying {order_size} units")
                    orders.append(Order(product, int(mid_price), order_size))
                elif macd_value < signal_line and current_rsi > 30:
                    logger.print(f"{product}: Bearish MACD signal, selling {order_size} units")
                    orders.append(Order(product, int(mid_price), -order_size))

            # --- Strategy for Rainforest resin: Breakout with Volatility Adjustment ---
            elif product == "Rainforest resin":
                volatility = statistics.stdev(price_history) if len(price_history) > 1 else 0
                base_threshold = 0.01 * mid_price
                threshold = max(base_threshold, volatility)

                consolidation = data["consolidation"].get(product, {"min": mid_price, "max": mid_price, "settled": False})

                if consolidation["max"] - consolidation["min"] < threshold:
                    consolidation["settled"] = True
                    consolidation["min"] = min(consolidation["min"], mid_price)
                    consolidation["max"] = max(consolidation["max"], mid_price)
                else:
                    consolidation = {"min": mid_price, "max": mid_price, "settled": False}
                data["consolidation"][product] = consolidation

                breakout = None
                if consolidation["settled"]:
                    if mid_price > consolidation["max"] + threshold:
                        breakout = "up"
                    elif mid_price < consolidation["min"] - threshold:
                        breakout = "down"

                logger.print(f"{product}: mid={mid_price:.2f}, Consolidation={consolidation}, Volatility={volatility:.4f}, Threshold={threshold:.4f}")
                if breakout == "up":
                    logger.print(f"{product}: Breakout up detected at {mid_price:.2f}")
                    orders.append(Order(product, int(mid_price), 10))
                elif breakout == "down":
                    logger.print(f"{product}: Breakout down detected at {mid_price:.2f}")
                    orders.append(Order(product, int(mid_price), -10))
                else:
                    logger.print(f"{product}: No breakout, maintaining position.")

            result[product] = orders

        trader_data = jsonpickle.encode(data)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data


# Optional main entry point for testing
if __name__ == "__main__":
    trader = Trader()
    print("Trader initialized. Please integrate with your trading environment.")