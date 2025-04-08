import json
import jsonpickle
from typing import Dict, List, Any
from datamodel import  Listing, Observation, ProsperityEncoder, Symbol, OrderDepth, TradingState, Order, Trade

PRODUCTS = {
    "KELP": {"use_ma": True},
    "SQUID_INK": {"use_ma": True},
    "RAINFOREST_RESIN": {"use_ma": False, "fair_price": 10000}
}

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

## --------------- MAIN CODE --------------------------------- 
class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        state_position = state.position
        limit = 50
        conversions = 0

        if state.traderData:
            data = jsonpickle.decode(state.traderData)
        else:
            data = {"kelp_prices": [], "squid_ink_prices": []}

        new_trader_data = data.copy()

        for product in state.order_depths.keys():
            if product not in PRODUCTS:
                continue

            orders: list[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            position = state_position.get(product, 0)
            available_buy = limit - position
            available_sell = limit + position

            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            price_key = f"{product.lower()}_prices"
            price_history = data.get(price_key, [])
            price_history.append(mid_price)
            if len(price_history) > 50:
                price_history.pop(0)
            new_trader_data[price_key] = price_history

            if product == "SQUID_INK":
                # === Mean Reversion Strategy ===
                fair_price, buy_signal, sell_signal = self.mean_reversion_signal(price_history, std_multiplier=2.0)
                logger.print(f"{product} Reversion Fair: {fair_price:.2f}, Buy: {buy_signal}, Sell: {sell_signal}")

                if buy_signal and available_buy > 0:
                    qty = min(10, available_buy)
                    logger.print("BUY", str(qty) + "x", mid_price)
                    orders.append(Order(product, int(mid_price), qty))
                    available_buy -= qty

                elif sell_signal and available_sell > 0:
                    qty = min(10, available_sell)
                    logger.print("SELL", str(qty) + "x", mid_price)
                    orders.append(Order(product, int(mid_price), -qty))
                    available_sell -= qty

            elif product == "KELP":
                # === Moving Average Strategy ===
                fair_price, trending_up, trending_down = self.get_moving_average(price_history)
                logger.print(f"{product} Trend: {'UP' if trending_up else 'DOWN' if trending_down else 'FLAT'}")

                for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
                    if ask_price < fair_price and trending_up and available_buy > 0:
                        qty = min(-ask_qty, available_buy)
                        logger.print("BUY", str(qty) + "x", ask_price)
                        orders.append(Order(product, ask_price, qty))
                        available_buy -= qty

                for bid_price, bid_qty in sorted(order_depth.buy_orders.items()):
                    if bid_price > fair_price and trending_down and available_sell > 0:
                        qty = min(bid_qty, available_sell)
                        logger.print("SELL", str(qty) + "x", bid_price)
                        orders.append(Order(product, bid_price, -qty))
                        available_sell -= qty

            elif product == "RAINFOREST_RESIN":

                buy_price = 9998
                sell_price = 10002
                orders = []

                # Place a passive BUY limit order at 9999
                if available_buy > 0:
                    qty = min(5, available_buy)  # You can change 5 to any order size you prefer
                    logger.print("Placing passive BUY at", buy_price, "for", qty)
                    orders.append(Order(product, buy_price, qty))

                # Place a passive SELL limit order at 10001
                if available_sell > 0:
                    qty = min(5, available_sell)
                    logger.print("Placing passive SELL at", sell_price, "for", qty)
                    orders.append(Order(product, sell_price, -qty))
                    

            result[product] = orders

        trader_data = jsonpickle.encode(new_trader_data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    
    def get_moving_average(self, price_history: List[float], fast_window: int = 5, slow_window: int = 20):
        if len(price_history) >= slow_window:
            fast_ma = sum(price_history[-fast_window:]) / fast_window
            slow_ma = sum(price_history[-slow_window:]) / slow_window
            fair_price = slow_ma
            return fair_price, fast_ma > slow_ma, fast_ma < slow_ma
        else:
            return price_history[-1], False, False
    
    def mean_reversion_signal(self, price_history: List[float], std_multiplier: float = 2.0):
        if len(price_history) < 30:
            return price_history[-1], False, False  # fallback

        window = 30
        prices = price_history[-window:]
        mean_price = sum(prices) / window
        std_dev = (sum((p - mean_price) ** 2 for p in prices) / window) ** 0.5

        upper_band = mean_price + std_multiplier * std_dev
        lower_band = mean_price - std_multiplier * std_dev

        current_price = prices[-1]
        buy_signal = current_price < lower_band
        sell_signal = current_price > upper_band

        return mean_price, buy_signal, sell_signal



