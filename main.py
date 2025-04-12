from datamodel import OrderDepth, TradingState, Order, Symbol,Listing, Trade, Observation, ProsperityEncoder
from typing import List
import json
import jsonpickle
import math
import numpy as np

"""
Market Make around 10_000 for RESIN,
Linear Regression Model on KELP and INK
"""

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
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

    def compress_state(self, state: TradingState, trader_data: str) -> list:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list]:
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

    def compress_observations(self, observations: Observation) -> list:
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

# === PRODUCT ENUM CLASS ===
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET_1 = "PICNIC_BASKET1"
    PICNIC_BASKET_2 = "PICNIC_BASKET2"


PARAMS = {
    # Market making strategy for resin, anchored around a static fair value of 10,000
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000, # static fair price assumption
        "take_width": 1, # minimum spread width considered for taking orders
        "clear_width": 0, # range beyond fair value where position-clearing occurs
        "disregard_edge": 1, # Minimum spread outside fair value to ignore certain quotes
        "join_edge": 2, # Range within which to join existing quotes
        "default_edge": 1, # Default price offset when placing market making quotes
        "soft_position_limit": 50,  # Position threshold for skewing quotes
    },
    # Strategy for KELP includes reversion-based fair value modeling and SQUID_INK influence
    Product.KELP: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,  # 20 - Avoid orders with large size
        "reversion_beta": -0.18,  # -0.2184 --> Beta for return based reversion (linear regression model)
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "ink_adjustment_factor": 0.05, # Sensitivity to changes in SQUID_INK price
    },
    # Strategy for SQUID_INK includes volatility spike detection and adaptive reversion
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 1,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.228, # Starting reversion beta
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "spike_lb": 3, # Lower bound: how far it must recover before exiting spike mode
        "spike_ub": 5.6, # Upper bound: how large a price jump must be to trigger spike logic
        "offset": 2, # Offset used when calculating recovery value post-spike
        "reversion_window": 55,  # Window size for adaptive reversion regression
        "reversion_weight": 0.12, # Weighted average for dynamic beta calculation
    },
    Product.DJEMBES: {
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "soft_position_limit": 50
    },
    Product.CROISSANTS: {
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "soft_position_limit": 50
    },
    Product.JAMS: {
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "soft_position_limit": 50
    },
    Product.PICNIC_BASKET_1: {
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "components": {Product.DJEMBES: 1, Product.CROISSANTS: 6, Product.JAMS: 3},
        "soft_position_limit": 50
    },
    Product.PICNIC_BASKET_2: {
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "components": {Product.CROISSANTS: 4, Product.JAMS: 2},
        "soft_position_limit": 50
    }
    

}

# == Trader Class === 
class Trader:
    def __init__(self, params=None):
        # Testing purposes
        if params is None:
            params = PARAMS

        self.params = params
        # Set maximum position limits per product
        self.PRODUCT_LIMIT = {Product.RAINFOREST_RESIN: 50,
                              Product.KELP: 50,
                              Product.SQUID_INK: 50,
                              Product.DJEMBES: 50,
                              Product.CROISSANTS: 50,
                              Product.JAMS: 50,
                              Product.PICNIC_BASKET_1: 50,
                              Product.PICNIC_BASKET_2: 50
                              }

    def take_best_orders(self, product: str,
                         fair_value: str, take_width: float,
                         orders: List[Order], order_depth: OrderDepth,
                         position: int, buy_order_volume: int,
                         sell_order_volume: int,
                         prevent_adverse: bool = False, adverse_volume: int = 0, traderObject: dict = None
                         ):
        # Check if squid ink, if it is then check for jumps and bet to go away from jump. (Look for big diff)
        # Core function that takes favorable existing orders from the book
        # Buys if the best ask is well below fair_value, sells if best bid is above
        # Includes logic for spike detection (especially for SQUID_INK)

        position_limit = self.PRODUCT_LIMIT[product]

        if product == "SQUID_INK":
            if "currentSpike" not in traderObject:
                traderObject["currentSpike"] = False
            prev_price = traderObject["ink_last_price"]

            # if currently in a spike, monitor recovery --> checks if it has stabilized enough to say that a previously detected spike has ended
            if traderObject["currentSpike"]:
                if abs(fair_value - prev_price) < self.params[Product.SQUID_INK]["spike_lb"]:
                    traderObject["currentSpike"] = False
                else:
                    # If still in spike, act in opposite direction of spike to fade it
                    # Spike down -> buy, Spike up -> sell
                    if fair_value < traderObject["recoveryValue"]:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_amount = order_depth.sell_orders[best_ask]
                        quantity = max(best_ask_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.buy_orders[best_ask]
                        
                        # Optionally try second-best ask if position limit allows
                        if best_ask_amount > position + position_limit:
                            # Try second-best bid if leftover space
                            best_ask = max(list(filter(lambda x: x != best_ask, order_depth.sell_orders.keys())))
                            best_ask_amount = order_depth.buy_orders[best_ask]
                            quantity = max(best_ask_amount, position_limit + position)
                            if quantity > 0:
                                orders.append(Order(product, best_ask, quantity))
                                buy_order_volume += quantity
                                order_depth.sell_orders[best_ask] += quantity
                                if order_depth.sell_orders[best_ask] == 0:
                                    del order_depth.buy_orders[best_ask]
                        return buy_order_volume, 0 # we only buy
                    
                    else:
                        # If price spiked upward too fast, we sell aggressively
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        quantity = max(best_bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]
                        if best_bid_amount > position + position_limit:
                            # Try second-best bid if leftover space
                            best_bid = max(list(filter(lambda x: x != best_bid, order_depth.buy_orders.keys())))
                            best_bid_amount = order_depth.buy_orders[best_bid]
                            quantity = max(best_bid_amount, position_limit + position)
                            if quantity > 0:
                                orders.append(Order(product, best_bid, -1 * quantity))
                                sell_order_volume += quantity
                                order_depth.buy_orders[best_bid] -= quantity
                                if order_depth.buy_orders[best_bid] == 0:
                                    del order_depth.buy_orders[best_bid]
                        return 0, sell_order_volume # we only sell
                    
            if abs(fair_value - prev_price) > self.params[Product.SQUID_INK]["spike_ub"]:
                traderObject["currentSpike"] = True
                traderObject["recoveryValue"] = prev_price + self.params[Product.SQUID_INK][
                    "offset"] if fair_value > prev_price else prev_price - self.params[Product.SQUID_INK]["offset"]
                # Main spike
                if fair_value > prev_price:
                    # Spike up, so sell bids until capacity reached
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    quantity = max(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
                    if best_bid_amount > position + position_limit:
                        # Try second-best bid if leftover space
                        best_bid = max(list(filter(lambda x: x != best_bid, order_depth.buy_orders.keys())))
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        quantity = max(best_bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]
                    return 0, sell_order_volume
                else:
                    # Spike down, so buy asks until capacity reached
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    quantity = max(best_ask_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.buy_orders[best_ask]
                    if best_ask_amount > position + position_limit:
                        # Try second-best bid if leftover space
                        best_ask = max(list(filter(lambda x: x != best_ask, order_depth.sell_orders.keys())))
                        best_ask_amount = order_depth.buy_orders[best_ask]
                        quantity = max(best_ask_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.buy_orders[best_ask]
                    return buy_order_volume, 0

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(self, product: str,
                    orders: List[Order],
                    bid: int, ask: int, position: int,
                    buy_order_volume: int, sell_order_volume: int,
                    ):
        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, math.floor(bid), buy_quantity))  # Buy order

        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, math.ceil(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    # Unwinds the position if it deviates from neutral
    def clear_position_order(self, product: str,
                             fair_value: float,
                             width: int, orders: List[Order],
                             order_depth: OrderDepth,
                             position: int, buy_order_volume: int,
                             sell_order_volume: int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)

        # Checks if we are long for too long and finds for potential buyers who would buy at fair_value + width
        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # Checks if we are short for too long and finds for sellers at or below fair_value - width to buy from
        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject,
                        ink_order_depth: OrderDepth):
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            valid_ask = [price for price in order_depth.sell_orders.keys()
                         if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders.keys()
                         if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]

            mm_ask = min(valid_ask) if len(valid_ask) > 0 else None
            mm_bid = max(valid_buy) if len(valid_buy) > 0 else None
            if valid_ask and valid_buy:
                mmmid_price = (mm_ask + mm_bid) / 2

            else:
                if traderObject.get('kelp_last_price', None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject['kelp_last_price']

            if traderObject.get('kelp_last_price', None) is None:
                fair = mmmid_price
            else:
                ### Alpha-ish - LR forecast
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (last_returns * self.params[Product.KELP]["reversion_beta"])
                fair = mmmid_price + (mmmid_price * pred_returns)

            if traderObject.get("ink_last_price", None) is not None:
                ### Alpha - Neg Corr Ink
                old_ink_price = traderObject["ink_last_price"]
                valid_ask_ink = [price for price in ink_order_depth.sell_orders.keys()
                                 if abs(ink_order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK][
                                     "adverse_volume"]]
                valid_buy_ink = [price for price in ink_order_depth.buy_orders.keys()
                                 if abs(ink_order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK][
                                     "adverse_volume"]]
                if valid_ask_ink and valid_buy_ink:
                    new_ink_mid = (min(valid_ask_ink) + max(valid_buy_ink)) / 2
                else:
                    new_ink_mid = (min(ink_order_depth.sell_orders.keys()) +
                                   max(ink_order_depth.buy_orders.keys())) / 2

                ink_return = (new_ink_mid - old_ink_price) / old_ink_price
                fair = fair - (self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price)
                # ink_return = (traderObject["ink_last_price"] - traderObject["prev_ink_price"]) / traderObject["prev_ink_price"]
                # adj = self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price
                # fair -= adj

            # traderObject["prev_ink_price"] = traderObject.get("ink_last_price", mmmid_price)
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def ink_fair_value(self, order_depth: OrderDepth, traderObject):
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            valid_ask = [price for price in order_depth.sell_orders.keys()
                         if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders.keys()
                         if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]

            mm_ask = min(valid_ask) if len(valid_ask) > 0 else None
            mm_bid = max(valid_buy) if len(valid_buy) > 0 else None
            if valid_ask and valid_buy:
                mmmid_price = (mm_ask + mm_bid) / 2

            else:
                if traderObject.get('ink_last_price', None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject['ink_last_price']

            if traderObject.get('ink_price_history', None) is None:
                traderObject['ink_price_history'] = []

            traderObject['ink_price_history'].append(mmmid_price)
            if len(traderObject['ink_price_history']) > self.params[Product.SQUID_INK]["reversion_window"]:
                traderObject['ink_price_history'] = traderObject['ink_price_history'][
                                                    -self.params[Product.SQUID_INK]["reversion_window"]:]

            # New Alpha attempt: adaptive mean reversion
            if len(traderObject['ink_price_history']) >= self.params[Product.SQUID_INK]["reversion_window"]:
                prices = np.array(traderObject['ink_price_history'])

                returns = (prices[1:] - prices[:-1]) / prices[:-1]
                X = returns[:-1]
                Y = returns[1:]
                if np.dot(X, X) != 0:
                    estimated_beta = - np.dot(X, Y) / np.dot(X, X)
                else:
                    estimated_beta = self.params[Product.SQUID_INK]["reversion_beta"]

                adaptive_beta = (self.params[Product.SQUID_INK]['reversion_weight'] * estimated_beta
                                 + (1 - self.params[Product.SQUID_INK]['reversion_weight']) *
                                 self.params[Product.SQUID_INK]["reversion_beta"])
            else:
                adaptive_beta = self.params[Product.SQUID_INK]["reversion_beta"]

            if traderObject.get('ink_last_price', None) is None:
                fair = mmmid_price
            else:
                last_price = traderObject["ink_last_price"]
                last_return = (mmmid_price - last_price) / last_price
                pred_return = last_return * adaptive_beta
                fair = mmmid_price + (mmmid_price * pred_return)
            traderObject["ink_last_price"] = mmmid_price
            return fair
        return None

    def basket_fair_value(self, components: dict, order_depths: dict):
        fair_value = 0
        for product, quantity in components.items():
            if product in order_depths:
                comp_depth = order_depths[product]
                if comp_depth.buy_orders and comp_depth.sell_orders:
                    mid = (max(comp_depth.buy_orders) + min(comp_depth.sell_orders)) / 2
                    fair_value += quantity * mid
                else:
                    logger.print(f"[BASKET] Missing depth for {product}")
            else:
                logger.print(f"[BASKET] Component {product} not in order_depths")
        return fair_value

    def take_orders(self, product: str, order_depth: OrderDepth,
                    fair_value: float, take_width: float,
                    position: int, prevent_adverse: bool = False,
                    adverse_volume: int = 0, traderObject: dict = None):
        orders: List[Order] = []

        buy_order_volume, sell_order_volume = 0, 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth,
            position, buy_order_volume, sell_order_volume, prevent_adverse,
            adverse_volume, traderObject
        )

        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth,
                     fair_value: float, clear_width: int,
                     position: int, buy_order_volume: int,
                     sell_order_volume: int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth,
            position, buy_order_volume, sell_order_volume
        )

        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth: OrderDepth, fair_value: float,
                    position: int, buy_order_volume: int, sell_order_volume: int,
                    disregard_edge: float, join_edge: float, default_edge: float,
                    manage_position: bool = False, soft_position_limit: int = 0,
                    cur_resin_price: float = None):

        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair + 1  # join
            else:
                ask = best_ask_above_fair  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # --- RAINFOREST_RESIN ---
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (state.position[Product.RAINFOREST_RESIN]
                              if Product.RAINFOREST_RESIN in state.position
                              else 0)
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(Product.RAINFOREST_RESIN,
                                 state.order_depths[Product.RAINFOREST_RESIN],
                                 self.params[Product.RAINFOREST_RESIN]['fair_value'],
                                 self.params[Product.RAINFOREST_RESIN]['take_width'],
                                 resin_position, )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(Product.RAINFOREST_RESIN,
                                  state.order_depths[Product.RAINFOREST_RESIN],
                                  self.params[Product.RAINFOREST_RESIN]['fair_value'],
                                  self.params[Product.RAINFOREST_RESIN]['clear_width'],
                                  resin_position,
                                  buy_order_volume,
                                  sell_order_volume,
                                  )
            )
            resin_make_orders, _, _ = self.make_orders(Product.RAINFOREST_RESIN,
                                                       state.order_depths[Product.RAINFOREST_RESIN],
                                                       self.params[Product.RAINFOREST_RESIN]['fair_value'],
                                                       resin_position,
                                                       buy_order_volume,
                                                       sell_order_volume,
                                                       self.params[Product.RAINFOREST_RESIN]['disregard_edge'],
                                                       self.params[Product.RAINFOREST_RESIN]['join_edge'],
                                                       self.params[Product.RAINFOREST_RESIN]['default_edge'],
                                                       True,
                                                       self.params[Product.RAINFOREST_RESIN]['soft_position_limit'],
                                                       )

            result[Product.RAINFOREST_RESIN] = (
                    resin_take_orders + resin_clear_orders + resin_make_orders
            )

        # --- KELP ---
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (state.position[Product.KELP]
                             if Product.KELP in state.position
                             else 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject,
                                                   state.order_depths[Product.SQUID_INK])
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(Product.KELP,
                                 state.order_depths[Product.KELP],
                                 kelp_fair_value,
                                 self.params[Product.KELP]['take_width'],
                                 kelp_position,
                                 self.params[Product.KELP]['prevent_adverse'],
                                 self.params[Product.KELP]['adverse_volume'],
                                 traderObject)
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(Product.KELP,
                                  state.order_depths[Product.KELP],
                                  kelp_fair_value,
                                  self.params[Product.KELP]['clear_width'],
                                  kelp_position,
                                  buy_order_volume,
                                  sell_order_volume, )
            )
            kelp_make_orders, _, _ = self.make_orders(Product.KELP,
                                                      state.order_depths[Product.KELP],
                                                      kelp_fair_value,
                                                      kelp_position,
                                                      buy_order_volume,
                                                      sell_order_volume,
                                                      self.params[Product.KELP]['disregard_edge'],
                                                      self.params[Product.KELP]['join_edge'],
                                                      self.params[Product.KELP]['default_edge'],
                                                      )

            result[Product.KELP] = (
                    kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        # --- SQUID_INK ---
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = (state.position[Product.SQUID_INK]
                            if Product.SQUID_INK in state.position
                            else 0)
            ink_fair_value = self.ink_fair_value(state.order_depths[Product.SQUID_INK], traderObject)
            ink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(Product.SQUID_INK,
                                 state.order_depths[Product.SQUID_INK],
                                 ink_fair_value,
                                 self.params[Product.SQUID_INK]['take_width'],
                                 ink_position,
                                 self.params[Product.SQUID_INK]['prevent_adverse'],
                                 self.params[Product.SQUID_INK]['adverse_volume'],
                                 traderObject)
            )
            ink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(Product.SQUID_INK,
                                  state.order_depths[Product.SQUID_INK],
                                  ink_fair_value,
                                  self.params[Product.SQUID_INK]['clear_width'],
                                  ink_position,
                                  buy_order_volume,
                                  sell_order_volume, )
            )
            ink_make_orders, _, _ = self.make_orders(Product.SQUID_INK,
                                                     state.order_depths[Product.SQUID_INK],
                                                     ink_fair_value,
                                                     ink_position,
                                                     buy_order_volume,
                                                     sell_order_volume,
                                                     self.params[Product.SQUID_INK]['disregard_edge'],
                                                     self.params[Product.SQUID_INK]['join_edge'],
                                                     self.params[Product.SQUID_INK]['default_edge'],
                                                     )

            result[Product.SQUID_INK] = (
                    ink_take_orders + ink_clear_orders + ink_make_orders
            )

        # --- CROISSANTS ---
        if Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths:
            cro_position = state.position.get(Product.CROISSANTS, 0)
            cro_depth = state.order_depths[Product.CROISSANTS]
            cro_fair = (min(cro_depth.sell_orders) + max(cro_depth.buy_orders)) / 2 if cro_depth.sell_orders and cro_depth.buy_orders else 0
            cro_take, buy_vol, sell_vol = self.take_orders(Product.CROISSANTS, cro_depth, cro_fair, self.params[Product.CROISSANTS]['take_width'], cro_position)
            cro_clear, buy_vol, sell_vol = self.clear_orders(Product.CROISSANTS, cro_depth, cro_fair, self.params[Product.CROISSANTS]['clear_width'], cro_position, buy_vol, sell_vol)
            cro_make, _, _ = self.make_orders(Product.CROISSANTS, cro_depth, cro_fair, cro_position, buy_vol, sell_vol, self.params[Product.CROISSANTS]['disregard_edge'], self.params[Product.CROISSANTS]['join_edge'], self.params[Product.CROISSANTS]['default_edge'], True, self.params[Product.CROISSANTS]['soft_position_limit'])
            result[Product.CROISSANTS] = cro_take + cro_clear + cro_make

        # --- JAMS ---
        if Product.JAMS in self.params and Product.JAMS in state.order_depths:
            jam_position = state.position.get(Product.JAMS, 0)
            jam_depth = state.order_depths[Product.JAMS]
            jam_fair = (min(jam_depth.sell_orders) + max(jam_depth.buy_orders)) / 2 if jam_depth.sell_orders and jam_depth.buy_orders else 0
            jam_take, buy_vol, sell_vol = self.take_orders(Product.JAMS, jam_depth, jam_fair, self.params[Product.JAMS]['take_width'], jam_position)
            jam_clear, buy_vol, sell_vol = self.clear_orders(Product.JAMS, jam_depth, jam_fair, self.params[Product.JAMS]['clear_width'], jam_position, buy_vol, sell_vol)
            jam_make, _, _ = self.make_orders(Product.JAMS, jam_depth, jam_fair, jam_position, buy_vol, sell_vol, self.params[Product.JAMS]['disregard_edge'], self.params[Product.JAMS]['join_edge'], self.params[Product.JAMS]['default_edge'], True, self.params[Product.JAMS]['soft_position_limit'])
            result[Product.JAMS] = jam_take + jam_clear + jam_make

        # --- DJEMBES ---
        if Product.DJEMBES in self.params and Product.DJEMBES in state.order_depths:
            dje_position = state.position.get(Product.DJEMBES, 0)
            dje_depth = state.order_depths[Product.DJEMBES]
            dje_fair = (min(dje_depth.sell_orders) + max(dje_depth.buy_orders)) / 2 if dje_depth.sell_orders and dje_depth.buy_orders else 0
            dje_take, buy_vol, sell_vol = self.take_orders(Product.DJEMBES, dje_depth, dje_fair, self.params[Product.DJEMBES]['take_width'], dje_position)
            dje_clear, buy_vol, sell_vol = self.clear_orders(Product.DJEMBES, dje_depth, dje_fair, self.params[Product.DJEMBES]['clear_width'], dje_position, buy_vol, sell_vol)
            dje_make, _, _ = self.make_orders(Product.DJEMBES, dje_depth, dje_fair, dje_position, buy_vol, sell_vol, self.params[Product.DJEMBES]['disregard_edge'], self.params[Product.DJEMBES]['join_edge'], self.params[Product.DJEMBES]['default_edge'], True, self.params[Product.DJEMBES]['soft_position_limit'])
            result[Product.DJEMBES] = dje_take + dje_clear + dje_make

        # --- PICNIC BASKET 1 ---
        if Product.PICNIC_BASKET_1 in self.params and Product.PICNIC_BASKET_1 in state.order_depths:
            pb1_position = state.position.get(Product.PICNIC_BASKET_1, 0)
            pb1_depth = state.order_depths[Product.PICNIC_BASKET_1]
            pb1_fair = self.basket_fair_value(self.params[Product.PICNIC_BASKET_1]['components'], state.order_depths)
            pb1_take, buy_vol, sell_vol = self.take_orders(Product.PICNIC_BASKET_1, pb1_depth, pb1_fair, 1, pb1_position)
            pb1_clear, buy_vol, sell_vol = self.clear_orders(Product.PICNIC_BASKET_1, pb1_depth, pb1_fair, 0, pb1_position, buy_vol, sell_vol)
            pb1_make, _, _ = self.make_orders(Product.PICNIC_BASKET_1, pb1_depth, pb1_fair, pb1_position, buy_vol, sell_vol, self.params[Product.PICNIC_BASKET_1]['disregard_edge'], self.params[Product.PICNIC_BASKET_1]['join_edge'], self.params[Product.PICNIC_BASKET_1]['default_edge'], True, self.params[Product.PICNIC_BASKET_1]['soft_position_limit'])
            result[Product.PICNIC_BASKET_1] = pb1_take + pb1_clear + pb1_make

        # --- PICNIC BASKET 2 ---
        if Product.PICNIC_BASKET_2 in self.params and Product.PICNIC_BASKET_2 in state.order_depths:
            pb2_position = state.position.get(Product.PICNIC_BASKET_2, 0)
            pb2_depth = state.order_depths[Product.PICNIC_BASKET_2]
            pb2_fair = self.basket_fair_value(self.params[Product.PICNIC_BASKET_2]['components'], state.order_depths)
            pb2_take, buy_vol, sell_vol = self.take_orders(Product.PICNIC_BASKET_2, pb2_depth, pb2_fair, 1, pb2_position)
            pb2_clear, buy_vol, sell_vol = self.clear_orders(Product.PICNIC_BASKET_2, pb2_depth, pb2_fair, 0, pb2_position, buy_vol, sell_vol)
            pb2_make, _, _ = self.make_orders(Product.PICNIC_BASKET_2, pb2_depth, pb2_fair, pb2_position, buy_vol, sell_vol, self.params[Product.PICNIC_BASKET_2]['disregard_edge'], self.params[Product.PICNIC_BASKET_2]['join_edge'], self.params[Product.PICNIC_BASKET_2]['default_edge'], True, self.params[Product.PICNIC_BASKET_2]['soft_position_limit'])
            result[Product.PICNIC_BASKET_2] = pb2_take + pb2_clear + pb2_make

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData