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

# == Trader Class === 
# ==================== PRODUCT ENUM ====================
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET_1 = "PICNIC_BASKET1"
    PICNIC_BASKET_2 = "PICNIC_BASKET2"

# ==================== PARAMETERS ====================
# For the enhanced products we lower thresholds to capture narrow spreads.
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
        "soft_position_limit": 50,
    },
    Product.KELP: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.18,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "ink_adjustment_factor": 0.05,
    },
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 1,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.228,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "spike_lb": 3,
        "spike_ub": 5.6,
        "offset": 2,
        "reversion_window": 55,
        "reversion_weight": 0.12,
    },
    # Enhanced strategy parameters for these products with tighter thresholds:
    Product.DJEMBES: {
        "take_width": 0.5,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 0.5,
        "default_edge": 0.5,
        "soft_position_limit": 50,
    },
    Product.CROISSANTS: {
        "take_width": 0.5,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 0.5,
        "default_edge": 0.5,
        "soft_position_limit": 240,
    },
    Product.JAMS: {
        "take_width": 0.5,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 0.5,
        "default_edge": 0.5,
        "soft_position_limit": 340,
    },
    Product.PICNIC_BASKET_1: {
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "components": {Product.DJEMBES: 1, Product.CROISSANTS: 6, Product.JAMS: 3},
        "soft_position_limit": 55,
        "take_width": 0.5,
        "clear_width": 0,
    },
    Product.PICNIC_BASKET_2: {
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 1,
        "components": {Product.CROISSANTS: 4, Product.JAMS: 2},
        "soft_position_limit": 90,
        "take_width": 0.5,
        "clear_width": 0,
    }
}

# ==================== TRADER CLASS ====================
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.PRODUCT_LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.DJEMBES: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.PICNIC_BASKET_1: 60,
            Product.PICNIC_BASKET_2: 100,
        }

    # ------------- ORIGINAL FUNCTIONS FOR RAINFOREST_RESIN, KELP, SQUID_INK -------------
    def take_best_orders(self, product: str, fair_value: float, take_width: float,
                         orders: List[Order], order_depth: OrderDepth,
                         position: int, buy_order_volume: int,
                         sell_order_volume: int, prevent_adverse: bool = False,
                         adverse_volume: int = 0, traderObject: dict = None):
        position_limit = self.PRODUCT_LIMIT[product]
        if product == "SQUID_INK":
            if "currentSpike" not in traderObject:
                traderObject["currentSpike"] = False
            prev_price = traderObject.get("ink_last_price", fair_value)
            if traderObject["currentSpike"]:
                if abs(fair_value - prev_price) < self.params[Product.SQUID_INK]["spike_lb"]:
                    traderObject["currentSpike"] = False
                else:
                    if fair_value < traderObject["recoveryValue"]:
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_amount = order_depth.sell_orders[best_ask]
                        quantity = max(best_ask_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                        return buy_order_volume, 0
                    else:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_amount = order_depth.buy_orders[best_bid]
                        quantity = max(best_bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -quantity))
                            sell_order_volume += quantity
                        return 0, sell_order_volume
            if abs(fair_value - prev_price) > self.params[Product.SQUID_INK]["spike_ub"]:
                traderObject["currentSpike"] = True
                traderObject["recoveryValue"] = (prev_price + self.params[Product.SQUID_INK]["offset"]
                                                 if fair_value > prev_price else prev_price - self.params[Product.SQUID_INK]["offset"])
                if fair_value > prev_price:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    quantity = max(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                    return 0, sell_order_volume
                else:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    quantity = max(best_ask_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                    return buy_order_volume, 0
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if (not prevent_adverse) or (abs(best_ask_amount) <= adverse_volume):
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (not prevent_adverse) or (abs(best_bid_amount) <= adverse_volume):
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order],
                    bid: int, ask: int, position: int,
                    buy_order_volume: int, sell_order_volume: int):
        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, math.floor(bid), buy_quantity))
        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, math.ceil(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int,
                             orders: List[Order], order_depth: OrderDepth,
                             position: int, buy_order_volume: int,
                             sell_order_volume: int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.PRODUCT_LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.PRODUCT_LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

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
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair + 1
            else:
                ask = best_ask_above_fair
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position,
            buy_order_volume, sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject, ink_order_depth: OrderDepth):
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            valid_ask = [price for price in order_depth.sell_orders if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            if valid_ask and valid_buy:
                mmmid_price = (min(valid_ask) + max(valid_buy)) / 2
            else:
                mmmid_price = traderObject.get('kelp_last_price', (best_ask + best_bid) / 2)
            if 'kelp_last_price' not in traderObject:
                fair = mmmid_price
            else:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            if traderObject.get("ink_last_price") is not None:
                old_ink_price = traderObject["ink_last_price"]
                valid_ask_ink = [price for price in ink_order_depth.sell_orders if abs(ink_order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
                valid_buy_ink = [price for price in ink_order_depth.buy_orders if abs(ink_order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
                if valid_ask_ink and valid_buy_ink:
                    new_ink_mid = (min(valid_ask_ink) + max(valid_buy_ink)) / 2
                else:
                    new_ink_mid = (min(ink_order_depth.sell_orders) + max(ink_order_depth.buy_orders)) / 2
                ink_return = (new_ink_mid - old_ink_price) / old_ink_price
                fair = fair - (self.params[Product.KELP].get("ink_adjustment_factor", 0.5) * ink_return * mmmid_price)
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def ink_fair_value(self, order_depth: OrderDepth, traderObject):
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            valid_ask = [price for price in order_depth.sell_orders if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            valid_buy = [price for price in order_depth.buy_orders if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            if valid_ask and valid_buy:
                mmmid_price = (min(valid_ask) + max(valid_buy)) / 2
            else:
                mmmid_price = traderObject.get('ink_last_price', (best_ask + best_bid) / 2)
            if 'ink_price_history' not in traderObject:
                traderObject['ink_price_history'] = []
            traderObject['ink_price_history'].append(mmmid_price)
            if len(traderObject['ink_price_history']) > self.params[Product.SQUID_INK]["reversion_window"]:
                traderObject['ink_price_history'] = traderObject['ink_price_history'][-self.params[Product.SQUID_INK]["reversion_window"]:]
            if len(traderObject['ink_price_history']) >= self.params[Product.SQUID_INK]["reversion_window"]:
                prices = np.array(traderObject['ink_price_history'])
                returns = (prices[1:] - prices[:-1]) / prices[:-1]
                X = returns[:-1]
                Y = returns[1:]
                if np.dot(X, X) != 0:
                    estimated_beta = - np.dot(X, Y) / np.dot(X, X)
                else:
                    estimated_beta = self.params[Product.SQUID_INK]["reversion_beta"]
                adaptive_beta = (self.params[Product.SQUID_INK]['reversion_weight'] * estimated_beta +
                                 (1 - self.params[Product.SQUID_INK]['reversion_weight']) * self.params[Product.SQUID_INK]["reversion_beta"])
            else:
                adaptive_beta = self.params[Product.SQUID_INK]["reversion_beta"]
            if 'ink_last_price' not in traderObject:
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
        # Original basket fair value: simple weighted sum using simple mid-price.
        fair_value = 0
        for product, quantity in components.items():
            if product in order_depths:
                comp_depth = order_depths[product]
                if comp_depth.buy_orders and comp_depth.sell_orders:
                    mid = (max(comp_depth.buy_orders) + min(comp_depth.sell_orders)) / 2
                    fair_value += quantity * mid
                else:
                    fair_value += 0
            else:
                pass  # Missing component; could add logging here.
        return fair_value

    # ------------- ENHANCED FUNCTIONS (with rolling fair value) -------------
    def compute_weighted_mid_price(self, order_depth: OrderDepth):
        if order_depth.buy_orders and order_depth.sell_orders:
            total_buy = sum(order_depth.buy_orders.values())
            total_sell = sum(abs(v) for v in order_depth.sell_orders.values())
            weighted_bid = sum(price * volume for price, volume in order_depth.buy_orders.items()) / total_buy
            weighted_ask = sum(price * abs(volume) for price, volume in order_depth.sell_orders.items()) / total_sell
            return (weighted_bid + weighted_ask) / 2
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        else:
            return None

    def enhanced_basket_fair_value(self, components: dict, order_depths: dict):
        total_value = 0.0
        for product, quantity in components.items():
            if product in order_depths:
                depth = order_depths[product]
                mid = self.compute_weighted_mid_price(depth)
                if mid is None:
                    if depth.buy_orders and depth.sell_orders:
                        mid = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
                    else:
                        mid = 0
                total_value += quantity * mid
            else:
                pass  # Component missing; optionally log.
        return total_value

    def update_rolling_fair(self, product: str, current_fair: float, traderObject: dict, window: int = 5):
        key = product + "_fair_history"
        if key not in traderObject:
            traderObject[key] = []
        traderObject[key].append(current_fair)
        if len(traderObject[key]) > window:
            traderObject[key] = traderObject[key][-window:]
        return sum(traderObject[key]) / len(traderObject[key])

    def enhanced_take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int):
        orders = []
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value - take_width:
                available_buy = self.PRODUCT_LIMIT[product] - position
                volume = min(available_buy, abs(order_depth.sell_orders[best_ask]))
                if volume > 0:
                    orders.append(Order(product, best_ask, volume))
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value + take_width:
                available_sell = self.PRODUCT_LIMIT[product] + position
                volume = min(available_sell, order_depth.buy_orders[best_bid])
                if volume > 0:
                    orders.append(Order(product, best_bid, -volume))
        return orders

    def enhanced_clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: float, position: int):
        orders = []
        if position > 0:
            eligible_bids = [price for price in order_depth.buy_orders if price >= fair_value + clear_width]
            if eligible_bids:
                best_bid = max(eligible_bids)
                volume = min(position, order_depth.buy_orders[best_bid])
                if volume > 0:
                    orders.append(Order(product, best_bid, -volume))
        elif position < 0:
            eligible_asks = [price for price in order_depth.sell_orders if price <= fair_value - clear_width]
            if eligible_asks:
                best_ask = min(eligible_asks)
                volume = min(abs(position), abs(order_depth.sell_orders[best_ask]))
                if volume > 0:
                    orders.append(Order(product, best_ask, volume))
        return orders

    def enhanced_make_orders(self, product: str, order_depth: OrderDepth, fair_value: float, position: int, default_edge: float, join_edge: float):
        orders = []
        if order_depth.sell_orders:
            best_sell = min(order_depth.sell_orders.keys())
            if abs(best_sell - fair_value) <= join_edge:
                ask = best_sell
            else:
                ask = math.ceil(fair_value + default_edge)
        else:
            ask = math.ceil(fair_value + default_edge)
        if order_depth.buy_orders:
            best_buy = max(order_depth.buy_orders.keys())
            if abs(fair_value - best_buy) <= join_edge:
                bid = best_buy
            else:
                bid = math.floor(fair_value - default_edge)
        else:
            bid = math.floor(fair_value - default_edge)
        remaining_buy = self.PRODUCT_LIMIT[product] - position
        remaining_sell = self.PRODUCT_LIMIT[product] + position
        if remaining_buy > 0:
            orders.append(Order(product, bid, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, ask, -remaining_sell))
        return orders

    # ------------------------- Run Method -------------------------
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}

        # --- RAINFOREST_RESIN (Original Strategy) ---
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]['fair_value'],
                self.params[Product.RAINFOREST_RESIN]['take_width'],
                resin_position
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]['fair_value'],
                self.params[Product.RAINFOREST_RESIN]['clear_width'],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
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
            result[Product.RAINFOREST_RESIN] = resin_take_orders + resin_clear_orders + resin_make_orders

        # --- KELP (Original Strategy) ---
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject,
                                                     state.order_depths[Product.SQUID_INK])
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]['take_width'],
                kelp_position,
                self.params[Product.KELP]['prevent_adverse'],
                self.params[Product.KELP]['adverse_volume'],
                traderObject
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]['clear_width'],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]['disregard_edge'],
                self.params[Product.KELP]['join_edge'],
                self.params[Product.KELP]['default_edge']
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        # --- SQUID_INK (Original Strategy) ---
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_fair_value = self.ink_fair_value(state.order_depths[Product.SQUID_INK], traderObject)
            ink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                ink_fair_value,
                self.params[Product.SQUID_INK]['take_width'],
                ink_position,
                self.params[Product.SQUID_INK]['prevent_adverse'],
                self.params[Product.SQUID_INK]['adverse_volume'],
                traderObject
            )
            ink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                ink_fair_value,
                self.params[Product.SQUID_INK]['clear_width'],
                ink_position,
                buy_order_volume,
                sell_order_volume,
            )
            ink_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                ink_fair_value,
                ink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]['disregard_edge'],
                self.params[Product.SQUID_INK]['join_edge'],
                self.params[Product.SQUID_INK]['default_edge'],
            )
            result[Product.SQUID_INK] = ink_take_orders + ink_clear_orders + ink_make_orders

        # --- Enhanced Strategy for DJEMBES, CROISSANTS, and JAMS ---
        for product in [Product.DJEMBES, Product.CROISSANTS, Product.JAMS]:
            if product in state.order_depths:
                depth = state.order_depths[product]
                pos = state.position.get(product, 0)
                weighted_mid = self.compute_weighted_mid_price(depth)
                # Fallback to simple mid-price or previous fair if needed
                if weighted_mid is None or weighted_mid == 0:
                    if depth.buy_orders and depth.sell_orders:
                        weighted_mid = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2
                    else:
                        weighted_mid = traderObject.get(product + "_last_fair", 0)
                # Update rolling fair value (window size = 5)
                fair_value = self.update_rolling_fair(product, weighted_mid, traderObject, window=5)
                traderObject[product + "_last_fair"] = fair_value
                enhanced_take = self.enhanced_take_orders(product, depth, fair_value, self.params[product]['take_width'], pos)
                enhanced_clear = self.enhanced_clear_orders(product, depth, fair_value, self.params[product]['clear_width'], pos)
                enhanced_make = self.enhanced_make_orders(product, depth, fair_value, pos, self.params[product]['default_edge'], self.params[product]['join_edge'])
                result[product] = enhanced_take + enhanced_clear + enhanced_make

        # --- Enhanced Strategy for Basket Products ---
        for basket in [Product.PICNIC_BASKET_1, Product.PICNIC_BASKET_2]:
            if basket in state.order_depths:
                depth = state.order_depths[basket]
                pos = state.position.get(basket, 0)
                components = self.params[basket]['components']
                # Use enhanced basket fair value instead of the simple version
                fair_value = self.enhanced_basket_fair_value(components, state.order_depths)
                # Update rolling fair value for basket product
                fair_value = self.update_rolling_fair(basket, fair_value, traderObject, window=5)
                traderObject[basket + "_last_fair"] = fair_value
                enhanced_take = self.enhanced_take_orders(basket, depth, fair_value, self.params[basket]['take_width'], pos)
                enhanced_clear = self.enhanced_clear_orders(basket, depth, fair_value, self.params[basket]['clear_width'], pos)
                enhanced_make = self.enhanced_make_orders(basket, depth, fair_value, pos, self.params[basket]['default_edge'], self.params[basket]['join_edge'])
                result[basket] = enhanced_take + enhanced_clear + enhanced_make

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        # The logger.flush() call remains unchanged.
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData