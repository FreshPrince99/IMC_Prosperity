"""
trader_advanced.py

This module implements an advanced market-making strategy for three products:
  - RAINFOREST_RESIN
  - KELP
  - SQUID_INK

The strategies are implemented using object-oriented design, with an abstract
base class ("Strategy") and a common "MarketMakingStrategy" that provides shared
functionality (e.g. calculating mid-price, exponential moving average, and volatility).
Concrete strategies (RainForestResinStrategy, KelpStrategy, and SquidInkStrategy)
override the get_default_price method to compute a product-specific fair value.

All state persistence is handled via JSON through jsonpickle.
"""

from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Any
import json
import jsonpickle
import math
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

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

##############################################################################
#                           Product Enumeration                              #
##############################################################################
class Product:
    # Round 1/2 products:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    # Round 3 products:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

##############################################################################
#                         Advanced Strategy Params                           #
##############################################################################
# All numeric parameters are stored externally.
ADVANCED_PARAMS = {
    # Round 1/2 strategies remain unchanged:
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "default_edge": 1,
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
        "take_width": 2.5,
        "clear_width": 1,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 0.5,   # Tighter offset for competitive quotes
        "spike_lb": 3,
        "spike_ub": 5.6,
        "offset": 2,
        "reversion_window": 55,
        "reversion_weight": 0.12,
        "vol_threshold": 0.03,
        "vol_multiplier": 2.0,
        "momentum_beta": 0.7,
        "raw_weight": 0.6,
        "ema_weight": 0.4,
        "momentum_guard_threshold": 2.0,
        "rolling_window": 5,
        "down_spike_threshold": 0.03,
        "up_spike_threshold": 0.03,
        "spike_recovery_threshold": 0.05,
        "trend_window": 12,
        "trend_slope_threshold": 0.002,
    },
    # Stdev-based mean reversion for Djembe, Croissants, Jams:
    Product.DJEMBES: {
        "strategy": "mean_reversion",
        "default_edge": 0.6,
        "ema_alpha": 0.25,
        "lookback": 20,
        "stdev_factor": 2.0
    },
    Product.CROISSANTS: {
        "strategy": "mean_reversion",
        "default_edge": 0.6,
        "ema_alpha": 0.25,
        "lookback": 20,
        "stdev_factor": 2.0
    },
    Product.JAMS: {
        "strategy": "mean_reversion",
        "default_edge": 0.6,
        "ema_alpha": 0.25,
        "lookback": 20,
        "stdev_factor": 2.0
    },
    Product.PICNIC_BASKET1: {
        "components": {Product.DJEMBES: 1, Product.CROISSANTS: 6, Product.JAMS: 3},
        "default_edge": 1.0
    },
    Product.PICNIC_BASKET2: {
        "components": {Product.CROISSANTS: 4, Product.JAMS: 2},
        "default_edge": 1.0
    },
    # Round 3: Underlying VOLCANIC_ROCK strategy.
    Product.VOLCANIC_ROCK: {
        "take_width": 1.5,
        "clear_width": 0,
        "default_edge": 0.7,
        "EMA_alpha": 0.3,
        "raw_weight": 0.5,
        "ema_weight": 0.5,
        "momentum_beta": 0.6,
        "momentum_guard_threshold": 1.8,
        "imbalance_weight": 1.0,
        "imbalance_threshold": 0.25,
    },
    # Round 3 vouchers:
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "strike": 9500,
        "TTE": 5.0,  # e.g. 5 "days" left; tune as needed
        "poly_fit_degree": 2,
        "implied_vol_buffer": 0.05,  # how far above/below fit to place orders
        "default_edge": 0.8
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "strike": 9750,
        "TTE": 5.0,
        "poly_fit_degree": 2,
        "implied_vol_buffer": 0.05,
        "default_edge": 0.8
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "strike": 9750,
        "TTE": 5.0,
        "poly_fit_degree": 2,
        "implied_vol_buffer": 0.05,
        "default_edge": 0.8
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "strike": 9750,
        "TTE": 5.0,
        "poly_fit_degree": 2,
        "implied_vol_buffer": 0.05,
        "default_edge": 0.8
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "strike": 9750,
        "TTE": 5.0,
        "poly_fit_degree": 2,
        "implied_vol_buffer": 0.05,
        "default_edge": 0.8
    },
}

##############################################################################
#                           Helper Functions                                 #
##############################################################################
def get_order_book_imbalance(order_depth: OrderDepth) -> float:
    total_buy = sum(order_depth.buy_orders.values())
    total_sell = sum(abs(v) for v in order_depth.sell_orders.values())
    if total_buy + total_sell == 0:
        return 0.0
    return (total_sell - total_buy) / (total_sell + total_buy)

##############################################################################
#                           Abstract Strategy Classes                        #
##############################################################################
class Strategy(ABC):
    def __init__(self, product: str, limit: int) -> None:
        self.product = product
        self.limit = limit
        self.orders: List[Order] = []
        
    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()
    
    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders
    
    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, int(price), quantity))
    
    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, int(price), -quantity))
    
    def save(self) -> Any:
        return {
            "EMA": getattr(self, "EMA", None),
            "last_mm_mid_price": getattr(self, "last_mm_mid_price", None),
            "history": list(getattr(self, "history", [])),
            "mid_price_history": list(getattr(self, "mid_price_history", [])),
            "spike_mode": getattr(self, "spike_mode", None),
            "spike_position": getattr(self, "spike_position", 0),
            "lowest_price": getattr(self, "lowest_price", None),
            "highest_price": getattr(self, "highest_price", None),
        }
    
    def load(self, data: Any) -> None:
        if not data:
            return
        self.EMA = data.get("EMA")
        self.last_mm_mid_price = data.get("last_mm_mid_price")
        from collections import deque
        self.history = deque(data.get("history", []))
        self.mid_price_history = deque(data.get("mid_price_history", []), maxlen=20)
        self.spike_mode = data.get("spike_mode", None)
        self.spike_position = data.get("spike_position", 0)
        self.lowest_price = data.get("lowest_price", None)
        self.highest_price = data.get("highest_price", None)

##############################################################################
#                      Market Making Strategy Base                           #
##############################################################################
class MarketMakingStrategy(Strategy):
    def __init__(self, product: str, limit: int, strategy_args: dict) -> None:
        super().__init__(product, limit)
        self.params = strategy_args if strategy_args is not None else {}
        from collections import deque
        self.history = deque(maxlen=20)
        self.mid_price_history = deque(maxlen=20)
        self.EMA = None
        self.last_mm_mid_price = None
        self.spike_mode = None  # For SQUID_INK and vouchers spike reversion.
        self.spike_position = 0
        self.lowest_price = None
        self.highest_price = None
    
    def get_popular_average(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.product]
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return self.last_mm_mid_price if self.last_mm_mid_price is not None else 10000
    
    def estimate_volatility(self) -> float:
        if len(self.mid_price_history) < 5:
            return 0.0
        return float(np.std(np.array(self.mid_price_history)))
    
    def get_EMA(self, state: TradingState) -> float:
        cur_mid = self.get_popular_average(state)
        alpha = self.params.get("EMA_alpha", 0.32)
        if self.EMA is None:
            self.EMA = cur_mid
        else:
            self.EMA = alpha * cur_mid + (1 - alpha) * self.EMA
        return self.EMA
    
    def act(self, state: TradingState) -> None:
        # Default symmetrical order placement
        mid = self.get_popular_average(state)
        self.mid_price_history.append(mid)
        fair = self.get_default_price(state)
        pos = state.position.get(self.product, 0)
        to_buy = self.limit - pos
        to_sell = self.limit + pos
        if self.product == Product.SQUID_INK or "VOUCHER" in self.product:
            vol = self.estimate_volatility()
            vol_thresh = self.params.get("vol_threshold", 0.03)
            multiplier = self.params.get("vol_multiplier", 2.0)
            default_edge = self.params.get("default_edge", 1)
            offset = default_edge * multiplier if vol > vol_thresh else default_edge
        else:
            offset = self.params.get("default_edge", 1)
        buy_price = math.floor(fair - offset)
        sell_price = math.ceil(fair + offset)
        self.buy(buy_price, to_buy)
        self.sell(sell_price, to_sell)
    
    @abstractmethod
    def get_default_price(self, state: TradingState) -> float:
        pass

##############################################################################
#                   Concrete Advanced Strategies (Round 1/2)                 #
##############################################################################
class RainForestResinStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> float:
        return self.params.get("fair_value", 10000)

class SquidInkStrategy(MarketMakingStrategy):
    def act(self, state: TradingState) -> None:
        mid = self.get_popular_average(state)
        self.mid_price_history.append(mid)
        vol = self.estimate_volatility()
        pos = state.position.get(self.product, 0)
        rolling_window = self.params.get("rolling_window", 5)
        if len(self.mid_price_history) >= rolling_window:
            rolling_mean = np.mean(list(self.mid_price_history)[-rolling_window:])
        else:
            rolling_mean = mid
        ema_mid = self.get_EMA(state)
        raw_weight = self.params.get("raw_weight", 0.6)
        ema_weight = self.params.get("ema_weight", 0.4)
        combined_mid = raw_weight * mid + ema_weight * ema_mid
        self.mid_price_history.append(combined_mid)
        down_thresh = self.params.get("down_spike_threshold", 0.03)
        up_thresh = self.params.get("up_spike_threshold", 0.03)
        recovery_thresh = self.params.get("spike_recovery_threshold", 0.05)
        max_buy = self.limit - pos
        max_sell = self.limit + pos
        if self.spike_mode is None:
            if combined_mid < rolling_mean * (1 - down_thresh) and max_buy > 0:
                buy_price = math.floor(combined_mid * 0.99)
                self.buy(buy_price, max_buy)
                self.spike_mode = "down"
                self.spike_position = max_buy
                self.lowest_price = combined_mid
                self.last_mm_mid_price = combined_mid
                return
            if combined_mid > rolling_mean * (1 + up_thresh) and max_sell > 0:
                sell_price = math.ceil(combined_mid * 1.01)
                self.sell(sell_price, max_sell)
                self.spike_mode = "up"
                self.spike_position = max_sell
                self.highest_price = combined_mid
                self.last_mm_mid_price = combined_mid
                return
        else:
            if self.spike_mode == "down":
                self.lowest_price = min(self.lowest_price, combined_mid) if self.lowest_price is not None else combined_mid
                recovery = (combined_mid - self.lowest_price) / self.lowest_price if self.lowest_price > 0 else 0
                if recovery >= recovery_thresh:
                    sell_price = math.ceil(combined_mid * 1.01)
                    self.sell(sell_price, self.spike_position)
                    self.spike_mode = None
                    self.spike_position = 0
                    self.lowest_price = None
                    self.last_mm_mid_price = combined_mid
                    return
            elif self.spike_mode == "up":
                self.highest_price = max(self.highest_price, combined_mid) if self.highest_price is not None else combined_mid
                dip = (self.highest_price - combined_mid) / self.highest_price if self.highest_price > 0 else 0
                if dip >= recovery_thresh:
                    buy_price = math.floor(combined_mid * 0.99)
                    self.buy(buy_price, self.spike_position)
                    self.spike_mode = None
                    self.spike_position = 0
                    self.highest_price = None
                    self.last_mm_mid_price = combined_mid
                    return
        fair = self.momentum_price(state)
        self.last_mm_mid_price = combined_mid
        pos = state.position.get(self.product, 0)
        to_buy = self.limit - pos
        to_sell = self.limit + pos
        vol = self.estimate_volatility()
        vol_thresh = self.params.get("vol_threshold", 0.03)
        multiplier = self.params.get("vol_multiplier", 2.0)
        default_edge = self.params.get("default_edge", 1)
        offset = default_edge * multiplier if vol > vol_thresh else default_edge
        buy_price = math.floor(fair - offset)
        sell_price = math.ceil(fair + offset)
        self.buy(buy_price, to_buy)
        self.sell(sell_price, to_sell)
    
    def momentum_price(self, state: TradingState) -> float:
        mid = self.get_popular_average(state)
        ema_mid = self.get_EMA(state)
        raw_weight = self.params.get("raw_weight", 0.6)
        ema_weight = self.params.get("ema_weight", 0.4)
        combined = raw_weight * mid + ema_weight * ema_mid
        if self.last_mm_mid_price is not None:
            diff = combined - self.last_mm_mid_price
            momentum_beta = self.params.get("momentum_beta", 0.7)
            momentum_adjust = (diff / (self.last_mm_mid_price + 1e-9)) * momentum_beta * combined
        else:
            momentum_adjust = 0
        momentum_guard = self.params.get("momentum_guard_threshold", 2.0)
        vol = self.estimate_volatility()
        if vol > 0 and abs(combined - ema_mid) > momentum_guard * vol:
            momentum_adjust = 0
        fair = combined + momentum_adjust
        return fair
    
    def get_default_price(self, state: TradingState) -> float:
        return self.momentum_price(state)

class KelpStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.last_mm_mid_price if self.last_mm_mid_price is not None else 10000
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mm_mid = (best_bid + best_ask) / 2
        if self.last_mm_mid_price is not None:
            last_price = self.last_mm_mid_price
            last_return = (mm_mid - last_price) / (last_price + 1e-9)
            pred_returns = last_return * -0.5
            fair = mm_mid + (mm_mid * pred_returns)
        else:
            fair = mm_mid
        self.last_mm_mid_price = mm_mid
        return fair

##############################################################################
#           Strategies for Round 2 Component & Basket Products               #
##############################################################################
class ComponentStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.get_popular_average(state)
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2
        ema_mid = self.get_EMA(state)
        raw_weight = self.params.get("raw_weight", 0.6)
        ema_weight = self.params.get("ema_weight", 0.4)
        combined = raw_weight * mid + ema_weight * ema_mid
        if self.last_mm_mid_price is not None:
            diff = combined - self.last_mm_mid_price
            momentum_beta = self.params.get("momentum_beta", 0.5)
            momentum_adjust = (diff / (self.last_mm_mid_price + 1e-9)) * momentum_beta * combined
        else:
            momentum_adjust = 0
        fair = combined + momentum_adjust
        self.last_mm_mid_price = combined
        return fair

class BasketStrategy(Strategy):
    def __init__(self, product: str, limit: int, strategy_args: dict) -> None:
        super().__init__(product, limit)
        self.params = strategy_args if strategy_args else {}
        from collections import deque
        self.mid_price_history = deque(maxlen=20)
        self.EMA = None
        self.components = self.params.get("components", {})
    def get_component_mid(self, comp: str, state: TradingState) -> float:
        if comp in state.order_depths:
            depth = state.order_depths[comp]
            if depth.buy_orders and depth.sell_orders:
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                return (best_bid + best_ask) / 2
        return self.params.get("default_fair", 0)
    def compute_fair_value(self, state: TradingState) -> float:
        fair = 0.0
        for comp, ratio in self.components.items():
            mid = self.get_component_mid(comp, state)
            fair += mid * ratio
        return fair
    def act(self, state: TradingState) -> None:
        fair = self.compute_fair_value(state)
        if self.EMA is None:
            self.EMA = fair
        else:
            alpha = self.params.get("EMA_alpha", 0.32)
            self.EMA = alpha * fair + (1 - alpha) * self.EMA
        pos = state.position.get(self.product, 0)
        to_buy = self.limit - pos
        to_sell = self.limit + pos
        default_edge = self.params.get("default_edge", 1)
        buy_price = math.floor(fair - default_edge)
        sell_price = math.ceil(fair + default_edge)
        self.buy(buy_price, to_buy)
        self.sell(sell_price, to_sell)
    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders
    def save(self) -> Any:
        return {"EMA": self.EMA, "mid_price_history": list(self.mid_price_history)}
    def load(self, data: Any) -> None:
        if not data:
            return
        self.EMA = data.get("EMA")
        from collections import deque
        self.mid_price_history = deque(data.get("mid_price_history", []), maxlen=20)

##############################################################################
#                  New Round 3 Strategies for Underlying & Vouchers           #
##############################################################################
class Round3UnderlyingStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.last_mm_mid_price if self.last_mm_mid_price is not None else 10000
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        raw_mid = (best_bid + best_ask) / 2
        ema_mid = self.get_EMA(state)
        raw_weight = self.params.get("raw_weight", 0.5)
        ema_weight = self.params.get("ema_weight", 0.5)
        combined_mid = raw_weight * raw_mid + ema_weight * ema_mid
        imbalance = get_order_book_imbalance(order_depth)
        imbalance_thresh = self.params.get("imbalance_threshold", 0.25)
        imbalance_weight = self.params.get("imbalance_weight", 1.0)
        adjustment = 0
        if imbalance > imbalance_thresh:
            adjustment = imbalance_weight * imbalance * combined_mid
        elif imbalance < -imbalance_thresh:
            adjustment = imbalance_weight * imbalance * combined_mid
        blended = combined_mid + adjustment
        if self.last_mm_mid_price is not None:
            diff = blended - self.last_mm_mid_price
            momentum_beta = self.params.get("momentum_beta", 0.6)
            momentum_adjust = (diff / (self.last_mm_mid_price + 1e-9)) * momentum_beta * blended
        else:
            momentum_adjust = 0
        fair = blended + momentum_adjust
        self.last_mm_mid_price = blended
        return fair

class VoucherStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> float:
        # Compute voucher's own market mid-price.
        order_depth = state.order_depths[self.product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.last_mm_mid_price if self.last_mm_mid_price is not None else 10000
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        voucher_mid = (best_bid + best_ask) / 2
        
        # Get underlying VOLCANIC_ROCK price.
        underlying_depth = state.order_depths.get(Product.VOLCANIC_ROCK, None)
        if underlying_depth and underlying_depth.buy_orders and underlying_depth.sell_orders:
            underlying_bid = max(underlying_depth.buy_orders.keys())
            underlying_ask = min(underlying_depth.sell_orders.keys())
            underlying_mid = (underlying_bid + underlying_ask) / 2
        else:
            underlying_mid = voucher_mid
        
        ema_mid = self.get_EMA(state)
        raw_weight = self.params.get("raw_weight", 0.5)
        ema_weight = self.params.get("ema_weight", 0.5)
        combined_mid = raw_weight * voucher_mid + ema_weight * ema_mid
        
        # Adjust based on difference between underlying price and voucher strike.
        strike = self.params.get("strike", 10000)
        adjustment_factor = self.params.get("voucher_adjustment_factor", 0.0005)
        voucher_adjustment = adjustment_factor * (underlying_mid - strike) * combined_mid
        
        blended = combined_mid + voucher_adjustment
        
        # Apply momentum adjustment.
        if self.last_mm_mid_price is not None:
            diff = blended - self.last_mm_mid_price
            momentum_beta = self.params.get("momentum_beta", 0.5)
            momentum_adjust = (diff / (self.last_mm_mid_price + 1e-9)) * momentum_beta * blended
        else:
            momentum_adjust = 0
        
        fair = blended + momentum_adjust
        self.last_mm_mid_price = blended
        return fair

##############################################################################
#                     Integrated Trader for All Products                     #
##############################################################################
class Trader:
    def __init__(self, strategy_args: dict = None) -> None:
        limits = {
            # Round 1/2 limits:
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.DJEMBES: 60,
            Product.CROISSANTS: 240,
            Product.JAMS: 350,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            # Round 3 limits:
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
        }
        self.strategy_args = strategy_args if strategy_args else {}
        self.advanced_strategies = {
            # Round 1/2 strategies:
            Product.RAINFOREST_RESIN: RainForestResinStrategy(
                Product.RAINFOREST_RESIN,
                limits[Product.RAINFOREST_RESIN],
                self.strategy_args.get(Product.RAINFOREST_RESIN, ADVANCED_PARAMS.get(Product.RAINFOREST_RESIN, {}))
            ),
            Product.KELP: KelpStrategy(
                Product.KELP,
                limits[Product.KELP],
                self.strategy_args.get(Product.KELP, ADVANCED_PARAMS.get(Product.KELP, {}))
            ),
            Product.SQUID_INK: SquidInkStrategy(
                Product.SQUID_INK,
                limits[Product.SQUID_INK],
                self.strategy_args.get(Product.SQUID_INK, ADVANCED_PARAMS.get(Product.SQUID_INK, {}))
            ),
            Product.DJEMBES: ComponentStrategy(
                Product.DJEMBES,
                limits[Product.DJEMBES],
                self.strategy_args.get(Product.DJEMBES, ADVANCED_PARAMS.get(Product.DJEMBES, {}))
            ),
            Product.CROISSANTS: ComponentStrategy(
                Product.CROISSANTS,
                limits[Product.CROISSANTS],
                self.strategy_args.get(Product.CROISSANTS, ADVANCED_PARAMS.get(Product.CROISSANTS, {}))
            ),
            Product.JAMS: ComponentStrategy(
                Product.JAMS,
                limits[Product.JAMS],
                self.strategy_args.get(Product.JAMS, ADVANCED_PARAMS.get(Product.JAMS, {}))
            ),
            Product.PICNIC_BASKET1: BasketStrategy(
                Product.PICNIC_BASKET1,
                limits[Product.PICNIC_BASKET1],
                self.strategy_args.get(Product.PICNIC_BASKET1, ADVANCED_PARAMS.get(Product.PICNIC_BASKET1, {}))
            ),
            Product.PICNIC_BASKET2: BasketStrategy(
                Product.PICNIC_BASKET2,
                limits[Product.PICNIC_BASKET2],
                self.strategy_args.get(Product.PICNIC_BASKET2, ADVANCED_PARAMS.get(Product.PICNIC_BASKET2, {}))
            ),
            # Round 3 strategies:
            Product.VOLCANIC_ROCK: Round3UnderlyingStrategy(
                Product.VOLCANIC_ROCK,
                limits[Product.VOLCANIC_ROCK],
                self.strategy_args.get(Product.VOLCANIC_ROCK, ADVANCED_PARAMS.get(Product.VOLCANIC_ROCK, {}))
            ),
            Product.VOLCANIC_ROCK_VOUCHER_9500: VoucherStrategy(
                Product.VOLCANIC_ROCK_VOUCHER_9500,
                limits[Product.VOLCANIC_ROCK_VOUCHER_9500],
                self.strategy_args.get(Product.VOLCANIC_ROCK_VOUCHER_9500, ADVANCED_PARAMS.get(Product.VOLCANIC_ROCK_VOUCHER_9500, {}))
            ),
            Product.VOLCANIC_ROCK_VOUCHER_9750: VoucherStrategy(
                Product.VOLCANIC_ROCK_VOUCHER_9750,
                limits[Product.VOLCANIC_ROCK_VOUCHER_9750],
                self.strategy_args.get(Product.VOLCANIC_ROCK_VOUCHER_9750, ADVANCED_PARAMS.get(Product.VOLCANIC_ROCK_VOUCHER_9750, {}))
            ),
            Product.VOLCANIC_ROCK_VOUCHER_10000: VoucherStrategy(
                Product.VOLCANIC_ROCK_VOUCHER_10000,
                limits[Product.VOLCANIC_ROCK_VOUCHER_10000],
                self.strategy_args.get(Product.VOLCANIC_ROCK_VOUCHER_10000, ADVANCED_PARAMS.get(Product.VOLCANIC_ROCK_VOUCHER_10000, {}))
            ),
            Product.VOLCANIC_ROCK_VOUCHER_10250: VoucherStrategy(
                Product.VOLCANIC_ROCK_VOUCHER_10250,
                limits[Product.VOLCANIC_ROCK_VOUCHER_10250],
                self.strategy_args.get(Product.VOLCANIC_ROCK_VOUCHER_10250, ADVANCED_PARAMS.get(Product.VOLCANIC_ROCK_VOUCHER_10250, {}))
            ),
            Product.VOLCANIC_ROCK_VOUCHER_10500: VoucherStrategy(
                Product.VOLCANIC_ROCK_VOUCHER_10500,
                limits[Product.VOLCANIC_ROCK_VOUCHER_10500],
                self.strategy_args.get(Product.VOLCANIC_ROCK_VOUCHER_10500, ADVANCED_PARAMS.get(Product.VOLCANIC_ROCK_VOUCHER_10500, {}))
            ),
        }
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        result = {}
        trader_data = {}
        if state.traderData and state.traderData != "":
            saved = json.loads(state.traderData)
            for prod in self.advanced_strategies:
                if prod in saved:
                    self.advanced_strategies[prod].load(saved.get(prod))
        for prod, strat in self.advanced_strategies.items():
            if prod in state.order_depths:
                result[prod] = strat.run(state)
            trader_data[prod] = strat.save()
        traderData = jsonpickle.encode(trader_data)
        conversions = 1
        # Assume logger.flush(state, result, conversions, traderData) is handled externally.
        return result, conversions, traderData