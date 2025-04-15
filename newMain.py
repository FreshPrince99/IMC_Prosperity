"""
trader_advanced.py

This module implements an advanced market-making strategy for three products:
  - RAINFOREST_RESIN
  - KELP
  - SQUID_INK

The strategies are implemented using object‑oriented design, with an abstract
base class ("Strategy") and a common "MarketMakingStrategy" that provides shared
functionality (e.g. calculating mid‐price, exponential moving average, and volatility).
Concrete strategies (RainForestResinStrategy, KelpStrategy, and SquidInkStrategy)
override the `get_default_price` method to compute a product‐specific fair value.

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

##############################################################################
#                           Product Enumeration                              #
##############################################################################
class Product:
    """
    Contains string constants for the products.
    """
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"


##############################################################################
#                           Advanced Strategy Params                         #
##############################################################################
# Advanced strategy parameters for the three products. They can be fine-tuned.
ADVANCED_PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,  # static fair price assumption
        "default_edge": 1,    # offset for market making orders
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
        "take_width": 2.5,    # larger threshold due to higher volatility
        "clear_width": 1,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.228,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2.0,  # offset for orders (increased for volatility)
        "spike_lb": 3,
        "spike_ub": 5.6,
        "offset": 2,
        "reversion_window": 55,
        "reversion_weight": 0.12,
        "vol_threshold": 0.03,   # if volatility exceeds 3%, adjust the order offset
        "vol_multiplier": 2.0,
    }
}

##############################################################################
#                          Abstract Strategy Classes                       #
##############################################################################
class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Attributes:
        product: The product symbol (string) to trade.
        limit: The maximum position limit for the product.
        orders: List of orders to be submitted.
    """
    def __init__(self, product: str, limit: int) -> None:
        self.product = product
        self.limit = limit
        self.orders: List[Order] = []
        
    @abstractmethod
    def act(self, state: TradingState) -> None:
        """
        Performs the trading logic for one iteration given the TradingState.
        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def run(self, state: TradingState) -> List[Order]:
        """
        Clears any previous orders, calls the `act` method, and returns the orders.
        """
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        """
        Appends a buy order.
        """
        self.orders.append(Order(self.product, int(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        """
        Appends a sell order.
        """
        self.orders.append(Order(self.product, int(price), -quantity))

    def save(self) -> Any:
        """
        Returns a dictionary of persistent state variables.
        """
        return {"EMA": getattr(self, "EMA", None),
                "last_mm_mid_price": getattr(self, "last_mm_mid_price", None),
                "history": list(getattr(self, "history", [])),
                "mid_price_history": list(getattr(self, "mid_price_history", []))}
    
    def load(self, data: Any) -> None:
        """
        Loads persistent state variables from `data`.
        """
        if not data:
            return
        self.EMA = data.get("EMA")
        self.last_mm_mid_price = data.get("last_mm_mid_price")
        self.history = deque(data.get("history", []))
        self.mid_price_history = deque(data.get("mid_price_history", []), maxlen=20)

##############################################################################
#                        Market Making Strategy Base                         #
##############################################################################
class MarketMakingStrategy(Strategy):
    """
    Implements common functionality for market-making strategies.

    Includes:
      - Calculating the popular (mid) price from the order book.
      - Maintaining a rolling history of mid prices.
      - Estimating volatility using the rolling history.
      - An abstract method `get_default_price` that each concrete subclass must implement.
    """
    def __init__(self, product: str, limit: int, strategy_args: dict) -> None:
        super().__init__(product, limit)
        # Store strategy-specific parameters in self.params to avoid attribute errors.
        self.params = strategy_args if strategy_args is not None else {}
        self.history = deque(maxlen=20)
        self.mid_price_history = deque(maxlen=20)
        self.EMA_alpha = self.params.get("EMA_alpha", 0.32)
        self.soft_liquidate_thresh = self.params.get("soft_liquidation_thresh", 0.5)
        self.volatility_multiplier = self.params.get("volatility_multiplier", 1.0)
        self.beta_reversion = self.params.get("beta_reversion", 0.369)
        self.volume_threshold = self.params.get("volume_threshold", 12)
        self.last_mm_mid_price = None
        self.EMA = None

    def get_popular_average(self, state: TradingState) -> float:
        """
        Returns the popular average price calculated as the average of the
        best bid and best ask from the order book.
        """
        order_depth = state.order_depths[self.product]
        if order_depth.buy_orders and order_depth.sell_orders:
            most_popular_sell_price = min(order_depth.sell_orders.items(), key=lambda item: item[1])[0]
            most_popular_buy_price = max(order_depth.buy_orders.items(), key=lambda item: item[1])[0]
            return (most_popular_buy_price + most_popular_sell_price) // 2
        return self.last_mm_mid_price if self.last_mm_mid_price is not None else 10000

    def get_EMA(self, state: TradingState) -> float:
        """
        Returns the exponential moving average (EMA) of the popular price.
        """
        avg_price = self.get_popular_average(state)
        if self.EMA is None:
            self.EMA = avg_price
        else:
            self.EMA = self.EMA_alpha * avg_price + (1 - self.EMA_alpha) * self.EMA
        return self.EMA

    def estimate_volatility(self) -> float:
        """
        Estimates volatility as the standard deviation of the recent mid prices.
        """
        if len(self.mid_price_history) < 5:
            return 0
        return float(np.std(np.array(self.mid_price_history)))

    def act(self, state: TradingState) -> None:
        """
        Implements the basic market making logic:
          - Computes a popular mid price.
          - Updates the rolling mid-price history.
          - Estimates volatility.
          - Uses get_default_price (implemented by subclasses) to set a fair price.
          - For SQUID_INK, adjusts the order offset if volatility is high.
          - Places orders to buy below the fair price and sell above.
        """
        mm_price = self.get_popular_average(state)
        self.mid_price_history.append(mm_price)
        vol = self.estimate_volatility()
        
        pos = state.position.get(self.product, 0)
        to_buy = self.limit - pos
        to_sell = self.limit + pos
        
        default_price = self.get_default_price(state)
        
        if self.product == Product.SQUID_INK:
            vol_thresh = self.params.get("vol_threshold", 0.03)
            multiplier = self.params.get("vol_multiplier", 2.0)
            if vol > vol_thresh:
                mm_offset = self.params.get("default_edge", 1) * multiplier
            else:
                mm_offset = self.params.get("default_edge", 1)
        else:
            mm_offset = self.params.get("default_edge", 1)
            
        # Place market making orders for inventory building
        self.buy(math.floor(default_price - mm_offset), to_buy)
        self.sell(math.ceil(default_price + mm_offset), to_sell)
        
    @abstractmethod
    def get_default_price(self, state: TradingState) -> float:
        """
        Abstract method to compute the default fair price for the product.
        """
        pass

##############################################################################
#                   Concrete Advanced Strategies                           #
##############################################################################
class RainForestResinStrategy(MarketMakingStrategy):
    """
    Market making strategy for RAINFOREST_RESIN using a fixed fair value.
    """
    def get_default_price(self, state: TradingState) -> float:
        return 10000

class SquidInkStrategy(MarketMakingStrategy):
    """
    Market making strategy for SQUID_INK.
    Uses a simple regression/mean reversion model on popular prices.
    """
    def get_default_price(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.product]
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is None or best_bid is None:
            return self.last_mm_mid_price if self.last_mm_mid_price is not None else 10000
        mm_mid_price = (best_ask + best_bid) / 2
        if self.last_mm_mid_price is not None:
            last_price = self.last_mm_mid_price
            last_return = (mm_mid_price - last_price) / last_price
            pred_returns = last_return * -0.369
            fair = mm_mid_price + (mm_mid_price * pred_returns)
        else:
            fair = mm_mid_price
        self.last_mm_mid_price = mm_mid_price
        return fair

class KelpStrategy(MarketMakingStrategy):
    """
    Market making strategy for KELP.
    Uses a mean reversion model with a slightly different coefficient.
    """
    def get_default_price(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.product]
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is None or best_bid is None:
            return self.last_mm_mid_price if self.last_mm_mid_price is not None else 10000
        mm_mid_price = (best_ask + best_bid) / 2
        if self.last_mm_mid_price is not None:
            last_price = self.last_mm_mid_price
            last_return = (mm_mid_price - last_price) / last_price
            pred_returns = last_return * -0.5
            fair = mm_mid_price + (mm_mid_price * pred_returns)
        else:
            fair = mm_mid_price
        self.last_mm_mid_price = mm_mid_price
        return fair

##############################################################################
#                     Integrated Trader for Advanced Products               #
##############################################################################
class Trader:
    """
    Integrated Trader class that instantiates advanced strategies for
    RAINFOREST_RESIN, KELP, and SQUID_INK, runs them, and handles state persistence.
    """
    def __init__(self, strategy_args: dict = None) -> None:
        # Define position limits for the advanced products.
        limits = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }
        self.strategy_args = strategy_args if strategy_args else {}
        # Instantiate advanced strategies.
        self.advanced_strategies = {
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
            )
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        result = {}
        trader_data = {}
        # Load persistent data if available.
        if state.traderData and state.traderData != "":
            saved = json.loads(state.traderData)
            for prod in self.advanced_strategies:
                if prod in saved:
                    self.advanced_strategies[prod].load(saved.get(prod))
        # Run advanced strategies.
        for prod, strat in self.advanced_strategies.items():
            if prod in state.order_depths:
                result[prod] = strat.run(state)
            trader_data[prod] = strat.save()
        traderData = jsonpickle.encode(trader_data)
        conversions = 1
        # The logger.flush() call is assumed to be handled externally.
        return result, conversions, traderData
