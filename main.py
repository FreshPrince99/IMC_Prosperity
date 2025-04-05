import json
from typing import Dict, List, Any
from datamodel import  Listing, Observation, ProsperityEncoder, Symbol, OrderDepth, TradingState, Order, Trade

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
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}
        acceptable_values = {
            "RAINFOREST_RESIN": 10000,
            "KELP": 5000
        }
        state_position = state.position
        limit = 50 # pre defined position limits for these products

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Retrieve the Order Depth containing all the market BUY and SELL orders
            order_depth: OrderDepth = state.order_depths[product]
            acceptable_price = acceptable_values[product]
            position = state_position.get(product, 0)

            available_buy = limit - position
            available_sell = limit + position

            # BUY if sell order is below fair value
            for best_ask, best_ask_qty in sorted(order_depth.sell_orders.items()):
                if best_ask < acceptable_price and available_buy > 0:
                    qty = min(-best_ask_qty, available_buy)
                    logger.print("BUY", str(-best_ask_qty) + "x", best_ask)
                    orders.append(Order(product, best_ask, -qty))
                    available_buy-= qty
            
            # SELL if buy order is above fair value
            for best_bid, best_bid_qty in sorted(order_depth.buy_orders.items()):
                if best_bid > acceptable_price and available_sell > 0:
                    qty = min(best_bid_qty, available_sell)
                    logger.print("SELL", str(best_bid_qty) + "x", best_bid)
                    orders.append(Order(product, best_bid, best_bid_qty))
                    available_sell-=qty

            # Add all the above the orders to the result dict
            result[product] = orders
        
        # This has to be modified
        conversions = 0
        trader_data=''

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data