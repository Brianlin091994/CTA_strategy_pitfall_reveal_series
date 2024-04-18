"""
RUMI策略
快速sma均线, 慢速wma均线, 
取快慢线差值diff, 再取sma均线得到rumi指标

设置止损trailing_percent
增加risk_appetite, 用于计算每次开仓的手数
设置K_window作为Bar合成的周期
设置max_holding_window 作为持仓周期的上限, 高于该值平仓

策略逻辑来自于VNPY官方文章:
【Elite量化策略实验室】RUMI策略 - 1
https://zhuanlan.zhihu.com/p/610377004

【Elite量化策略实验室】RUMI策略 - 2
https://zhuanlan.zhihu.com/p/618394552

感谢VNPY团队
"""

from vnpy_ctastrategy import CtaTemplate, BarGenerator, ArrayManager, BarData, TradeData
from vnpy.trader.object import Offset, Direction, OrderData, TickData
from bar_generator import ExoticBarGenerator
import numpy as np
from datetime import time

class RumiStrategy(CtaTemplate):
    
    author = "Brian"
    # parameters
    fast_window: int = 5 # 快速均线窗口
    slow_window: int = 60 # 慢速均线窗口
    rumi_window: int = 20 # RUMI计算窗口
    atr_window: int = 20 # 波动atr指标窗口
    trailing_percent: float = 0.01 # 止损止盈百分比
    price_add: float = 1 # 发单超价
    risk_appetite:float = 0.005 # 风险偏好
    contract_size: int = 10 # 合约乘数
    K_window: int = 30 # K线合成窗口
    max_holding_window: int = 100 # 最长持有周期
    
    # variables
    capital: float = 1000000 # 账户资金
    trading_size: int = 1 # 交易手数
    rumi_value: float = 0.0 # rumi值
    rumi_value_prev: float = 0.0 # 上一个周期rumi值
    long_entry: float = 0.0 # 多头建仓价格
    short_entry: float = 0.0 # 空头建仓价格
    target: int = 0 # 目标仓位
    atr: float = 0.0 # atr指标值
    max_window: int = 0 # 最大窗口, 该值用于初始化ArrayManager长度使用
    holding_period_count: int = 0 # 用于最大持仓周期计数
    last_pos: int = 0 # 记录上一个K线的持仓
    
    parameters = ["fast_window", "slow_window", "rumi_window", "trailing_percent", "contract_size", "K_window", "max_holding_window"]
    variables = ["rumi_value", "rumi_value_prev", "long_entry", "short_entry"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name,  vt_symbol, setting)
        
        self.max_window = max(self.fast_window, self.slow_window)
        self.am = ArrayManager(self.max_window + self.rumi_window + 5)
        self.bg = ExoticBarGenerator(self.on_bar, window = self.K_window, on_window_bar = self.on_window_bar)

    def on_init(self) -> None:
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(self.max_window + self.rumi_window + 10)

    def on_start(self) -> None:
        """
        Callback when strategy is started.
        """
        self.write_log("策略开始")

    def on_stop(self) -> None:
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData) -> None:
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """
        Callback of new bar data update.
        """
        self.cancel_all()
        
        self.cal_target_on_stop_order(bar)
        
        self.send_trade_orders(bar)
        
        self.bg.update_bar(bar)

    def on_window_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.am.update_bar(bar)
        
        if not self.am.inited:
            return

        fast_sma_array = self.am.sma(self.fast_window, array = True)
        slow_wma_array = self.am.wma(self.slow_window, array = True)
        self.atr = self.am.atr(self.atr_window)
        diff_array = fast_sma_array - slow_wma_array
        # 丢弃NA值
        diff_array = diff_array[~np.isnan(diff_array)]
        # 借助np.convolve计算均值
        rumi_array = np.convolve(diff_array, np.ones(self.rumi_window) / self.rumi_window, mode = "valid")
        self.rumi_value = rumi_array[-1]
        self.rumi_value_prev = rumi_array[-2]
        # 根据atr计算每次开仓手数, 波动大开仓数减少, 波动小开仓数增加
        if self.atr:
            self.trading_size = self.capital * self.risk_appetite / self.atr / self.contract_size
            self.trading_size = max(self.trading_size, 1)
        # RUMI值上穿0轴做多, 下穿0轴做空
        if self.rumi_value > 0 and self.rumi_value_prev < 0:
            self.target = self.trading_size      
        elif self.rumi_value < 0 and self.rumi_value_prev > 0:
            self.target = -self.trading_size
        # 检查持仓周期
        self.cal_target_on_holding_period()

    def on_trade(self, trade: TradeData) -> None:
        """
        Callback of new trade data update.
        """
        # 记录开仓价格
        if trade.offset == Offset.OPEN:
            if trade.direction == Direction.LONG:
                self.long_entry = trade.price
            elif trade.direction == Direction.SHORT:
                self.short_entry = trade.price    
        # 重置开仓价
        if not self.pos:
            self.long_entry = 0
            self.short_entry = 0

        # self.pos_record = self.pos

    def on_order(self, order: OrderData) -> None:
        """
        Callback of new order data update.
        """
        pass

    def send_trade_orders(self, bar: BarData):
        """根据target发单"""
        diff = self.target - self.pos
        if diff > 0:
            price = bar.close_price + self.price_add
            if self.pos < 0 and abs(self.pos) < diff:
                self.cover(price, abs(self.pos))
                self.buy(price, diff - abs(self.pos))
            elif self.pos < 0 and abs(self.pos) >= diff:
                self.cover(price, diff)
            else:
                self.buy(price, diff)

        elif diff < 0:
            price = bar.close_price - self.price_add
            if self.pos > 0 and self.pos < abs(diff):
                self.sell(price, self.pos)
                self.short(price, abs(diff) - self.pos)
            elif self.pos > 0 and self.pos >= abs(diff):
                self.sell(price, self.pos)
            else:
                self.short(price, diff)

    def cal_target_on_stop_order(self, bar: BarData):
        """计算止损止盈价格目标"""
        if self.long_entry:
            price = self.long_entry * (1 - self.trailing_percent)
            if bar.low_price < price:
                self.target = 0
        elif self.short_entry:
            price = self.short_entry * (1 + self.trailing_percent)
            if bar.high_price > price:
                self.target = 0

    def cal_target_on_holding_period(self):
        """根据持有周期数计算目标仓位"""
        # 如果没有持仓, 直接返回
        if not self.pos:
            self.holding_period_count = 0
            return
        # 如果持仓发生变动, 重新计数
        if self.pos != self.last_pos:
            self.last_pos = self.pos
            self.holding_period_count = 0
            return
        # 计数器+1
        self.holding_period_count += 1
        # 持仓周期超过最大限度, 设置target为0进行平仓
        if self.holding_period_count > self.max_holding_window:
            self.target = 0
            self.holding_period_count = 0

        
