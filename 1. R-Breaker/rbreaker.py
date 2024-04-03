from dailybar_generator import DailyBarGenerator
import numpy as np
from datetime import time
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager
)

class RBreakerStrategy(CtaTemplate):
    """"""
    author = "Brian"
    # parameters
    observe_size: float = 0.4 # 观察系数
    reversal_size: float = 0.1 # 反转系数
    break_size: float = 0.25 # 突破系数
    unit: int = 1 # 交易单位

    trailing_long: float = 0.1 # 高位下跌,多头移动出场百分位
    trailing_short: float = 0.1 # 低位上涨,空头移动出场百分位
    # multiplier = 3

    # variables
    # 以下价格从高到低排列
    buy_break: float = 0   # 突破买入价
    sell_setup: float = 0  # 观察卖出价
    sell_enter: float = 0  # 反转卖出价
    buy_enter: float = 0   # 反转买入价
    buy_setup: float = 0   # 观察买入价
    sell_break: float = 0  # 突破卖出价

    intra_day_high: float = 0.0 # 日内最高价
    intra_day_low: float = np.inf # 日内最低价

    prev_high: float = 0 # 前一日高点
    prev_low: float = 0 # 前一日低点
    prev_close: float = 0 # 前一日收盘价

    entry_time = time(21,00) # 交易开始时间
    exit_time = time(14,50) # 交易结束时间

    parameters = ["observe_size", "reversal_size", "break_size", "unit", "trailing_long", "trailing_short"]
    variables = ["buy_break", "sell_setup", "sell_enter", "buy_enter", "buy_setup", "sell_break"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(
            cta_engine, strategy_name, vt_symbol, setting
        )

        self.bg_daily = DailyBarGenerator(self.on_daily_bar, time(14,59))
        # self.am_daily = ArrayManager()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(1)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        pass
        # self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        if bar.datetime.time() <= self.exit_time or bar.datetime.time() >= self.entry_time:
            self.send_trade_order(bar)
        else:
            self.close_all_pos(bar)
        
        self.bg_daily.update_bar(bar)
        self.put_event()
    def on_daily_bar(self, bar:BarData):
        # 收到日K线推送后, 计算下一个交易日的信号.
        # self.am_daily.update_bar(bar)

        self.prev_high = bar.high_price
        self.prev_low = bar.low_price
        self.prev_close = bar.close_price
        self.intra_day_high: float = 0.0
        self.intra_day_low: float = np.inf

        self.cal_signal()
    
    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
    
    def cal_signal(self):
        # 计算6个交易信号
        f1 = self.observe_size
        f2 = self.reversal_size
        f3 = self.break_size
        
        # 突破买入 
        self.buy_break = self.prev_high + f1*(self.prev_close-self.prev_low) + f3*(1+f1)*(self.prev_high-self.prev_low)
        # 观察卖出价
        self.sell_setup = self.prev_high + f1*(self.prev_close-self.prev_low)
        # 反转卖出价
        self.sell_enter = 0.5*(self.prev_high+self.prev_low) + 0.5*f2*(self.prev_high-self.prev_low)
        # 反转买入价
        self.buy_enter = 0.5*(self.prev_high+self.prev_low) - 0.5*f2*(self.prev_high-self.prev_low)
        # 观察买入
        self.buy_setup = self.prev_low - f1*(self.prev_high-self.prev_close)
        # 突破卖出
        self.sell_break = self.prev_low - f1*(self.prev_high-self.prev_close) - f3*(1+f1)*(self.prev_high-self.prev_close)
    
    def send_trade_order(self, bar: BarData):
        
        # 今日最高价和最低价
        self.intra_day_high = max(self.intra_day_high, bar.high_price)
        self.intra_day_low = min(self.intra_day_low, bar.low_price)

        if not self.pos:
            # 今日最高价>观察卖出价, 且最低价>观察买入价
            if self.intra_day_high > self.sell_setup and self.intra_day_low > self.buy_setup:
                # 发出停止单, 当价格超过buy_break追多, 跌下sell_enter追空.
                self.buy(self.buy_break, self.unit, stop=True)
                self.short(self.sell_enter, self.unit, stop=True)
            # 今日最低价<观察买入价, 且最高价低于观察卖出价
            elif self.intra_day_low < self.buy_setup and self.intra_day_high < self.sell_setup:
                # 发出停止单, 当价格超过buy_enter追多, 跌下sell_break追空
                self.buy(self.buy_enter, self.unit, stop=True)
                self.short(self.sell_break, self.unit, stop=True)
        elif self.pos > 0:
            # 多头仓位设置出场条件
            long_stop = self.intra_day_high * (1 - self.trailing_long)
            self.sell(long_stop, self.pos, stop = True)
        elif self.pos < 0:
            # 空头仓位设置出场条件
            short_stop = self.intra_day_low * (1 + self.trailing_short)
            self.cover(short_stop, abs(self.pos), stop = True)
            
    def close_all_pos(self, bar: BarData):
        if self.pos > 0:
            self.sell(bar.close_price*0.99, self.pos)
        elif self.pos < 0:
            self.cover(bar.close_price * 1.01, abs(self.pos))