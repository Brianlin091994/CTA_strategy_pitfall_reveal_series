from datetime import datetime, time
from typing import Callable, Dict

from vnpy.trader.object import BarData, TickData
from vnpy.trader.constant import Interval
from vnpy.trader.utility import BarGenerator


class DailyBarGenerator:
    """日K线合成器"""

    def __init__(self, on_daily_bar: Callable, end_time: time) -> None:
        """构造函数"""
        # 日K线推送函数
        self.on_daily_bar: Callable = on_daily_bar

        # 每日收盘时间
        self.end_time: time = end_time

        # 合成中的日K线
        self.daily_bar: BarData = None

    def update_bar(self, bar: BarData) -> None:
        """更新分钟K线"""
        # 当日第一根分钟K线
        if not self.daily_bar:
            self.daily_bar = BarData(
                gateway_name=bar.gateway_name,
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=bar.datetime.replace(hour=0, minute=0),
                interval=Interval.DAILY,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price
            )
        # 日内后续更新
        else:
            daily_bar = self.daily_bar

            daily_bar.volume += bar.volume
            daily_bar.turnover += bar.turnover
            daily_bar.open_interest = bar.open_interest
            daily_bar.high_price = max(daily_bar.high_price, bar.high_price)
            daily_bar.low_price = min(daily_bar.low_price, bar.low_price)
            daily_bar.close_price = bar.close_price
            daily_bar.datetime = bar.datetime

        # 如果到了收盘时间点
        if bar.datetime.time() == self.end_time:
            # 推送日K线
            self.daily_bar.datetime = self.daily_bar.datetime.replace(hour=0, minute=0)
            self.on_daily_bar(self.daily_bar)

            # 清空缓存
            self.daily_bar = None


class ExoticBarGenerator(BarGenerator):
    """
    合成非60整除的Bar数据
    """

    def __init__(
            self,
            on_bar: Callable,
            window: int = 0,
            on_window_bar: Callable = None,
            interval: Interval = Interval.MINUTE
    ):
        super().__init__(on_bar, window, on_window_bar, interval)
        self.last_bar: BarData = None

    def update_exotic_bar(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.window_bar:
            dt: datetime = bar.datetime.replace(second=0, microsecond=0)
            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )
        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high_price = max(
                self.window_bar.high_price,
                bar.high_price
            )
            self.window_bar.low_price = min(
                self.window_bar.low_price,
                bar.low_price
            )

        # Update close price/volume/turnover into window bar
        self.window_bar.close_price = bar.close_price
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover
        self.window_bar.open_interest = bar.open_interest

        # Check if window bar completed
        # if not (bar.datetime.minute + 1) % self.window:
        #     self.on_window_bar(self.window_bar)
        #     self.window_bar = None
        if self.last_bar and bar.datetime.minute != self.last_bar.datetime.minute:
            self.interval_count += 1
            if not self.interval_count % self.window:
                finished = True
                self.interval_count = 0

        if finished:
            self.on_window_bar(self.window_bar)
            self.window_bar = None

        self.last_bar = bar
