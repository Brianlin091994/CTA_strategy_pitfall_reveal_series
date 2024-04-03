from datetime import datetime, time
from typing import Callable, Dict

from vnpy.trader.object import BarData, TickData
from vnpy.trader.constant import Interval


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
