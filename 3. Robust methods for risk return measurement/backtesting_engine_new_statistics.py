from pandas import DataFrame
from statsmodels.api import OLS
import statsmodels.api as sm
from pandas import Series
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict

from vnpy_ctastrategy.backtesting import (BacktestingEngine, 
                                          CtaTemplate, 
                                          Interval, 
                                          BacktestingMode,
                                          OptimizationSetting,
                                          check_optimization_setting,
                                          run_bf_optimization,
                                          run_ga_optimization,
                                          get_target_value,
                                          partial)

class BacktestingEngineNewStatistics(BacktestingEngine):
    """"""
    def __init__(self):
        super().__init__()
        self.ddpercent_only: bool = True # 用于控制计算R_cubed参数时, 是否考虑回撤周期长度的布尔开关.

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df: DataFrame = self.daily_df

        # Init all statistics default value
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_slippage: float = 0
        daily_slippage: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: int = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        return_drawdown_ratio: float = 0
        regressed_annual_return: float = 0
        r_cubed: float = 0
        robust_sharpe_ratio: float = 0
        # Check if balance is always positive
        positive_balance: bool = False

        if df is not None:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            # When balance falls below 0, set daily return to 0
            pre_balance: Series = df["balance"].shift(1)
            pre_balance.iloc[0] = self.capital
            x = df["balance"] / pre_balance
            x[x <= 0] = np.nan
            df["return"] = np.log(x).fillna(0)

            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # 添加部分 calculate regressed_annual_return
            # df["cumreturn"] = (df["balance"] / self.capital - 1) * 100
            df["cumreturn"] = df["return"].expanding(1).sum() * 100


            
            # All balance value needs to be positive
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

        # Calculate statistics value
        if positive_balance:
            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days: int = len(df)
            profit_days: int = len(df[df["net_pnl"] > 0])
            loss_days: int = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration: int = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration: int = 0

            total_net_pnl: float = df["net_pnl"].sum()
            daily_net_pnl: float = total_net_pnl / total_days

            total_commission: float = df["commission"].sum()
            daily_commission: float = total_commission / total_days

            total_slippage: float = df["slippage"].sum()
            daily_slippage: float = total_slippage / total_days

            total_turnover: float = df["turnover"].sum()
            daily_turnover: float = total_turnover / total_days

            total_trade_count: int = df["trade_count"].sum()
            daily_trade_count: int = total_trade_count / total_days

            total_return: float = (end_balance / self.capital - 1) * 100
            annual_return: float = total_return / total_days * self.annual_days
            daily_return: float = df["return"].mean() * 100
            return_std: float = df["return"].std() * 100

            if return_std:
                daily_risk_free: float = self.risk_free / np.sqrt(self.annual_days)
                sharpe_ratio: float = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days)
            else:
                sharpe_ratio: float = 0

            if max_ddpercent:
                return_drawdown_ratio: float = -total_return / max_ddpercent
            else:
                return_drawdown_ratio = 0

            # 计算regressed_annual_return
            regressed_annual_return = calculate_regressed_annual_return(df, self.annual_days)
            # 计算各阶段回撤 calculate period dropdowns
            dropdowns = find_periodic_dropdowns(df)
            # 计算r-cubed指标
            r_cubed = calculate_r_cubed(dropdowns, regressed_annual_return, self.ddpercent_only)
            # 计算robust_sharpe_ratio
            robust_sharpe_ratio = regressed_annual_return / (return_std * np.sqrt(self.annual_days))
        
        
        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")
            self.output(f"Regressed Annual Return: \t {regressed_annual_return:,.4f}")
            self.output(f"R-cubed Ratio: \t{r_cubed:,.4f}")
            self.output(f"Robust Sharpe Ratio: \t{robust_sharpe_ratio:,.4f}")

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
            "regressed_annual_return": regressed_annual_return,
            "r_cubed_ratio": r_cubed,
            "robust_sharpe_ratio": robust_sharpe_ratio,
        }
        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("策略统计指标计算完成")
        
        return statistics

    def run_bf_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output: bool = True,
        max_workers: int = None
    ) -> list:
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_bf_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results

    run_optimization = run_bf_optimization

    def run_ga_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output: bool = True,
        max_workers: int = None
    ) -> list:
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_ga_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results


def calculate_regressed_annual_return(df: DataFrame, annual_days) -> float:
    """计算regressed_annual_return"""
    X = np.linspace(1, len(df), len(df))
    # 回归方程不需要添加常数项, 以y=ax为模型
    Y1 = df["cumreturn"].values
    model1 = OLS(Y1, X)
    result1 = model1.fit()
    # annualise fitted return 
    regressed_annual_return = result1.params[0] * annual_days
    return regressed_annual_return

def calculate_r_cubed(dropdowns: List[Dict], regressed_annual_return: float, ddpercent_only: bool) -> float:
    # 回测中有出现净值不断下跌的情况, 此时dropdowns会是个空值列表
    if dropdowns:
        dropdowns_sorted = sorted(dropdowns, key = lambda x:x["max_ddpercent"])
        if len(dropdowns_sorted)  > 4:
            top_dropdowns_sorted = dropdowns_sorted[:5]
        else:
            top_dropdowns_sorted = dropdowns_sorted
        top_max_dropdown_percents = [i["max_ddpercent"] for i in top_dropdowns_sorted]
        top_max_dropdown_lengths = [i["max_ddpercent_length"] for i in top_dropdowns_sorted]
        # 公式计算r-cubed
        # 由于计算dropdown_length使用的是自然日, 则年化需要用365/年的常数.
        average_top_dropdowns = abs(np.mean(top_max_dropdown_percents))
        average_top_dropdowns_length = np.mean(top_max_dropdown_lengths)
        
        if not ddpercent_only:
            # 计算RAR/top平均回撤/top回撤时间*365天年化
            r_cubed = regressed_annual_return * 365 / average_top_dropdowns / average_top_dropdowns_length
        else:
            # 只计算RAR/前N词平均回撤
            r_cubed = regressed_annual_return / average_top_dropdowns
    else: 
        r_cubed = 0.0

    return r_cubed

def find_periodic_dropdowns(df: DataFrame) -> List[Dict[str, float]]:
    dropdowns = []
    current_dropdown = None

    for index, row in df.iterrows():
        ddpercent = row['ddpercent']

        if not current_dropdown and ddpercent < 0:
            # 新建一个 current_dropdown
            current_dropdown = {
                'start': index,
                'max_ddpercent': ddpercent,
                'end': index,
                'max_ddpercent_length': 1
            }
        elif current_dropdown:
            # 如果balance新高， 意味着本次回撤期结束.
            if ddpercent == 0:
                current_dropdown['end'] = index
                dropdowns.append(current_dropdown)
                current_dropdown = None
            else:
                current_dropdown['end'] = index
                current_dropdown['max_ddpercent_length'] = (current_dropdown["end"] - current_dropdown["start"]).days
                if ddpercent < current_dropdown['max_ddpercent']:
                    current_dropdown['max_ddpercent'] = ddpercent

    return dropdowns
        
def new_evaluate(
    target_name: str,
    strategy_class: CtaTemplate,
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    rate: float,
    slippage: float,
    size: float,
    pricetick: float,
    capital: int,
    end: datetime,
    mode: BacktestingMode,
    ddpercent_only: bool,
    setting: dict,
    
) -> tuple:
    """
    Function for running in multiprocessing.pool
    """
    engine: BacktestingEngineNewStatistics = BacktestingEngineNewStatistics()

    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
        mode=mode
    )
    engine.ddpercent_only = ddpercent_only
    
    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics: dict = engine.calculate_statistics(output=False)

    target_value: float = statistics[target_name]
    return (str(setting), target_value, statistics)

def wrap_evaluate(engine: BacktestingEngine, target_name: str) -> callable:
    """
    Wrap evaluate function with given setting from backtesting engine.
    """
    func: callable = partial(
        new_evaluate,
        target_name,
        engine.strategy_class,
        engine.vt_symbol,
        engine.interval,
        engine.start,
        engine.rate,
        engine.slippage,
        engine.size,
        engine.pricetick,
        engine.capital,
        engine.end,
        engine.mode,
        engine.ddpercent_only
    )
    return func
