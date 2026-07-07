# Day 7 - 最终项目与研究报告

版本说明：本文档把前 6 天内容整合成一个完整量化研究项目。目标是产出一份可复现、可解释、谨慎的研究报告。

## 今日目标

今天的目标是完成一个小型量化研究项目：

```text
四 ETF 横截面动量策略研究报告
```

完成今天后，你应该拥有：

- 一个完整 notebook。
- 一个策略回测结果。
- 一份 Markdown 研究报告。
- 清晰的局限性说明。
- 后续学习和迁移路径。

## 0. 入门者如何完成最终项目

今天不是学习新概念，而是把前 6 天的内容串起来。你可以把它当成一次小型研究交付。

建议按这个顺序完成：

```text
先运行数据下载
再计算信号
再生成权重
再运行回测
再生成图表和表格
最后写报告
```

不要一边改策略一边写报告。最终项目的纪律是：

1. 先固定策略规则。
2. 再运行回测。
3. 再观察结果。
4. 最后写解释和局限性。

如果结果不好，不要临时改参数让它变好。结果不好也是研究结果。第一周的目标是完成规范研究流程，不是挑出最好看的参数。

## 1. 最终项目定义

项目名称：

```text
四 ETF 横截面动量策略研究
```

投资范围：

- SPY：美国大盘股票。
- QQQ：美国科技成长。
- TLT：长期美国国债。
- GLD：黄金。

研究假设：

```text
过去 12 个月表现较强的大类资产，未来 1 个月可能继续表现较强。
```

策略规则：

```text
1. 使用日线自动调整价格。
2. 每个交易日计算 252 日动量：price / price.shift(252) - 1。
3. 信号延迟 1 个交易日。
4. 每月月末读取最新信号。
5. 选择动量最高的 2 个 ETF。
6. 每个入选 ETF 权重 50%。
7. 不做空，不杠杆。
8. 下一交易日开始持有。
9. 每月调仓。
10. 单边交易成本假设为 0.05%。
```

基准：

```text
SPY、QQQ、TLT、GLD 四 ETF 等权买入持有。
```

评估指标：

- 年化收益。
- 年化波动率。
- Sharpe ratio。
- 最大回撤。
- 最好单日收益。
- 最差单日收益。
- 年度收益。
- 年度换手率。

## 2. 最终目录结构

建议项目结构：

```text
quant-research-lab/
  notebooks/
    day-07-final-project.ipynb
  data/
    processed/
      etf_prices.parquet
      etf_returns.parquet
  reports/
    four-etf-momentum-report.md
    four-etf-momentum-performance.csv
    four-etf-momentum-yearly-returns.csv
```

## 3. 完整代码骨架

把下面内容放进 `notebooks/day-07-final-project.ipynb`，按单元格运行并检查每一步输出。

### 3.1 导入依赖

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
```

### 3.2 参数

```python
ASSETS = ["SPY", "QQQ", "TLT", "GLD"]
START = "2015-01-01"
END = "2026-01-01"
LOOKBACK = 252
TOP_N = 2
COST_RATE = 0.0005
PERIODS_PER_YEAR = 252
```

参数解释：

- `LOOKBACK = 252`：约 12 个月交易日。
- `TOP_N = 2`：选择前两个动量资产。
- `COST_RATE = 0.0005`：单边成交金额 0.05%。
- `PERIODS_PER_YEAR = 252`：美股常用年交易日近似。

### 参数不要随手改

最终项目里先不要优化参数。默认参数的意义是：

- `LOOKBACK = 252`：近似 12 个月动量，是常见动量窗口。
- `TOP_N = 2`：在 4 个 ETF 中持有一半资产，既有选择性又不过度集中。
- `COST_RATE = 0.0005`：给交易成本一个保守但不极端的学习假设。

如果你运行后发现结果不理想，不要马上改成 126、189 或 300。参数稳健性会在第二周 Day 10 深入处理。今天你要先交付一个诚实、可复现的基准版本。

### 3.3 获取数据

```python
raw = yf.download(
    tickers=ASSETS,
    start=START,
    end=END,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

prices = raw["Close"].dropna(how="all")
prices = prices[ASSETS]
returns = prices.pct_change().dropna(how="all")

prices.to_parquet("data/processed/etf_prices.parquet")
returns.to_parquet("data/processed/etf_returns.parquet")

prices.tail()
```

如果下载失败，检查网络、ticker、日期，或稍后重试。学习阶段可以使用 Day 2 保存的数据继续。

### 3.4 绩效函数

```python
def annualized_return(daily_returns: pd.Series, periods_per_year: int = 252):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan
    ending_wealth = (1 + daily_returns).prod()
    return ending_wealth ** (periods_per_year / len(daily_returns)) - 1

def annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan
    return daily_returns.std() * np.sqrt(periods_per_year)

def drawdown_series(daily_returns: pd.Series):
    daily_returns = daily_returns.dropna()
    wealth = (1 + daily_returns).cumprod()
    peaks = wealth.cummax()
    return wealth / peaks - 1

def max_drawdown(daily_returns: pd.Series):
    dd = drawdown_series(daily_returns)
    if len(dd) == 0:
        return np.nan
    return dd.min()

def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252):
    ann_ret = annualized_return(daily_returns, periods_per_year)
    ann_vol = annualized_volatility(daily_returns, periods_per_year)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol

def performance_summary(daily_returns: pd.DataFrame | pd.Series):
    if isinstance(daily_returns, pd.Series):
        daily_returns = daily_returns.to_frame(daily_returns.name or "strategy")

    rows = []
    for column in daily_returns.columns:
        series = daily_returns[column].dropna()
        rows.append(
            {
                "name": column,
                "annual_return": annualized_return(series),
                "annual_volatility": annualized_volatility(series),
                "sharpe": sharpe_ratio(series),
                "max_drawdown": max_drawdown(series),
                "best_day": series.max(),
                "worst_day": series.min(),
            }
        )
    return pd.DataFrame(rows).set_index("name")
```

### 3.5 构造信号和权重

```python
momentum = prices / prices.shift(LOOKBACK) - 1
tradable_signal = momentum.shift(1)
monthly_signal = tradable_signal.resample("ME").last()

def top_n_weights(signal_row: pd.Series, n: int):
    weights = pd.Series(0.0, index=signal_row.index)
    valid = signal_row.dropna()
    if len(valid) < n:
        return weights
    winners = valid.nlargest(n).index
    weights.loc[winners] = 1 / n
    return weights

monthly_target_weights = monthly_signal.apply(top_n_weights, axis=1, n=TOP_N)
daily_target_weights = monthly_target_weights.reindex(returns.index).ffill().fillna(0.0)

daily_target_weights.tail()
```

### 3.6 回测函数

```python
def backtest_weight_strategy(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_rate: float,
):
    target_weights = target_weights.reindex(asset_returns.index).ffill().fillna(0.0)

    realized_weights = target_weights.shift(1).fillna(0.0)
    gross_returns = realized_weights.mul(asset_returns, axis=1).sum(axis=1)

    turnover = target_weights.diff().abs().sum(axis=1).fillna(target_weights.abs().sum(axis=1))
    costs = turnover * cost_rate
    net_returns = gross_returns - costs

    result = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "cost": costs,
            "net_return": net_returns,
            "turnover": turnover,
        }
    )

    return result, realized_weights

bt, realized_weights = backtest_weight_strategy(
    asset_returns=returns,
    target_weights=daily_target_weights,
    cost_rate=COST_RATE,
)

bt.tail()
```

### 3.7 基准

```python
equal_weights = pd.DataFrame(
    1 / len(ASSETS),
    index=returns.index,
    columns=ASSETS,
)

benchmark_returns = equal_weights.mul(returns, axis=1).sum(axis=1)
benchmark_returns.name = "equal_weight"
```

### 3.8 结果对比

```python
comparison_returns = pd.DataFrame(
    {
        "momentum_gross": bt["gross_return"],
        "momentum_net": bt["net_return"],
        "equal_weight": benchmark_returns,
    }
).dropna()

perf = performance_summary(comparison_returns)
perf
```

保存：

```python
perf.to_csv("reports/four-etf-momentum-performance.csv")
```

### 3.9 图表

累计收益：

```python
(1 + comparison_returns).cumprod().plot(figsize=(12, 6), title="Growth of $1")
plt.grid(True)
plt.show()
```

回撤：

```python
comparison_drawdowns = comparison_returns.apply(drawdown_series)
comparison_drawdowns.plot(figsize=(12, 6), title="Drawdown")
plt.grid(True)
plt.show()
```

权重：

```python
realized_weights.plot.area(figsize=(12, 6), title="Realized Strategy Weights")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
```

年度收益：

```python
yearly_returns = comparison_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
yearly_returns.to_csv("reports/four-etf-momentum-yearly-returns.csv")
yearly_returns
```

年度收益图：

```python
yearly_returns.plot(kind="bar", figsize=(12, 6), title="Calendar Year Returns")
plt.axhline(0, color="black", linewidth=1)
plt.grid(True)
plt.show()
```

换手率：

```python
annual_turnover = bt["turnover"].resample("YE").sum()
annual_turnover.plot(kind="bar", figsize=(12, 5), title="Annual Turnover")
plt.grid(True)
plt.show()

print("Average annual turnover:", annual_turnover.mean())
print("Total estimated cost:", bt["cost"].sum())
```

## 4. 报告模板

创建 `reports/four-etf-momentum-report.md`，使用下面模板。

```markdown
# 四 ETF 横截面动量策略研究报告

## 摘要

本文研究 SPY、QQQ、TLT、GLD 四个 ETF 的横截面动量策略。策略每月选择过去 252 个交易日动量最高的 2 个 ETF，各配置 50%，不做空、不使用杠杆，并假设单边交易成本为 0.05%。研究目标是检验简单大类资产动量规则在历史样本中的风险调整表现。

## 研究假设

过去 12 个月表现较强的大类资产，未来 1 个月可能继续表现较强。可能原因包括宏观趋势延续、信息扩散缓慢、资金流入强势资产和投资者反应不足。

## 数据

数据范围：2015-01-01 到 2026-01-01。

资产范围：SPY、QQQ、TLT、GLD。

数据频率：日线。

价格处理：使用自动调整后的价格计算收益率。

数据限制：yfinance 适合学习和教育研究，不是生产级授权数据源。

## 策略规则

1. 每日计算 252 日动量。
2. 信号延迟 1 个交易日，避免未来函数。
3. 每月月末读取信号。
4. 选择动量最高的 2 个 ETF。
5. 每个入选 ETF 权重 50%。
6. 权重在下一交易日生效。
7. 不做空，不杠杆。
8. 单边交易成本为 0.05%。

## 基准

基准为 SPY、QQQ、TLT、GLD 四 ETF 等权买入持有组合。

## 结果

在此处粘贴绩效表：

| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|
| momentum_net |  |  |  |  |
| equal_weight |  |  |  |  |

## 图表说明

累计收益图显示策略和基准的长期财富增长路径。

回撤图显示策略在历史高点后的下跌幅度。

年度收益图显示策略是否依赖少数年份。

权重图显示策略是否频繁切换资产。

## 风险和局限

- 样本区间有限，结果可能依赖 2015 到 2026 的特殊市场环境。
- 使用 yfinance 数据，不适合作为生产级实盘依据。
- 成本模型简化，没有包含买卖价差、市场冲击、税费和真实成交约束。
- ETF 池很小，结果可能不具备普遍性。
- 参数未做稳健性测试，例如 126 日、189 日、252 日、12 个月跳过最近 1 个月等。
- 策略可能在趋势反转和震荡市场中表现较差。
- 历史表现不能证明未来收益。

## 结论

根据回测结果，说明策略相对等权基准是否改善了收益、风险或回撤。结论必须谨慎表达，只能说“在本样本和假设下表现如何”，不能说“未来会赚钱”。

## 下一步

- 测试不同动量窗口。
- 加入现金或短债作为防守资产。
- 增加更多大类资产 ETF。
- 做样本内和样本外分区。
- 使用正式数据源复核。
- 加入更真实的交易成本模型。
```

## 5. 如何写结论

如果策略更好，不要写：

```text
策略有效，可以实盘。
```

应该写：

```text
在本样本、数据源和成本假设下，动量策略的成本后 Sharpe 高于等权基准，最大回撤低于等权基准。这说明简单横截面动量在该 ETF 池上有进一步研究价值。但结果可能受样本区间、参数选择和数据源影响，不能直接推广到未来或实盘。
```

如果策略不好，也不是失败。应该写：

```text
在本样本和成本假设下，动量策略未能稳定优于等权基准。可能原因包括 ETF 池过小、参数不适合、趋势不稳定、交易成本侵蚀收益，或等权基准本身已经足够强。该结果说明复杂规则未必优于简单分散化。
```

量化研究最重要的能力不是让每个策略都赚钱，而是诚实判断一个想法是否值得继续。

## 入门者补充：报告评分标准

你可以用下面标准检查自己的报告。

| 项目 | 合格标准 |
|---|---|
| 研究假设 | 能一句话说明策略为什么可能有效 |
| 数据说明 | 写清资产、日期、频率、价格处理和数据源限制 |
| 策略规则 | 别人按文字能复现信号、权重和调仓 |
| 成本假设 | 明确写出成本率和扣除方式 |
| 基准 | 至少和等权组合比较 |
| 指标 | 包含年化收益、波动率、Sharpe、最大回撤 |
| 图表 | 至少有累计收益和回撤 |
| 局限性 | 主动说明数据、成本、样本和过拟合风险 |
| 结论 | 谨慎表达，不承诺未来收益 |

一个专业报告不是只展示好结果，而是让读者知道这个结果该如何被质疑。

## 6. 最终检查清单

提交项目之前检查：

- 数据下载区间清楚。
- 资产列表清楚。
- 信号计算公式清楚。
- 所有信号都延迟使用。
- 权重下一交易日生效。
- 成本已经扣除。
- 有等权基准。
- 有累计收益图。
- 有回撤图。
- 有年度收益表。
- 有换手率分析。
- 结论包含局限性。
- 没有把回测说成未来保证。

## 入门者补充：常见卡点和处理方法

### 数据下载失败

处理：

- 先确认网络。
- 只下载一个 ticker 测试。
- 缩短日期范围。
- 使用 Day 2 保存的 parquet 数据继续。

### 回测结果全是 0

可能原因：

- 信号全是 NaN。
- 权重没有成功生成。
- `daily_target_weights` 没有对齐到 `returns.index`。
- `shift(1)` 后早期数据被清空，但后续没有恢复。

检查：

```python
print(monthly_target_weights.tail())
print(daily_target_weights.tail())
print(bt.tail())
```

### 策略表现远差于基准

这不一定是代码错。可能原因：

- 动量策略在该样本期无效。
- 参数不适合。
- ETF 池太小。
- 交易成本影响。
- 等权基准本身很强。

你要先检查代码和时序，再解释结果。

### 报告不知道怎么写

按这个顺序写：

```text
我研究什么
为什么可能有效
我用了什么数据
我怎么生成信号
我怎么形成组合
我如何扣成本
结果如何
哪里可能错
下一步做什么
```

## 7. 一周后你应该掌握的东西

如果你认真完成了 7 天，你现在应该具备：

- 金融市场基本语言。
- 宏观变量和资产价格关系的直觉。
- Python 金融数据处理能力。
- 收益和风险指标计算能力。
- 组合风险理解。
- 时间序列信号构造能力。
- 因子思维和策略设计能力。
- 基础回测能力。
- 偏差识别能力。
- 研究报告写作能力。

这已经是量化研究入门的完整闭环。

## 8. 后续学习路径

### 路线 A：量化研究

继续学习：

- 资产定价。
- 因子模型。
- 统计学习。
- 横截面回归。
- 投资组合优化。
- 稳健性检验。
- 交易成本建模。

建议项目：

- 多 ETF 动量和趋势组合。
- 股票横截面动量。
- 低波动因子。
- 行业轮动。

### 路线 B：金融工程

继续学习：

- 随机过程。
- Ito 引理。
- Black-Scholes。
- Monte Carlo。
- 利率模型。
- 波动率曲面。

建议工具：

- NumPy。
- SciPy。
- QuantLib。

### 路线 C：量化开发

继续学习：

- 事件驱动回测。
- 数据管道。
- 订单管理。
- 撮合和成交模拟。
- 风险系统。
- 部署和监控。

建议项目：

- 从零写一个事件驱动回测器。
- 实现订单、成交、持仓、账户模块。
- 接入正式行情源。

## 9. 现代工具建议

本周默认工具已经足够：

- `uv`：Python 项目和依赖管理。
- `JupyterLab`：研究 notebook。
- `pandas`：表格和时间序列。
- `NumPy`：数值计算。
- `SciPy`：优化。
- `statsmodels`：统计模型。
- `scikit-learn`：机器学习和时间序列拆分。
- `matplotlib`：基础图表。
- `pyarrow`：Parquet 数据存储。
- `ruff`：代码检查。

暂时不要急着引入复杂框架。你需要先理解回测逻辑，再决定是否使用 vectorbt、Zipline、Backtrader、QuantConnect LEAN 等框架。

## 10. 最终验收标准

今天结束时，你应该有：

- `notebooks/day-07-final-project.ipynb`
- `reports/four-etf-momentum-report.md`
- `reports/four-etf-momentum-performance.csv`
- `reports/four-etf-momentum-yearly-returns.csv`
- 至少三张图：累计收益、回撤、年度收益或权重。
- 一份能被别人复现的策略说明。

如果这些都完成，你已经完成了量化研究入门的第一周。
