# Day 6 - 回测与偏差控制

版本说明：本文档使用从零写的向量化回测。目标是理解回测机制和偏差，而不是依赖黑箱框架。

## 今日目标

今天的目标是把 Day 5 的策略规则变成一个可信的基础回测。

完成今天后，你应该能够：

- 理解回测的基本数据流。
- 正确处理信号、权重、收益和交易成本的时间对齐。
- 计算换手率和成本后收益。
- 识别常见回测偏差。
- 用基准策略比较结果。
- 写出谨慎的研究结论。

## 0. 入门者先理解回测的定位

回测不是“证明策略能赚钱”，而是“检查一个明确规则如果放在历史中会发生什么”。

你可以把回测看成一个模拟器：

```text
输入：历史数据、信号规则、权重规则、成本假设
过程：按时间顺序模拟每一天能知道什么、能交易什么、赚亏什么
输出：收益、风险、回撤、换手、成本、相对基准表现
```

一个可信回测最重要的不是代码复杂，而是时间顺序正确。

你每一步都要问：

```text
这个信息在当时是否已经可得？
这个交易在当时是否能够执行？
这个收益是否来自已经持有的仓位？
这个结果是否扣除了必要成本？
```

如果这四个问题答不清楚，回测结果就不可信。

## 1. 回测是什么

回测是用历史数据模拟策略在过去会如何表现。

一个回测至少包含：

- 数据。
- 信号。
- 持仓权重。
- 交易执行假设。
- 成本和滑点。
- 组合收益。
- 风险指标。
- 基准比较。

回测不是证明未来赚钱。它只回答：

```text
如果历史数据可靠，且我们当时按这些规则交易，结果大概会怎样？
```

## 2. 回测数据流

正确的数据流：

```text
历史价格
  -> 用过去数据计算信号
  -> 信号延迟
  -> 生成目标权重
  -> 权重延迟生效
  -> 用实际资产收益计算组合收益
  -> 扣除交易成本
  -> 计算风险收益指标
```

最容易出错的地方：

- 用当天收盘价计算信号，又假设当天收盘成交。
- 排名时使用未来收益。
- 成本没有扣。
- 股票池包含未来才知道的信息。
- 只展示最好的参数。

### 一个具体时间线

假设今天是 2024-01-31，策略每月调仓。

正确时间线：

```text
2024-01-31 收盘后：获得 2024-01-31 收盘价
2024-01-31 收盘后：计算 252 日动量信号
2024-01-31 收盘后：生成目标权重
2024-02-01：按目标权重成交
2024-02-01 到下次调仓：用持仓权重获得收益
```

错误时间线：

```text
2024-01-31 使用 2024-01-31 收盘价计算信号
2024-01-31 假设当天已经按这个信号持有
2024-01-31 赚取当天收益
```

第二种写法用了未来信息。即使只差一天，也会让回测结果偏乐观。

## 3. 从权重到组合收益

如果当天开盘前已经持有权重 `w_t`，当天资产收益是 `r_t`，组合收益为：

```text
portfolio_return_t = sum_i w_{t,i} * r_{t,i}
```

如果权重是在当天收盘后才生成，则不能用于当天收益。通常做：

```python
realized_weights = target_weights.shift(1)
portfolio_returns = realized_weights.mul(asset_returns, axis=1).sum(axis=1)
```

这是回测纪律的核心。

### `target_weights` 和 `realized_weights`

新手经常混淆两个权重：

| 名称 | 含义 | 用途 |
|---|---|---|
| `target_weights` | 策略希望下一期持有的权重 | 计算调仓和换手 |
| `realized_weights` | 当天实际已经持有的权重 | 计算当天组合收益 |

如果你在收盘后生成目标权重，那么当天收益应该由昨天已经持有的权重决定。

这就是：

```python
realized_weights = target_weights.shift(1)
```

的含义。

## 4. 交易成本和换手率

换手率衡量权重变化幅度：

```text
turnover_t = sum_i abs(w_t,i - w_{t-1,i})
```

如果从全仓 SPY 换到全仓 TLT：

```text
SPY 权重变化：abs(0 - 1) = 1
TLT 权重变化：abs(1 - 0) = 1
turnover = 2
```

这代表卖出 100% 组合，再买入 100% 组合，总成交金额约为组合净值的 200%。

交易成本：

```text
cost_t = turnover_t * cost_rate
```

如果 cost_rate = 0.0005，代表单边成交金额的 0.05%。

### 成本为什么会改变结论

假设一个策略每月平均换手率是 1.0，也就是每月成交金额约等于组合净值的 100%。如果单边成本是 0.05%，那么每月成本约：

```text
1.0 * 0.05% = 0.05%
```

一年约 12 次：

```text
0.05% * 12 = 0.60%
```

如果策略年化超额收益只有 1%，成本已经吃掉大部分优势。

如果换手率更高，或者真实成本高于假设，策略可能从盈利变成亏损。所以任何短周期策略都必须严肃处理成本。

## 5. 实现基础绩效函数

代码：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def annualized_return(daily_returns: pd.Series, periods_per_year: int = 252):
    daily_returns = daily_returns.dropna()
    ending_wealth = (1 + daily_returns).prod()
    n_periods = len(daily_returns)
    return ending_wealth ** (periods_per_year / n_periods) - 1

def annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252):
    return daily_returns.dropna().std() * np.sqrt(periods_per_year)

def drawdown_series(daily_returns: pd.Series):
    wealth = (1 + daily_returns.dropna()).cumprod()
    peaks = wealth.cummax()
    return wealth / peaks - 1

def max_drawdown(daily_returns: pd.Series):
    return drawdown_series(daily_returns).min()

def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252):
    ann_ret = annualized_return(daily_returns, periods_per_year)
    ann_vol = annualized_volatility(daily_returns, periods_per_year)
    if ann_vol == 0:
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

## 6. 实现动量策略权重

代码：

```python
prices = pd.read_parquet("data/processed/etf_prices.parquet")
returns = pd.read_parquet("data/processed/etf_returns.parquet")

assets = ["SPY", "QQQ", "TLT", "GLD"]
prices = prices[assets]
returns = returns[assets]

lookback = 252
top_n = 2

momentum = prices / prices.shift(lookback) - 1
tradable_signal = momentum.shift(1)
monthly_signal = tradable_signal.resample("ME").last()

def top_n_weights(signal_row: pd.Series, n: int = 2):
    weights = pd.Series(0.0, index=signal_row.index)
    valid = signal_row.dropna()
    if len(valid) < n:
        return weights
    winners = valid.nlargest(n).index
    weights.loc[winners] = 1 / n
    return weights

monthly_target_weights = monthly_signal.apply(top_n_weights, axis=1, n=top_n)
daily_target_weights = monthly_target_weights.reindex(returns.index).ffill().fillna(0.0)
```

## 7. 回测函数

这个函数假设：

- `target_weights` 是每天收盘后已经知道的目标权重。
- 下一交易日开始持有这些权重。
- 成本按权重变化乘以成本率估算。

代码：

```python
def backtest_weight_strategy(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_rate: float = 0.0005,
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
    cost_rate=0.0005,
)

bt.head()
```

重要说明：

- 这个模型仍然简化了真实交易。
- 它没有建模买卖价差、市场冲击、订单部分成交、税费、借券成本。
- 对 ETF 日频学习研究已经足够。

## 8. 基准策略

基准是研究的参照物。没有基准，策略收益没有意义。

四 ETF 等权买入持有：

```python
equal_weights = pd.DataFrame(
    1 / len(assets),
    index=returns.index,
    columns=assets,
)

benchmark_returns = equal_weights.mul(returns, axis=1).sum(axis=1)
```

对比：

```python
comparison_returns = pd.DataFrame(
    {
        "momentum_gross": bt["gross_return"],
        "momentum_net": bt["net_return"],
        "equal_weight": benchmark_returns,
    }
).dropna()

performance_summary(comparison_returns)
```

画图：

```python
(1 + comparison_returns).cumprod().plot(figsize=(12, 6), title="Momentum Strategy vs Equal Weight")
plt.grid(True)
plt.show()

drawdowns = comparison_returns.apply(drawdown_series)
drawdowns.plot(figsize=(12, 6), title="Drawdown Comparison")
plt.grid(True)
plt.show()
```

## 9. 换手率分析

```python
monthly_turnover = bt["turnover"].resample("YE").sum()
monthly_turnover.plot(kind="bar", figsize=(12, 5), title="Annual Turnover")
plt.grid(True)
plt.show()

print("Average daily turnover:", bt["turnover"].mean())
print("Average annual turnover:", bt["turnover"].resample("YE").sum().mean())
print("Total cost impact:", bt["cost"].sum())
```

成本不是小事。高换手策略即使毛收益好，也可能在成本后失效。

## 10. 分段表现

全样本结果容易掩盖失效时期。

按年份统计：

```python
yearly_returns = comparison_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
yearly_returns
```

画图：

```python
yearly_returns.plot(kind="bar", figsize=(12, 6), title="Calendar Year Returns")
plt.axhline(0, color="black", linewidth=1)
plt.grid(True)
plt.show()
```

观察：

- 策略是否只靠一两年赚钱？
- 是否在某些市场状态下明显失效？
- 最大回撤发生在哪个阶段？
- 成本后收益是否还明显？

## 11. 常见回测偏差

### 未来函数

使用了当时不可能知道的信息。

例子：

- 用今天收盘价计算信号，又假设今天收盘成交。
- 用全年财报数据回填到年初。
- 用未来成分股列表回测过去。

### 幸存者偏差

只使用现在还存在的资产，忽略历史上退市或失败的资产。

ETF 大类资产项目影响较小，但股票策略影响巨大。

### 数据窥探和过拟合

不断测试参数，最后只选择历史最好的那个。

例子：

- 测试 10、20、30、60、120、252 日窗口，挑最高 Sharpe。
- 调了几十次规则，但报告只说最终版本。

### 多重检验

测试次数越多，偶然发现“有效策略”的概率越高。

如果你测试了 1000 个随机信号，总会有几个看起来很棒。

### 交易成本低估

真实交易会遇到：

- 买卖价差。
- 滑点。
- 冲击成本。
- 佣金和费用。
- 税费。
- 借券成本。

### 流动性忽略

一个策略在小资金下可行，不代表大资金可行。成交量、市场深度和冲击成本会限制容量。

### 参数不稳定

一个策略只在特定参数下有效，稍微改参数就失效，通常说明它不稳健。

## 12. 回测结论的正确写法

不好的结论：

```text
这个策略能赚钱。
```

好的结论：

```text
在 2015-01-01 到 2026-01-01 的 SPY、QQQ、TLT、GLD 日线数据上，252 日横截面动量、月度调仓、持有前 2 名资产的策略，在假设单边交易成本 0.05% 后，相对四 ETF 等权基准表现为：年化收益 X，年化波动 Y，Sharpe Z，最大回撤 W。结果依赖样本区间和数据源，且未考虑税费、真实滑点和数据供应商误差，因此只能作为研究原型，不能作为实盘建议。
```

量化研究的专业性来自这种谨慎表达。

## 入门者补充：回测结果检查清单

每次完成回测后，先不要看收益最高不最高，先检查下面这些问题：

1. 信号是否使用 `shift(1)` 或等价的延迟处理？
2. 权重是否在下一交易日才生效？
3. 收益是否由已经持有的权重计算？
4. 是否扣除了交易成本？
5. 是否有清晰基准？
6. 是否比较了毛收益和净收益？
7. 是否展示了最大回撤？
8. 是否看了年度收益，而不只是全样本结果？
9. 是否计算了换手率？
10. 是否写清楚数据来源和样本区间？

常见误区：

- 只看最终累计收益曲线。
- 忘记成本。
- 用全样本最佳参数当作策略。
- 看到 Sharpe 高就忽略回撤。
- 只和现金比，不和合理基准比。
- 把历史模拟写成未来预测。

## 13. 今日任务

完成以下任务：

1. 实现 `performance_summary`。
2. 实现 `backtest_weight_strategy`。
3. 用 Day 5 的月度动量权重回测策略。
4. 扣除 0.05% 单边交易成本。
5. 构建四 ETF 等权买入持有基准。
6. 比较毛收益、成本后收益和基准。
7. 画累计收益图和回撤图。
8. 计算年度收益。
9. 计算年度换手率。
10. 写一段谨慎结论，必须包含数据区间、成本假设和局限。

## 14. 小测验

1. 为什么权重通常要 `shift(1)` 后再计算收益？
2. 换手率为什么可能大于 1？
3. 毛收益和净收益有什么区别？
4. 未来函数有哪些常见形式？
5. 为什么只看全样本 Sharpe 不够？
6. 为什么回测不能证明未来收益？

## 15. 验收标准

今天结束时，你应该有：

- 一个可复用的基础回测函数。
- 一个成本前后对比表。
- 一个和等权基准比较的图。
- 一个年度收益表。
- 一个换手率分析。
- 一段专业、谨慎的策略结论。

如果你的回测没有扣成本，或者没有信号延迟，不要进入 Day 7。
