# Day 3 - 投资组合与风险

版本说明：本文档使用 pandas、NumPy 和 SciPy 做组合分析。目标是理解组合风险，而不是追求复杂优化器。

## 今日目标

今天的目标是理解为什么“资产组合”不是几个资产的简单相加。

完成今天后，你应该能够：

- 计算组合收益率。
- 理解权重、协方差矩阵和相关性矩阵。
- 解释分散化为什么有效以及什么时候失效。
- 构建等权组合、逆波动率组合、最小方差组合。
- 理解均值方差优化的优点和脆弱性。
- 用滚动窗口估计组合风险。

## 0. 入门者先理解组合的核心直觉

投资组合的核心不是“多买几个资产”，而是“把不同风险来源放在一起”。

如果你持有 4 个资产，但它们在危机中总是一起跌，那么你并没有真正分散风险。真正有用的分散化来自：

- 收益来源不同。
- 对宏观变量敏感性不同。
- 相关性不总是接近 1。
- 某些资产在其他资产下跌时能提供缓冲。

可以用一个系统工程类比：

```text
单资产投资 = 系统只有一个服务，服务挂了整个系统挂。
组合投资 = 系统有多个服务，某个服务挂了不一定全挂。
相关性 = 这些服务是否依赖同一个故障源。
```

但金融里要小心：平时看起来独立的资产，在危机中可能突然变得高度相关。所以组合研究不能只看全样本平均相关性，还要看滚动相关性和压力时期表现。

今天先只研究最基础的组合约束：

```text
每个权重 >= 0
所有权重加起来 = 1
不做空
不使用杠杆
```

这叫 long-only、fully invested 组合。它不是唯一的组合形式，但最适合入门。

## 1. 组合的基本定义

假设你有 4 个资产，权重为：

```text
SPY: 25%
QQQ: 25%
TLT: 25%
GLD: 25%
```

权重向量：

```text
w = [0.25, 0.25, 0.25, 0.25]
```

如果每天资产收益率是：

```text
r = [0.01, 0.02, -0.005, 0.003]
```

组合收益率：

```text
portfolio_return = w_1*r_1 + w_2*r_2 + w_3*r_3 + w_4*r_4
```

矩阵写法：

```text
r_p = w^T r
```

代码：

```python
import numpy as np
import pandas as pd

returns = pd.read_parquet("data/processed/etf_returns.parquet")

weights = pd.Series(
    [0.25, 0.25, 0.25, 0.25],
    index=["SPY", "QQQ", "TLT", "GLD"],
)

portfolio_returns = returns[weights.index].mul(weights, axis=1).sum(axis=1)
portfolio_returns.head()
```

## 2. 组合风险不是加权平均风险

组合收益是资产收益的加权平均，但组合风险不是资产风险的简单加权平均。

组合方差：

```text
portfolio_variance = w^T Sigma w
```

其中 `Sigma` 是收益率协方差矩阵。

如果两个资产不完全同涨同跌，组合波动率可以低于单个资产波动率的加权平均。这就是分散化。

代码：

```python
cov_daily = returns.cov()
cov_annual = cov_daily * 252

portfolio_variance = weights.T @ cov_annual.loc[weights.index, weights.index] @ weights
portfolio_volatility = np.sqrt(portfolio_variance)

portfolio_volatility
```

### 两资产数值例子

假设有两个资产 A 和 B：

```text
A 年化波动率 = 20%
B 年化波动率 = 20%
A 和 B 权重都是 50%
```

如果 A 和 B 相关性是 1，它们完全同涨同跌，组合波动率仍然是 20%。

如果相关性是 0，它们没有明显线性关系，组合波动率约为 14.14%。

如果相关性是 -1，它们完全反向波动，理论上组合波动率可以降到 0。

现实中几乎没有长期稳定的 -1 相关资产，但这个例子说明：降低组合风险的关键不是资产数量，而是资产之间不要完全同向波动。

## 3. 协方差和相关性

协方差衡量两个资产共同变动的绝对程度。相关性是标准化后的协方差。

```text
correlation(i, j) = covariance(i, j) / (vol_i * vol_j)
```

直觉：

- 协方差受资产波动率大小影响。
- 相关性更适合比较共同变动方向和强度。
- 优化组合时使用协方差矩阵，因为组合方差需要绝对风险尺度。

代码：

```python
corr = returns.corr()
cov = returns.cov()

print(corr)
print(cov)
```

## 4. 等权组合

等权组合是最好的基准之一。它简单、透明、很难被复杂模型稳定击败。

代码：

```python
def equal_weight_returns(daily_returns: pd.DataFrame):
    n_assets = daily_returns.shape[1]
    weights = pd.Series(1 / n_assets, index=daily_returns.columns)
    return daily_returns.mul(weights, axis=1).sum(axis=1), weights

ew_returns, ew_weights = equal_weight_returns(returns[["SPY", "QQQ", "TLT", "GLD"]])
ew_weights
```

计算绩效：

```python
def annualized_return(daily_returns, periods_per_year=252):
    ending_wealth = (1 + daily_returns.dropna()).prod()
    n_periods = daily_returns.dropna().shape[0]
    return ending_wealth ** (periods_per_year / n_periods) - 1

def annualized_volatility(daily_returns, periods_per_year=252):
    return daily_returns.dropna().std() * np.sqrt(periods_per_year)

def max_drawdown(daily_returns):
    wealth = (1 + daily_returns.dropna()).cumprod()
    peaks = wealth.cummax()
    return (wealth / peaks - 1).min()

def sharpe_ratio(daily_returns, risk_free_rate=0.0, periods_per_year=252):
    return (annualized_return(daily_returns, periods_per_year) - risk_free_rate) / annualized_volatility(daily_returns, periods_per_year)

pd.Series(
    {
        "annual_return": annualized_return(ew_returns),
        "annual_volatility": annualized_volatility(ew_returns),
        "sharpe": sharpe_ratio(ew_returns),
        "max_drawdown": max_drawdown(ew_returns),
    }
)
```

## 5. 逆波动率组合

逆波动率组合给低波动资产更高权重，给高波动资产更低权重。

公式：

```text
raw_weight_i = 1 / volatility_i
weight_i = raw_weight_i / sum(raw_weight)
```

代码：

```python
def inverse_volatility_weights(daily_returns: pd.DataFrame, lookback: int = 252):
    vol = daily_returns.tail(lookback).std() * np.sqrt(252)
    inv_vol = 1 / vol
    return inv_vol / inv_vol.sum()

iv_weights = inverse_volatility_weights(returns[["SPY", "QQQ", "TLT", "GLD"]])
iv_returns = returns[iv_weights.index].mul(iv_weights, axis=1).sum(axis=1)

iv_weights
```

优点：

- 简单。
- 不需要预期收益估计。
- 比均值方差优化更稳健。

缺点：

- 没有考虑相关性。
- 低波动不等于低风险。
- 在危机中相关性可能上升。

## 6. 最小方差组合

最小方差组合只使用协方差矩阵，不使用预期收益。

目标：

```text
minimize w^T Sigma w
subject to sum(w) = 1
           w_i >= 0
```

这是 long-only、fully invested 的组合。它不允许做空，不允许杠杆。

代码：

```python
from scipy.optimize import minimize

def minimum_variance_weights(daily_returns: pd.DataFrame):
    assets = daily_returns.columns
    cov = daily_returns.cov().values * 252
    n = len(assets)
    initial = np.repeat(1 / n, n)

    def portfolio_variance(w):
        return w.T @ cov @ w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(
        portfolio_variance,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return pd.Series(result.x, index=assets)

mv_weights = minimum_variance_weights(returns[["SPY", "QQQ", "TLT", "GLD"]])
mv_returns = returns[mv_weights.index].mul(mv_weights, axis=1).sum(axis=1)

mv_weights
```

## 7. 均值方差优化为什么脆弱

均值方差优化的经典目标：

```text
maximize expected_return - risk_aversion * variance
```

或者寻找给定风险下收益最高的组合。

问题是：预期收益很难估计。

如果你用历史平均收益当未来预期收益，优化器会过度相信样本里表现最好的资产，给出极端权重。这通常不是智慧，而是估计误差被放大。

初学阶段建议：

- 把等权作为强基准。
- 使用逆波动率和最小方差作为稳健替代。
- 避免过早追求复杂最优组合。
- 对任何优化结果设置权重上限。

## 8. 有效前沿的直觉

有效前沿是一组“在给定风险下预期收益最高，或在给定收益下风险最低”的组合。

你可以用随机权重模拟直观看到它：

```python
def simulate_random_portfolios(daily_returns: pd.DataFrame, n_portfolios: int = 5000):
    assets = daily_returns.columns
    mean_returns = daily_returns.mean() * 252
    cov = daily_returns.cov() * 252
    rows = []

    for _ in range(n_portfolios):
        raw = np.random.random(len(assets))
        weights = raw / raw.sum()
        weights = pd.Series(weights, index=assets)
        ann_ret = weights @ mean_returns
        ann_vol = np.sqrt(weights.T @ cov @ weights)
        sharpe = ann_ret / ann_vol
        row = {
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
        }
        row.update({f"weight_{asset}": weights[asset] for asset in assets})
        rows.append(row)

    return pd.DataFrame(rows)

sim = simulate_random_portfolios(returns[["SPY", "QQQ", "TLT", "GLD"]])
sim.head()
```

画图：

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    sim["annual_volatility"],
    sim["annual_return"],
    c=sim["sharpe"],
    cmap="viridis",
    s=8,
)
fig.colorbar(scatter, ax=ax, label="Sharpe")
ax.set_xlabel("Annual Volatility")
ax.set_ylabel("Annual Return")
ax.set_title("Random Portfolios")
ax.grid(True)
plt.show()
```

注意：这里使用历史平均收益作为预期收益，只是为了学习有效前沿形状，不代表未来可实现。

## 9. 滚动组合风险

风险会随时间变化。计算等权组合 60 日滚动波动率：

```python
ew_rolling_vol = ew_returns.rolling(60).std() * np.sqrt(252)

ew_rolling_vol.plot(figsize=(12, 5), title="Equal-Weight Portfolio 60-Day Rolling Volatility")
plt.grid(True)
plt.show()
```

滚动相关性示例，观察 SPY 和 TLT：

```python
rolling_corr = returns["SPY"].rolling(252).corr(returns["TLT"])
rolling_corr.plot(figsize=(12, 5), title="SPY-TLT 252-Day Rolling Correlation")
plt.axhline(0, color="black", linewidth=1)
plt.grid(True)
plt.show()
```

这一步很重要。很多策略在历史上看似分散，危机中却因为相关性上升而同时亏损。

## 入门者补充：组合优化最容易误解的地方

优化器没有金融常识。它只会机械地最小化或最大化你给它的目标函数。

如果某个资产在样本期里刚好收益高、波动低、相关性好，优化器可能给它非常高的权重。但这可能只是历史偶然。初学者看到优化结果时要问：

- 权重是否过度集中？
- 结果是否依赖某个资产的历史表现？
- 换一个时间段，权重是否完全不同？
- 加入交易成本后是否仍然好？
- 等权组合是否已经差不多？

常见误区：

1. 以为资产数量越多，风险一定越低。实际上如果资产高度相关，数量增加帮助有限。
2. 以为低波动资产永远安全。低波动资产也可能有利率风险、流动性风险或估值风险。
3. 以为优化器输出就是最佳答案。优化器只对输入数据和目标函数负责。
4. 混淆资本权重和风险贡献。权重 25% 不代表贡献 25% 风险。
5. 用全样本协方差预测未来。相关性和波动率都会变化。

你今天的判断标准很简单：复杂组合如果不能明显改善等权组合，就不要急着相信复杂性。

## 10. 今日任务

完成以下任务：

1. 用 Day 2 保存的 ETF 收益率数据。
2. 构建 SPY、QQQ、TLT、GLD 的等权组合。
3. 构建逆波动率组合，使用最近 252 个交易日估计波动率。
4. 构建 long-only 最小方差组合。
5. 比较三个组合的年化收益、年化波动率、Sharpe 和最大回撤。
6. 画出三个组合的累计财富曲线。
7. 画出等权组合的 60 日滚动波动率。
8. 画出 SPY 和 TLT 的 252 日滚动相关性。
9. 写一段结论：复杂组合是否明显优于等权组合？如果是，可能原因是什么？如果不是，说明了什么？

## 11. 小测验

1. 为什么组合风险不是单个资产风险的加权平均？
2. 协方差矩阵在组合风险计算中起什么作用？
3. 为什么均值方差优化对预期收益估计很敏感？
4. 逆波动率组合有什么缺点？
5. 如果两个资产平时负相关，危机时突然正相关，会发生什么？
6. 为什么等权组合是一个强基准？

## 12. 验收标准

今天结束时，你应该有：

- 三个组合的收益率序列。
- 三个组合的绩效对比表。
- 一张累计财富曲线图。
- 一张滚动波动率图。
- 一张滚动相关性图。
- 一段能解释组合风险来源的文字。

如果你还不能解释 `w^T Sigma w` 的含义，先不要进入 Day 4。
