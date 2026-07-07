# Day 4 - 时间序列与交易信号

版本说明：本文档使用 pandas 和 statsmodels 做时间序列分析。重点是理解金融时间序列的性质，以及如何把市场假设转成信号。

## 今日目标

今天的目标是从“描述资产表现”进入“构造可检验信号”。

完成今天后，你应该能够：

- 理解平稳性、自相关、趋势和均值回复。
- 使用 ADF 检验和 ACF 观察时间序列结构。
- 构造动量信号、均值回复信号、移动平均信号。
- 明白信号必须延迟使用，避免未来函数。
- 初步理解 ARIMA 模型的用途和局限。

## 0. 入门者先理解时间序列研究在做什么

时间序列研究关注同一个对象随时间变化的结构。

在量化研究里，你通常会问：

```text
过去的信息是否能帮助解释未来收益？
```

这句话里有三个关键点：

- 过去的信息：价格、收益率、成交量、波动率、宏观数据等。
- 未来收益：你真正想预测或排序的目标。
- 是否能帮助：不是保证预测正确，而是统计上是否有一点优势。

金融时间序列很难，因为信号通常弱、噪声通常强。你不能期待像图像识别那样有清晰标签。很多策略的优势可能只是：

```text
长期平均来看，胜率略高或盈亏比略好。
```

这也是为什么交易成本、信号延迟和过拟合会严重影响结果。

## 1. 金融时间序列的特点

金融时间序列和普通工程数据不同。

常见特点：

- 价格通常非平稳。
- 收益率比价格更接近平稳。
- 波动率会聚集，高波动时期常常连续出现。
- 极端收益出现频率高于正态分布假设。
- 相关性随时间变化。
- 市场结构会变化，历史规律可能失效。

量化研究因此更关注：

- 收益率而非价格。
- 滚动统计而非全样本统计。
- 样本外表现而非样本内拟合。
- 简单稳健信号而非复杂但脆弱模型。

### 价格、收益率、波动率的关系

初学者可以这样理解：

```text
价格：资产现在处于什么水平。
收益率：价格变化了多少百分比。
波动率：收益率变化有多剧烈。
```

价格常常像随机游走一样漂移，收益率更适合统计分析，波动率经常成团出现。

“波动率聚集”是金融数据的重要现象。它的意思是：

```text
大波动后面更容易跟着大波动，小波动后面更容易跟着小波动。
```

这就是为什么很多风险分析会使用滚动窗口，而不是假设风险永远不变。

## 2. 平稳性

一个时间序列如果统计性质大致不随时间变化，就叫平稳。

弱平稳通常要求：

- 均值稳定。
- 方差稳定。
- 协方差只和时间间隔有关，不依赖具体日期。

价格通常不是平稳的。收益率往往更接近平稳，但也不是完美平稳。

为什么重要：

- 很多统计模型默认数据平稳。
- 用非平稳价格做回归容易得到伪相关。
- 策略信号应该尽量基于相对变化、收益率、价差或标准化指标。

## 3. ADF 检验

ADF 检验用于测试序列是否存在单位根。

粗略理解：

- 原假设：序列有单位根，非平稳。
- p-value 小于 0.05：倾向拒绝原假设，认为序列更像平稳。
- p-value 大于 0.05：没有足够证据说明平稳。

代码：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf

prices = pd.read_parquet("data/processed/etf_prices.parquet")
returns = pd.read_parquet("data/processed/etf_returns.parquet")

def adf_report(series: pd.Series):
    series = series.dropna()
    stat, pvalue, used_lag, nobs, critical_values, icbest = adfuller(series)
    return pd.Series(
        {
            "adf_statistic": stat,
            "p_value": pvalue,
            "used_lag": used_lag,
            "nobs": nobs,
            "critical_1pct": critical_values["1%"],
            "critical_5pct": critical_values["5%"],
            "critical_10pct": critical_values["10%"],
        }
    )

print(adf_report(prices["SPY"]))
print(adf_report(returns["SPY"]))
```

注意：

- 统计检验不是绝对真理。
- p-value 不能证明策略有效。
- 检验结果会受样本区间影响。

## 4. 自相关

自相关衡量序列和自身滞后值之间的相关性。

如果收益率有显著正自相关，可能意味着趋势延续。

如果收益率有显著负自相关，可能意味着短期反转。

代码：

```python
from statsmodels.tsa.stattools import acf

spy_returns = returns["SPY"].dropna()
acf_values = acf(spy_returns, nlags=20)

pd.Series(acf_values, index=range(21), name="acf")
```

画图：

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(acf_values)), acf_values)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("SPY Return Autocorrelation")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
plt.show()
```

现实中，日频股票收益率的自相关通常很弱。很多看起来显著的模式会被交易成本吃掉。

## 5. 信号是什么

信号是用历史信息计算出的、对未来收益有预测或排序意义的变量。

一个信号应该满足：

- 只使用当时已经可得的信息。
- 有明确经济或行为解释。
- 可以稳定复现。
- 可以转换为持仓规则。
- 能在样本外接受检验。

信号不是策略。策略至少还需要：

- 选股或选资产范围。
- 建仓规则。
- 权重规则。
- 调仓频率。
- 风控规则。
- 交易成本假设。
- 评估方法。

## 6. 动量信号

动量假设：

```text
过去表现强的资产，未来一段时间可能继续表现强。
```

常见原因：

- 信息扩散需要时间。
- 投资者反应不足。
- 资金流追逐表现。
- 趋势跟随行为。

252 日动量：

```python
momentum_252 = prices / prices.shift(252) - 1
momentum_252.tail()
```

关键点：交易时不能使用当天收盘后才知道的信号去买当天收盘。为了避免未来函数，信号至少要延迟一期：

```python
tradable_momentum = momentum_252.shift(1)
```

如果你用日收盘计算信号，并假设下一天执行，那么 `shift(1)` 是基本纪律。

### 为什么必须 `shift(1)`

假设今天是 2024-01-31。

如果你用今天收盘价计算 252 日动量，那么这个信号只有在今天收盘后才知道。你不能假设自己已经在今天收盘前知道它，并用今天的收益赚钱。

正确做法：

```text
今天收盘后计算信号
明天或下一个可交易时点执行
```

在 pandas 里通常写成：

```python
tradable_signal = raw_signal.shift(1)
```

这不是形式主义。很多看起来很好的回测，都是因为没有正确延迟信号。

## 7. 移动平均信号

移动平均常用于趋势跟踪。

规则示例：

```text
如果价格高于 200 日均线，持有资产。
如果价格低于 200 日均线，持有现金。
```

代码：

```python
spy_price = prices["SPY"]
ma_200 = spy_price.rolling(window=200).mean()

signal = (spy_price > ma_200).astype(int)
tradable_signal = signal.shift(1).fillna(0)

strategy_returns = tradable_signal * returns["SPY"]
```

画图：

```python
fig, ax = plt.subplots(figsize=(12, 6))
spy_price.plot(ax=ax, label="SPY")
ma_200.plot(ax=ax, label="200D MA")
ax.set_title("SPY and 200-Day Moving Average")
ax.legend()
ax.grid(True)
plt.show()
```

解释：

- 这个信号试图规避长期下跌趋势。
- 它可能减少大回撤。
- 它也可能在震荡市场频繁误判。

## 8. 均值回复信号

均值回复假设：

```text
价格或收益偏离某个均衡水平后，未来可能回到均值。
```

常用 z-score：

```text
z = (current_value - rolling_mean) / rolling_std
```

示例：

```python
lookback = 20
spy_return = returns["SPY"]
rolling_mean = spy_return.rolling(lookback).mean()
rolling_std = spy_return.rolling(lookback).std()
zscore = (spy_return - rolling_mean) / rolling_std

mean_reversion_signal = (-zscore).clip(-1, 1)
tradable_mr_signal = mean_reversion_signal.shift(1).fillna(0)
mr_strategy_returns = tradable_mr_signal * spy_return
```

解释：

- 如果昨天收益显著为正，今天降低或做空暴露。
- 如果昨天收益显著为负，今天增加暴露。
- 这是短期反转思路，可能非常受交易成本影响。

## 9. 横截面信号和时间序列信号

时间序列信号：

```text
用资产自己的历史判断是否持有它。
```

例子：

- SPY 高于 200 日均线就持有 SPY。
- TLT 过去 12 个月收益为正就持有 TLT。

横截面信号：

```text
在同一时点比较多个资产，买排名靠前的，卖或不买排名靠后的。
```

例子：

- 每月选择过去 12 个月收益最高的两个 ETF。
- 在股票池里买入低估值、高质量、高动量的股票。

本周最终项目建议用横截面动量，因为它直观且容易验证。

## 10. ARIMA 的位置

ARIMA 是经典时间序列模型：

```text
ARIMA(p, d, q)
```

- AR：自回归，使用过去值解释当前值。
- I：差分，让序列更平稳。
- MA：移动平均，使用过去误差解释当前值。

示例：

```python
from statsmodels.tsa.arima.model import ARIMA

series = returns["SPY"].dropna()
model = ARIMA(series, order=(1, 0, 1))
result = model.fit()
print(result.summary())
```

重要提醒：

- ARIMA 可以帮助理解时间序列结构。
- 它不自动等于可交易策略。
- 金融日收益率可预测性通常很弱。
- 预测误差、交易成本和过拟合会让模型表现下降。

## 11. 比较三个信号

创建三个简单策略：

1. SPY 买入持有。
2. SPY 200 日均线趋势策略。
3. SPY 20 日均值回复策略。

代码：

```python
def performance_summary_one(daily_returns: pd.Series):
    daily_returns = daily_returns.dropna()
    wealth = (1 + daily_returns).cumprod()
    ann_ret = wealth.iloc[-1] ** (252 / len(daily_returns)) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    dd = wealth / wealth.cummax() - 1
    return pd.Series(
        {
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": dd.min(),
        }
    )

buy_hold = returns["SPY"]
trend = tradable_signal * returns["SPY"]
mean_reversion = tradable_mr_signal * returns["SPY"]

comparison = pd.DataFrame(
    {
        "buy_hold": performance_summary_one(buy_hold),
        "trend": performance_summary_one(trend),
        "mean_reversion": performance_summary_one(mean_reversion),
    }
).T

comparison
```

画累计收益：

```python
strategy_returns = pd.DataFrame(
    {
        "buy_hold": buy_hold,
        "trend": trend,
        "mean_reversion": mean_reversion,
    }
).dropna()

(1 + strategy_returns).cumprod().plot(figsize=(12, 6), title="Signal Strategy Comparison")
plt.grid(True)
plt.show()
```

不要急着下结论。今天的策略还没有考虑交易成本，也没有做样本外验证。

## 入门者补充：信号研究常见误区

1. 把信号图画得漂亮就认为策略有效。
2. 忘记 `shift(1)`，不小心使用未来信息。
3. 用价格水平直接建模，忽略非平稳性。
4. 看到 ADF p-value 就过度解读。
5. 用同一段数据反复调参数，然后相信最好的结果。
6. 忽略交易成本，尤其是短期均值回复策略。
7. 把 ARIMA 或机器学习预测当成完整交易策略。

信号进入策略前，至少要写清楚：

```text
信号使用什么历史数据？
信号在什么时候可得？
信号预测什么未来目标？
信号为什么可能有效？
信号会带来多少换手？
```

如果这些问题回答不清楚，先不要回测。

## 12. 今日任务

完成以下任务：

1. 对 SPY 价格和收益率分别做 ADF 检验。
2. 画 SPY 日收益率前 20 阶自相关。
3. 构造 SPY 200 日移动平均信号。
4. 构造 SPY 20 日收益率 z-score 均值回复信号。
5. 构造 ETF 横截面 252 日动量信号。
6. 所有信号都必须使用 `shift(1)` 变成可交易信号。
7. 比较买入持有、趋势信号、均值回复信号的初步表现。
8. 写一段文字说明哪个信号看起来更合理，以及为什么不能直接相信结果。

## 13. 小测验

1. 为什么价格通常比收益率更不平稳？
2. ADF 检验的原假设是什么？
3. 什么是自相关？正自相关和负自相关分别可能代表什么？
4. 为什么信号要 `shift(1)`？
5. 动量和均值回复的市场假设有什么不同？
6. ARIMA 为什么不等于交易策略？

## 14. 验收标准

今天结束时，你应该有：

- ADF 检验结果。
- 自相关图。
- 三类信号：动量、趋势、均值回复。
- 至少一个信号策略的初步收益曲线。
- 一段说明未来函数风险的笔记。

如果你在代码里没有明确处理信号延迟，不要进入 Day 5。
