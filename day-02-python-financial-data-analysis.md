# Day 2 - Python 金融数据分析

版本说明：本文档使用现代 Python 研究工作流：`uv` 管理项目，JupyterLab 做探索，pandas 做时间序列，yfinance 仅用于学习和教育用途。

## 今日目标

今天的目标是建立量化研究的第一个技术闭环：

```text
获取价格数据 -> 清洗数据 -> 计算收益率 -> 计算风险指标 -> 可视化 -> 写出观察
```

完成今天后，你应该能够：

- 使用 `uv` 创建和运行研究环境。
- 使用 `yfinance.download` 获取多个资产的日线数据。
- 理解调整后价格和普通收盘价的区别。
- 用 pandas 计算日收益率、累计收益率、年化收益、年化波动率、最大回撤、Sharpe。
- 画出价格、累计收益和回撤图。
- 形成一份最小数据分析报告。

## 0. 入门者先建立三个心智模型

今天你会写很多 pandas 代码。不要先背 API，先建立三个心智模型。

### 心智模型一：DataFrame 是带标签的二维表

可以把 `DataFrame` 理解成 Excel 表格，但更适合程序处理。

```text
index: 通常是日期
columns: 通常是资产代码或字段名
values: 价格、收益率、权重或指标
```

例如：

```text
date        SPY     QQQ     TLT
2024-01-02  475.1   408.2   94.3
2024-01-03  472.8   403.9   95.1
```

这里 `date` 是 index，`SPY`、`QQQ`、`TLT` 是 columns。

### 心智模型二：金融时间序列必须对齐日期

金融数据经常有缺失：

- 某个资产上市较晚。
- 某个市场假期不同。
- 数据源临时缺数据。
- 下载失败导致某些字段为空。

pandas 会按 index 自动对齐。这个特性很强，但也容易让你在不知不觉中产生 NaN。每次计算前后都要检查：

```python
prices.isna().sum()
returns.isna().sum()
```

### 心智模型三：先让数据可信，再让模型复杂

初学者常想尽快跑策略，但专业流程应该是：

```text
下载数据
检查数据形状
检查缺失值
检查日期范围
检查价格是否合理
计算收益率
检查收益率异常值
再做指标和策略
```

如果数据错了，后面所有模型都只是更快地产生错误结论。

## 1. 今日项目结构

建议在 Day 1 创建的项目里工作：

```text
quant-research-lab/
  pyproject.toml
  uv.lock
  notebooks/
    day-02-data-analysis.ipynb
  data/
    raw/
    processed/
  reports/
```

创建目录：

```bash
mkdir -p notebooks data/raw data/processed reports
```

启动 JupyterLab：

```bash
uv run jupyter lab
```

如果还没有安装依赖：

```bash
uv add numpy pandas matplotlib scipy statsmodels scikit-learn yfinance jupyterlab ipykernel pyarrow ruff
```

## 2. 金融数据的基本形态

日线价格数据通常包含：

- `Open`：开盘价。
- `High`：最高价。
- `Low`：最低价。
- `Close`：收盘价。
- `Adj Close` 或自动调整后的价格：考虑分红、拆股等公司行为后的价格。
- `Volume`：成交量。

研究收益率时，要特别注意公司行为：

- 拆股会让价格突然变小，但投资者财富没有相同比例缩水。
- 分红会让除息日价格下降，但投资者收到现金。
- 只用未调整收盘价可能错误估计长期收益。

现代 yfinance 的 `download` 支持 `auto_adjust=True`，这会自动调整 OHLC 价格。学习阶段可以使用它，但你要记住：数据供应商和调整规则会影响研究结果。

## 3. 数据源边界

本周使用 yfinance，因为它简单、免费、适合学习。

但必须记住：

- yfinance 不是 Yahoo 官方产品。
- 它适合研究和教育用途。
- Yahoo 数据权利需要遵守 Yahoo 的使用条款。
- 不应把它当作生产级实盘交易数据源。

真实机构研究常用更正式的数据源，例如交易所数据、Refinitiv、Bloomberg、FactSet、CRSP、Compustat、Polygon、Databento、Interactive Brokers 等。那些数据源通常收费，且有清晰授权。

## 4. 下载多资产价格

今天先研究 4 个 ETF：

- SPY：美国大盘股票。
- QQQ：美国科技成长风格。
- TLT：长期美国国债。
- GLD：黄金。

代码：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

tickers = ["SPY", "QQQ", "TLT", "GLD"]

raw = yf.download(
    tickers=tickers,
    start="2015-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=True,
    progress=False,
)

raw.head()
```

yfinance 下载多个 ticker 时，常见返回格式是多层列索引。我们只取调整后的收盘价格。在 `auto_adjust=True` 时，`Close` 通常已经是调整后的收盘价。

```python
prices = raw["Close"].copy()
prices = prices.dropna(how="all")
prices.tail()
```

检查数据：

```python
print(prices.info())
print(prices.isna().sum())
print(prices.describe())
```

## 5. 价格、收益率和累计收益

价格序列通常不可直接比较。SPY 价格 500、GLD 价格 200，并不代表 SPY 更“贵”。收益率才是可比较对象。

日收益率：

```python
returns = prices.pct_change().dropna(how="all")
returns.head()
```

累计财富曲线：

```python
wealth = (1 + returns).cumprod()
wealth.plot(figsize=(12, 6), title="Growth of $1")
plt.grid(True)
plt.show()
```

解释：

- 如果财富曲线最终到 2.5，代表 1 美元变成 2.5 美元。
- 它不是价格图，而是收益累积图。
- 这使不同价格水平的资产可以放在一起比较。

## 6. 年化收益率

平均日收益率不能直接代表年收益。更好的做法是用几何年化收益率。

公式：

```text
annualized_return = ending_wealth ** (252 / number_of_days) - 1
```

代码：

```python
def annualized_return(daily_returns: pd.Series | pd.DataFrame, periods_per_year: int = 252):
    daily_returns = daily_returns.dropna()
    ending_wealth = (1 + daily_returns).prod()
    n_periods = daily_returns.shape[0]
    return ending_wealth ** (periods_per_year / n_periods) - 1

ann_ret = annualized_return(returns)
ann_ret
```

注意：

- 用简单平均日收益乘以 252 会忽略复利效应。
- 几何年化收益更适合报告实际投资结果。

## 7. 年化波动率

代码：

```python
def annualized_volatility(daily_returns: pd.Series | pd.DataFrame, periods_per_year: int = 252):
    return daily_returns.std() * np.sqrt(periods_per_year)

ann_vol = annualized_volatility(returns)
ann_vol
```

解释：

- 日收益率标准差衡量每日波动。
- 乘以 `sqrt(252)` 是在独立同分布近似下的年化。
- 真实市场有波动聚集和尾部风险，所以这是近似，不是真理。

## 8. Sharpe Ratio

学习阶段先使用无风险利率为 0 的简化版本：

```python
def sharpe_ratio(daily_returns: pd.Series | pd.DataFrame, risk_free_rate: float = 0.0, periods_per_year: int = 252):
    ann_ret = annualized_return(daily_returns, periods_per_year)
    ann_vol = annualized_volatility(daily_returns, periods_per_year)
    return (ann_ret - risk_free_rate) / ann_vol

sr = sharpe_ratio(returns)
sr
```

严格研究中，无风险利率应使用同币种、同期限的短端利率，并按频率转换。

## 9. 最大回撤

函数：

```python
def drawdown(daily_returns: pd.Series | pd.DataFrame):
    wealth = (1 + daily_returns).cumprod()
    peaks = wealth.cummax()
    dd = wealth / peaks - 1
    return dd

dd = drawdown(returns)
max_dd = dd.min()
max_dd
```

画图：

```python
dd.plot(figsize=(12, 6), title="Drawdown")
plt.grid(True)
plt.show()
```

解释：

- 最大回撤越接近 0 越好。
- -0.30 表示从高点最大跌了 30%。
- 回撤是路径相关风险，不能只靠收益率分布看出来。

## 10. 汇总表

创建一个研究中常用的绩效表：

```python
def performance_summary(daily_returns: pd.DataFrame, periods_per_year: int = 252):
    summary = pd.DataFrame(index=daily_returns.columns)
    summary["annual_return"] = annualized_return(daily_returns, periods_per_year)
    summary["annual_volatility"] = annualized_volatility(daily_returns, periods_per_year)
    summary["sharpe"] = sharpe_ratio(daily_returns, periods_per_year=periods_per_year)
    summary["max_drawdown"] = drawdown(daily_returns).min()
    summary["best_day"] = daily_returns.max()
    summary["worst_day"] = daily_returns.min()
    return summary.sort_values("sharpe", ascending=False)

summary = performance_summary(returns)
summary
```

格式化显示：

```python
display(
    summary.style.format(
        {
            "annual_return": "{:.2%}",
            "annual_volatility": "{:.2%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.2%}",
            "best_day": "{:.2%}",
            "worst_day": "{:.2%}",
        }
    )
)
```

## 11. 相关性

相关性矩阵：

```python
corr = returns.corr()
corr
```

可视化：

```python
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.index)
fig.colorbar(im, ax=ax)
ax.set_title("Return Correlation")
plt.show()
```

解释：

- 股票类 ETF 之间通常相关性较高。
- 股票和长期国债的相关性可能随宏观环境变化。
- 黄金的相关性也不是固定的。
- 相关性不是常数，后续策略不能假设它永远稳定。

## 12. 滚动指标

静态全样本指标会掩盖时间变化。滚动窗口可以观察指标如何变化。

60 日滚动波动率：

```python
rolling_vol = returns.rolling(window=60).std() * np.sqrt(252)
rolling_vol.plot(figsize=(12, 6), title="60-Day Rolling Annualized Volatility")
plt.grid(True)
plt.show()
```

252 日滚动收益：

```python
rolling_return = (1 + returns).rolling(window=252).apply(np.prod, raw=True) - 1
rolling_return.plot(figsize=(12, 6), title="252-Day Rolling Return")
plt.grid(True)
plt.show()
```

现代 pandas 推荐使用：

```python
series.rolling(window=60).mean()
```

不要使用早期已经淘汰的 `pd.rolling_mean` 这类旧接口。

## 13. 保存数据

研究要可复现。保存清洗后的价格和收益率：

```python
prices.to_parquet("data/processed/etf_prices.parquet")
returns.to_parquet("data/processed/etf_returns.parquet")
summary.to_csv("reports/day-02-performance-summary.csv")
```

读取：

```python
loaded_returns = pd.read_parquet("data/processed/etf_returns.parquet")
loaded_returns.head()
```

为什么用 Parquet：

- 保留列类型和索引信息更好。
- 读写速度快。
- 适合中大型表格数据。

CSV 适合人类检查，Parquet 更适合程序使用。

## 14. 常见报错和调试方法

### 问题一：下载结果是空的

可能原因：

- ticker 写错。
- 网络失败。
- 日期区间没有数据。
- 数据源临时不可用。

检查：

```python
print(raw.shape)
print(raw.tail())
```

处理：

- 先只下载一个 ticker 测试。
- 缩短日期区间。
- 稍后重试。

### 问题二：`raw["Close"]` 报错

可能原因：

- yfinance 返回结构和你预期不同。
- 单个 ticker 和多个 ticker 的列结构不同。

检查：

```python
print(raw.columns)
```

如果只下载一个 ticker，可能直接有 `Close` 列；如果下载多个 ticker，通常是多层列索引。

### 问题三：收益率里有很多 NaN

第一行 NaN 是正常的，因为没有前一天价格。其他 NaN 可能来自缺失价格。

检查：

```python
returns.isna().sum()
prices.loc[prices.isna().any(axis=1)].head()
```

处理原则：

- 不要盲目 `fillna(0)`。
- 价格缺失要先理解原因。
- 组合计算时可以只在有效资产上计算，后续会学更严格方法。

### 问题四：图看起来很奇怪

可能原因：

- 画的是价格，不是累计收益。
- 某个资产数据缺失或异常。
- 日期 index 不是 DatetimeIndex。

检查：

```python
print(type(prices.index))
print(prices.describe())
```

如果 index 不是日期：

```python
prices.index = pd.to_datetime(prices.index)
```

## 15. 指标解释模板

以后每次生成绩效表，都用下面模板解释，不要只贴表格。

```text
年化收益：说明长期复合增长速度。
年化波动率：说明收益过程有多不稳定。
Sharpe：说明单位波动获得多少收益。
最大回撤：说明从历史高点到低点最痛苦亏损。
最好单日和最差单日：说明尾部单日波动。
相关性：说明资产之间是否一起涨跌。
```

写报告时可以这样表达：

```text
在 2015 到 2026 的样本内，QQQ 的年化收益较高，但波动和最大回撤也更高。TLT 的收益路径和股票 ETF 不完全同步，因此可能提供分散化价值。不过这种关系会随利率环境变化，不应假设未来稳定。
```

## 16. 今日任务

完成以下任务：

1. 下载 SPY、QQQ、TLT、GLD 从 2015-01-01 到 2026-01-01 的日线数据。
2. 提取自动调整后的 `Close` 价格。
3. 计算日收益率。
4. 计算每个资产的年化收益、年化波动率、Sharpe、最大回撤。
5. 画出累计财富曲线。
6. 画出最大回撤曲线。
7. 画出相关性矩阵。
8. 保存清洗后的价格、收益率和绩效表。
9. 写一段 300 字以内的观察：哪个资产收益最高，哪个风险最高，哪个回撤最大，资产之间相关性如何。

## 17. 小测验

1. 为什么长期收益率应该用几何年化，而不是简单平均日收益乘以 252？
2. 为什么使用调整后价格？
3. 相关性低是否一定代表组合风险低？
4. 为什么最大回撤是路径相关指标？
5. yfinance 的 `auto_adjust=True` 有什么作用？
6. 为什么研究要保存中间数据？

## 18. 验收标准

今天结束时，你应该有：

- `notebooks/day-02-data-analysis.ipynb`
- `data/processed/etf_prices.parquet`
- `data/processed/etf_returns.parquet`
- `reports/day-02-performance-summary.csv`
- 三张图：累计财富、回撤、相关性。
- 一段文字解释你的观察。

如果你只能跑出代码，但无法解释每个指标的含义，先复习 Day 1。
