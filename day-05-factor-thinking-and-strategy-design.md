# Day 5 - 因子思维与策略设计

版本说明：本文档使用 pandas、statsmodels 和 scikit-learn 的时间序列拆分思想。目标是把信号升级成可解释、可验证的研究假设。

## 今日目标

今天的目标是学习量化研究的核心工作方式：

```text
市场假设 -> 因子或信号 -> 组合规则 -> 样本外验证 -> 风险解释
```

完成今天后，你应该能够：

- 理解什么是因子。
- 区分风险因子、风格因子和 alpha 信号。
- 用回归估计 Beta 和 Alpha。
- 写出结构化策略假设。
- 使用时间序列拆分做简单样本外验证。
- 形成最终项目的策略设计书。

## 0. 入门者先理解“因子思维”

因子思维是一种把市场现象拆成可解释变量的方法。

不要把因子想成神秘公式。因子更像一个有经济含义的特征：

```text
这个资产有什么特征？
这个特征是否能解释它未来收益的差异？
这个解释是否稳定？
这个解释是否能交易？
```

CS 背景可以把因子理解成机器学习里的 feature，但金融因子多了两个要求：

- 需要经济或行为解释。
- 需要考虑交易成本、风险暴露和市场结构。

一个纯粹在历史数据里碰出来的 feature，即使回测好看，也不一定是好因子。

## 1. 什么是因子

因子是解释资产收益差异的变量。

它可以是：

- 风险来源：市场风险、利率风险、信用风险、通胀风险。
- 风格特征：价值、动量、质量、规模、低波动。
- 行为偏差：过度反应、反应不足、处置效应。
- 结构性约束：再平衡、指数调仓、流动性需求。

因子不是神秘公式。它本质上是一个可解释的变量，用来说明：

```text
为什么某些资产在某些时期表现更好？
```

### 因子的三个层次

| 层次 | 问题 | 例子 |
|---|---|---|
| 描述 | 资产有什么特征？ | 过去 12 个月涨幅高 |
| 解释 | 为什么这个特征可能有用？ | 投资者反应不足导致趋势延续 |
| 交易 | 如何把它转成组合？ | 每月买动量最高的 2 个 ETF |

很多初学者停在第一层：算了一个指标，然后直接回测。更完整的研究要走完三层。

## 2. 风险补偿和错误定价

一个信号能赚钱，通常有两类解释。

### 风险补偿

投资者承担了某种不舒服的风险，因此获得补偿。

例子：

- 小盘股长期可能有更高收益，因为它们流动性差、经营风险高。
- 价值股可能在经济压力期表现差，因此需要风险溢价。
- 趋势策略可能在震荡市场承受连续亏损。

### 错误定价

市场参与者因为行为偏差或信息处理限制，让价格偏离合理水平。

例子：

- 投资者反应不足，导致动量延续。
- 投资者过度恐慌，导致短期反转。
- 分析师覆盖不足，导致小股票信息反映慢。

一个成熟研究员会问：

```text
这个收益是风险补偿，还是市场错误？如果是风险补偿，风险在哪里？如果是错误定价，为什么没有被套利消失？
```

## 3. 常见因子直觉

### 市场因子

市场整体上涨时，大多数股票上涨。市场因子是最基础的风险来源。

CAPM 简化表达：

```text
r_asset - r_f = alpha + beta * (r_market - r_f) + error
```

学习阶段可忽略无风险利率：

```text
r_asset = alpha + beta * r_market + error
```

### 动量

过去表现强的资产未来继续表现强。

可能原因：

- 信息扩散慢。
- 资金追逐赢家。
- 投资者反应不足。

### 价值

价格相对于基本面便宜的资产未来表现更好。

常见指标：

- 市净率。
- 市盈率。
- 企业价值倍数。
- 股息率。

本周不做基本面数据，所以只理解概念。

### 质量

高盈利能力、低杠杆、稳定现金流的公司更优。

常见指标：

- ROE。
- 毛利率。
- 债务率。
- 盈利稳定性。

### 低波动

低波动股票或资产可能有更好的风险调整收益。

可能原因：

- 杠杆限制让投资者追逐高 Beta 资产。
- 低波动资产被忽视。

## 4. Alpha 和 Beta 的回归估计

今天用 SPY 作为市场，用 QQQ 作为资产，估计 QQQ 对 SPY 的 Beta。

代码：

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

returns = pd.read_parquet("data/processed/etf_returns.parquet")

asset = returns["QQQ"].dropna()
market = returns["SPY"].dropna()

reg_data = pd.concat([asset, market], axis=1).dropna()
reg_data.columns = ["asset", "market"]

X = sm.add_constant(reg_data["market"])
y = reg_data["asset"]

model = sm.OLS(y, X)
result = model.fit()
print(result.summary())
```

解释：

- `const` 是日频 Alpha。
- `market` 的系数是 Beta。
- `R-squared` 表示市场收益解释了资产收益波动的多少。
- t 值和 p 值用于判断估计值是否显著，但不能证明未来有效。

年化 Alpha：

```python
daily_alpha = result.params["const"]
annual_alpha = (1 + daily_alpha) ** 252 - 1
annual_alpha
```

注意：

- 日频 Alpha 很小是正常的。
- 回归 Alpha 不是策略 Alpha。
- 统计显著不等于可交易。

### Alpha 和 Beta 的直观解释

Beta 回答：

```text
这个资产有多像市场？
```

Alpha 回答：

```text
扣掉市场影响后，还有没有额外平均收益？
```

例子：

```text
资产 A 年收益 12%
市场年收益 10%
资产 A Beta = 1
```

粗略看，A 比市场多 2%，可能有正 Alpha。

但如果：

```text
资产 B 年收益 12%
市场年收益 10%
资产 B Beta = 1.5
```

B 收益高可能只是因为承担了更多市场风险，不一定有 Alpha。

这就是为什么只看收益率不够，要看风险暴露。

## 5. 信号质量检查

一个信号至少要检查：

- 覆盖率：有多少日期和资产有信号。
- 稳定性：信号是否极端跳动。
- 滞后性：是否只使用过去数据。
- 分布：是否被少数极端值支配。
- 相关性：是否只是市场 Beta 的变体。
- 换手率：是否会导致过高交易成本。

示例：检查 252 日动量信号。

```python
prices = pd.read_parquet("data/processed/etf_prices.parquet")

momentum = prices / prices.shift(252) - 1
tradable_momentum = momentum.shift(1)

signal_summary = pd.DataFrame(
    {
        "mean": tradable_momentum.mean(),
        "std": tradable_momentum.std(),
        "min": tradable_momentum.min(),
        "max": tradable_momentum.max(),
        "missing_rate": tradable_momentum.isna().mean(),
    }
)

signal_summary
```

## 6. 从信号到持仓

一个横截面动量策略：

```text
每月最后一个交易日观察 252 日动量。
选择动量最高的 2 个 ETF。
每个入选 ETF 权重 50%。
下一个交易日开始持有。
每月调仓。
```

这是一个完整但简单的策略规则。

生成月度调仓日期：

```python
month_end_signal = tradable_momentum.resample("ME").last()
month_end_signal.tail()
```

注意：pandas 中月末频率过去常见写法是 `"M"`，现代 pandas 中更推荐明确使用 `"ME"` 表示 month end。

选择前 2 名：

```python
def top_n_weights(signal_row: pd.Series, n: int = 2):
    valid = signal_row.dropna()
    weights = pd.Series(0.0, index=signal_row.index)
    if len(valid) < n:
        return weights
    winners = valid.nlargest(n).index
    weights.loc[winners] = 1 / n
    return weights

monthly_weights = month_end_signal.apply(top_n_weights, axis=1)
monthly_weights.tail()
```

把月度权重扩展到日频：

```python
daily_weights = monthly_weights.reindex(returns.index).ffill().fillna(0)
strategy_returns = daily_weights.shift(1).mul(returns, axis=1).sum(axis=1)
```

这里又用了 `shift(1)`，代表调仓权重下一天才生效。

## 7. 时间序列样本外验证

金融数据不能随机打乱。你不能用未来训练，再去预测过去。

scikit-learn 的 `TimeSeriesSplit` 用于时间有序数据拆分。思想是：

```text
训练集在前，测试集在后。
后面的折叠包含更多历史训练数据。
```

示例：

```python
from sklearn.model_selection import TimeSeriesSplit

data = strategy_returns.dropna().to_frame("strategy_return")

tscv = TimeSeriesSplit(n_splits=5, test_size=252)

fold_rows = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
    train = data.iloc[train_idx]["strategy_return"]
    test = data.iloc[test_idx]["strategy_return"]
    fold_rows.append(
        {
            "fold": fold,
            "train_start": train.index.min(),
            "train_end": train.index.max(),
            "test_start": test.index.min(),
            "test_end": test.index.max(),
            "test_annual_return": (1 + test).prod() ** (252 / len(test)) - 1,
            "test_volatility": test.std() * np.sqrt(252),
        }
    )

fold_summary = pd.DataFrame(fold_rows)
fold_summary
```

这不是完整机器学习流程，只是让你建立纪律：任何策略都要看不同时间段的表现。

### 为什么不能随机切分

机器学习里常见随机训练测试切分，但金融时间序列不能随便这样做。

原因：

- 未来市场状态可能泄漏到训练集。
- 时间顺序本身包含信息。
- 市场制度和宏观环境会变化。
- 随机切分会让训练集和测试集过于相似。

正确原则：

```text
永远用过去训练或设计，用未来验证。
```

## 8. 策略设计模板

以后每个策略都用这个模板：

```text
策略名称：

研究假设：

经济或行为解释：

投资范围：

数据：

信号计算：

调仓频率：

持仓规则：

风险控制：

交易成本假设：

基准：

评估指标：

样本内区间：

样本外区间：

可能失效原因：
```

一个合格示例：

```text
策略名称：四 ETF 横截面动量

研究假设：过去 12 个月表现更强的大类资产，未来 1 个月仍可能表现更强。

经济或行为解释：趋势可能来自宏观信息的缓慢扩散、资金流动和投资者反应不足。

投资范围：SPY, QQQ, TLT, GLD。

数据：2015-01-01 到 2026-01-01 的日线自动调整价格。

信号计算：252 日简单收益率，信号延迟 1 个交易日。

调仓频率：每月月末生成信号，下一交易日生效。

持仓规则：选择动量最高的 2 个 ETF，各 50%。如果有效信号不足 2 个，则持有现金。

风险控制：不做空，不使用杠杆，单资产权重不超过 50%。

交易成本假设：每次成交金额 0.05%。

基准：四 ETF 等权买入持有。

评估指标：年化收益、年化波动率、Sharpe、最大回撤、换手率。

样本内区间：2015-01-01 到 2020-12-31。

样本外区间：2021-01-01 到 2026-01-01。

可能失效原因：趋势反转、交易成本上升、资产相关性变化、样本期特殊。
```

### 不合格策略假设示例

```text
我想找一个高 Sharpe 策略。
```

问题：

- 没有市场假设。
- 没有投资范围。
- 没有信号定义。
- 没有交易规则。
- 没有成本假设。

更好的写法：

```text
我想检验大类资产横截面动量是否存在。具体做法是每月比较 ETF 过去 252 日收益，买入排名前 2 的资产，并和等权基准比较成本后表现。
```

策略设计的价值是让你先定义问题，再写代码。否则你很容易在 notebook 里不断调参数，最后忘了自己原本要研究什么。

## 入门者补充：因子研究常见误区

1. 把任何技术指标都叫因子，但没有解释。
2. 只看策略收益，不检查因子是否有稳定排序能力。
3. 把高 Beta 误认为 Alpha。
4. 用未来数据选参数，再说是样本外表现。
5. 忽略信号缺失率和换手率。
6. 策略设计书写得太模糊，导致回测可以随意解释。
7. 统计显著后直接认为可交易，忽略成本和容量。

## 9. 今日任务

完成以下任务：

1. 用 statsmodels 估计 QQQ 相对 SPY 的 Beta 和 Alpha。
2. 用同样方法估计 TLT 和 GLD 相对 SPY 的 Beta。
3. 构造 252 日横截面动量信号。
4. 检查信号的均值、标准差、缺失率和极值。
5. 写出四 ETF 横截面动量策略设计书。
6. 实现月度选择前 2 名 ETF 的权重表。
7. 使用 `TimeSeriesSplit` 或手动时间切分，观察不同年份策略收益。
8. 写一段文字解释：这个策略收益如果存在，可能来自风险补偿还是错误定价？

## 10. 小测验

1. 因子和普通技术指标有什么区别？
2. 风险补偿和错误定价的区别是什么？
3. 回归中的 Alpha 是否等于可交易 Alpha？
4. 为什么金融样本外验证不能随机打乱数据？
5. 为什么策略设计必须写交易成本假设？
6. 为什么一个经济解释很漂亮的信号仍然可能无效？

## 11. 验收标准

今天结束时，你应该有：

- 一张 ETF Beta/Alpha 回归结果表。
- 一个 252 日动量信号表。
- 一个月度目标权重表。
- 一个策略设计书。
- 一个样本外或分段表现表。

如果你还不能把信号清楚地转成持仓规则，不要进入 Day 6。
