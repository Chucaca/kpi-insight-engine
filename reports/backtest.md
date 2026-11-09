# Month-end forecast backtest (synthetic data)

Rolling origin: last 12 complete month-ends x 4 verticals x origins on day 7/14/21 (144 forecasts per KPI). MAPE of the month-end value.

| KPI | Type | Model MAPE | Naive MAPE (baseline) | Seasonal-naive MAPE | Improvement vs naive | Improvement vs best baseline |
|---|---|---:|---:|---:|---:|---:|
| active_merchant | absolute | 0.48% | 2.68% (last value) | 0.65% | +82% | +26% |
| paying_merchant | absolute | 0.19% | 0.36% (last value) | 0.44% | +48% | +48% |
| active_rate | rate | 0.44% | 2.63% (last value) | 0.42% | +83% | -5% |
| new_merchant | accumulate | 2.71% | 3.60% (MTD run-rate) | 3.60% | +25% | +25% |
| expiry_merchant | accumulate | 1.78% | 2.43% (MTD run-rate) | 2.43% | +27% | +27% |
| renewed_merchant | accumulate | 2.86% | 4.27% (MTD run-rate) | 4.27% | +33% | +33% |
| renewal_ratio | rate_accumulate | 3.31% | 4.65% (MTD ratio) | 4.65% | +29% | +29% |
| paying_merchant (formula: current + new - expiry + renewed) | formula | 0.16% | 0.36% (last value) | 0.44% | +55% | +55% |

Seasonal naive = last observed value on the same weekday as the month-end (identical to the run-rate for flow KPIs).

## MAPE by forecast origin (day of month)

| KPI | day 7 | day 14 | day 21 |
|---|---:|---:|---:|
| active_merchant | 0.62% | 0.49% | 0.32% |
| paying_merchant | 0.24% | 0.19% | 0.13% |
| active_rate | 0.56% | 0.45% | 0.32% |
| new_merchant | 4.08% | 2.61% | 1.43% |
| expiry_merchant | 2.41% | 1.52% | 1.40% |
| renewed_merchant | 3.25% | 3.24% | 2.09% |
| renewal_ratio | 3.74% | 3.55% | 2.63% |
| paying_merchant__formula | 0.22% | 0.15% | 0.12% |
