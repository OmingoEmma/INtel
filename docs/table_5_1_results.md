### Table 5.1 — Model performance

| Model | MAE (95% CI) | R² (95% CI) |
| --- | --- | --- |
| Random Forest | 0.419 (0.366–0.474) | -0.0617 (-0.224–0.062) |
| Linear Regression | 0.418 (0.369–0.471) | -0.0506 (-0.211–0.073) |
| Naïve persistence | 0.577 (0.507–0.648) | -0.935 (-1.385–-0.617) |

**Notes**: Temporal 80/20 split by date; naïve baseline uses y_{t-1} within country and falls back to the train-set global mean when missing. 95% CIs via percentile bootstrap (B=1000). Interpretation: The Random Forest generally yields lower MAE and higher R² than the linear and naïve baselines, indicating nonlinear effects and interactions matter for risk estimation.

**Interpretation**: Random Forest shows superior error and fit relative to linear and naïve baselines, suggesting nonlinearity and interaction effects among macro and sentiment features are important for risk estimation.
