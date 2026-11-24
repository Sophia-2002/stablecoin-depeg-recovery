# Stablecoin Depeg Recovery: Fast or Slow?

**Can we predict whether a stablecoin recovers its $1 peg quickly after a depeg event?**

MSc Business Analytics dissertation, University of Manchester

`Python` · `scikit-learn` · `statsmodels` · `pandas` · `imbalanced-learn` · `matplotlib`

---

## 📌 Background
Stablecoins are meant to hold a $1 value, but they regularly lose their peg during market stress (Black Thursday 2020, Terra/UST 2022, FTX, SVB 2023). Some recover within a day, while others stay off-peg for weeks. For risk managers, exchanges and regulators, knowing **which events are likely to persist** is valuable for early warning.

## ❓ Research questions
- **H1:** Does the stablecoin's **mechanism type** (tokenised fund, off-chain collateral, on-chain collateral, algorithmic) affect recovery speed?
- **H2:** Do **event characteristics** (size and direction of the initial deviation) predict recovery speed?
- Do **market conditions before the event** (stablecoin, BTC and ETH dynamics) add predictive power?

## 📊 Data
- **135 depeg events** across 9 stablecoins: USDT, USDC, TUSD, BUSD, DAI, LUSD, FRAX, USDD, USTC
- Depeg event = daily price outside the **±2% band** around $1
- Target: **fast** (recovered in ≤3 days, n=107) vs **slow** (>3 days, n=28), an imbalance of about 3.8 : 1
- 33 candidate features in 5 groups, all measured at **t-1** (the day before the event) to avoid look-ahead bias

## 🔧 Method
1. **Feature engineering:** 33 features (event characteristics, mechanism type, stablecoin market dynamics, BTC and ETH market conditions)
2. **Multicollinearity removal:** pairs with |r| > 0.7 resolved on theoretical grounds, leaving 26 features
3. **Nested cross-validation** (no data leakage):
   - Outer: repeated stratified 5-fold × 5 seeds = 25 test folds
   - Inner: 3-fold CV for Random Forest–based top-N feature selection and hyperparameter tuning
4. **Models:** Logistic Regression vs Random Forest, with class imbalance handled by `class_weight='balanced'` (SMOTE as a sensitivity check)
5. **Robustness checks:** group-aware CV (events from the same crisis kept in the same fold), 7-day cutoff, excluding coins with no slow events, a direct H1 subsample test, and an overfitting (train–test gap) diagnostic

## 📈 Key results

| Model | F1 | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression (class_weight) | 0.415 | 0.593 | 0.697 |
| Random Forest (class_weight) | 0.402 | 0.544 | 0.669 |
| Baseline: univariate LR (initial deviation only) | 0.314 | 0.299 | 0.765 |

- **Initial deviation size** was selected in **100% of folds**, making it the most stable predictor of slow recovery.
- **Mechanism type was never selected.** In a direct test (on-chain vs algorithmic, controlling for deviation size), the effect was not significant (OR = 2.30, p = 0.18), so **H1 is not supported**.
- The difference between LR and RF is **not significant** (paired bootstrap 95% CI for the F1 difference: [-0.056, 0.087]). The simpler model performs as well as the complex one.
- Under group-aware CV the scores fall, which shows that **crisis contagion** inflates naive CV results. This is a key caveat for predictive models in crypto markets.

**Takeaway:** Recovery speed is only partly predictable with pre-event data. How far the price falls at the start of the event matters more than how the stablecoin is designed.

## 📁 Repository structure
```
├── notebooks/
│   └── stablecoin_complete_pipeline.ipynb   # full pipeline, Stages 0–7
├── data/          # see data/README.md
├── figures/       # EDA figures (class imbalance, duration, correlation heatmaps)
├── results/       # model results and robustness tables (CSV)
└── requirements.txt
```

## ▶️ How to run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/stablecoin_complete_pipeline.ipynb
```

## ⚠️ Limitations
- Small sample (n=135, 28 slow events), so fold-level variance is high
- Daily data only; intraday recovery dynamics are not captured
- The slow class is dominated by a few systemic crises

---
**Author:** Yi-Wen (Sophia) Su · [LinkedIn](https://www.linkedin.com/in/yi-wen-su-523a54433/)
