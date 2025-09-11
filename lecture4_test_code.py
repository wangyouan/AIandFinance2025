# -*- coding: utf-8 -*-
"""
Demo: Comparing OLS, Lasso, RandomForest, and GradientBoosting
on monthly CRSP data for 1–3 large-cap stocks, with RMSE/MAE/Hit,
fit-time, and DM / Clark–West tests.

Author: Youan Wang
Course: AI & Finance – Lecture 4
"""

from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
import wrds
import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
wrds_user = os.getenv("WRDS_USER")

# -------------------------
# 0) Reproducibility
# -------------------------
SEED = 42
np.random.seed(SEED)

# -------------------------
# 1) Pull CRSP monthly from WRDS
# -------------------------

db = wrds.Connection(wrds_username=wrds_user)  # will prompt for your WRDS credentials

# Choose 1–3 liquid PERMNOs (example: AAPL=14593 historically; confirm in CRSP)
PERMNO_LIST = [14593]  # replace / extend if desired

# Query monthly returns and basic fields from CRSP monthly stock file (msf)
# - RET: total return including dividends
# - PRC: price, SHROUT: shares outstanding (to proxy dollar volume/turnover)
# You can left join to msiz if you want exchange codes; keep minimal here.
sql = f"""
    select permno, date, ret, prc, shrout, vol
    from crsp.msf
    where permno in ({",".join(str(x) for x in PERMNO_LIST)})
        and date between '2005-01-31' and '2024-12-31'
    order by permno, date;
"""
raw = db.raw_sql(sql, date_cols=['date'])


# Basic cleaning
df = raw.copy()
# Convert returns to numeric; CRSP RET sometimes has 'C' or 'B' codes—coerce to NaN
df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
df = df.dropna(subset=['ret', 'prc', 'shrout']).sort_values(['permno', 'date'])

# -------------------------
# 2) Feature engineering (per permno, then stack back)
# -------------------------


def build_features(g):
    g = g.sort_values('date').copy()

    # 2.1 Simple lags / momentum
    g['ret_l1'] = g['ret'].shift(1)
    # 12-1 momentum: cumret over t-12..t-2 * (1+ret_{t-1}) - 1, but we keep standard proxy:
    g['mom_12_1'] = (1.0 + g['ret'].shift(1)).rolling(12).apply(np.prod, raw=True) - 1.0

    # 2.2 Rolling volatility (6m)
    g['vol_6m'] = g['ret'].rolling(6).std()

    # 2.3 Dollar volume proxy and (very) simplified Amihud illiquidity
    # dollar volume ~ |PRC| * SHROUT * VOL?  At monthly freq, we use:
    # Turnover (approx): VOL / SHROUT (both in CRSP units)
    g['turnover'] = (g['vol'] / (g['shrout'] * 1000)).replace([np.inf, -np.inf], np.nan)

    # Dollar volume proxy: |PRC| * VOL (ignoring price sign conventions)
    dollar_vol = g['prc'].abs() * g['vol']
    g['illiq'] = (g['ret'].abs() / (dollar_vol.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)

    # 2.4 Log price to capture scale
    g['log_prc'] = np.log(g['prc'].abs())

    return g


df = df.groupby('permno', group_keys=False).apply(build_features)

# Drop early NaNs from rolling features
feats = ['ret_l1', 'mom_12_1', 'vol_6m', 'illiq', 'turnover', 'log_prc']
df = df.dropna(subset=feats + ['ret']).copy()

# For classroom: focus on one permno for speed & clarity (you can loop permno if you like)
permno_focus = df['permno'].unique()[0]
df1 = df[df['permno'] == permno_focus].copy()

# -------------------------
# 3) Train/test via expanding window, 1-step-ahead prediction
# -------------------------

X = df1[feats].values
y = df1['ret'].values
dates = df1['date'].values

# Start testing after we have enough history (e.g., 60 months)
start_test_idx = 60

preds = {
    'OLS': np.full_like(y, np.nan, dtype=float),
    'Lasso': np.full_like(y, np.nan, dtype=float),
    'RF': np.full_like(y, np.nan, dtype=float),
    'GBDT': np.full_like(y, np.nan, dtype=float),
}
fit_time = {k: 0.0 for k in preds.keys()}

for t in range(start_test_idx, len(y)):
    X_train, y_train = X[:t, :], y[:t]
    X_test = X[t, :].reshape(1, -1)

    # -- OLS --
    t0 = time.time()
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    fit_time['OLS'] += (time.time() - t0)
    preds['OLS'][t] = ols.predict(X_test)[0]

    # -- Lasso --
    t0 = time.time()
    lasso = Lasso(alpha=0.001, random_state=SEED, max_iter=5000)
    lasso.fit(X_train, y_train)
    fit_time['Lasso'] += (time.time() - t0)
    preds['Lasso'][t] = lasso.predict(X_test)[0]

    # -- Random Forest (keep small for speed) --
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    fit_time['RF'] += (time.time() - t0)
    preds['RF'][t] = rf.predict(X_test)[0]

    # -- Gradient Boosting (sklearn; fast & light) --
    t0 = time.time()
    gbdt = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=2,
        random_state=SEED
    )
    gbdt.fit(X_train, y_train)
    fit_time['GBDT'] += (time.time() - t0)
    preds['GBDT'][t] = gbdt.predict(X_test)[0]

# Keep only the OOS period
mask = ~np.isnan(preds['OLS'])
y_oos = y[mask]
dates_oos = dates[mask]


def hit_ratio(y_true, y_pred):
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    # count zeros as miss (state your rule clearly on slides)
    return np.mean(s_true == s_pred)


# Collect metrics
records = []
for name, yhat in preds.items():
    yhat = yhat[mask]
    rmse = np.sqrt(mean_squared_error(y_oos, yhat))
    mae = mean_absolute_error(y_oos, yhat)
    hit = hit_ratio(y_oos, yhat)
    records.append([name, rmse, mae, hit, fit_time[name]])

results = pd.DataFrame(records, columns=['Model', 'RMSE', 'MAE', 'HitRatio', 'FitTime_sec'])
print(results.sort_values('RMSE'))

# -------------------------
# 4) DM test (squared-error loss) & Clark–West test (nested)
# -------------------------


def dm_test(e1, e2, h=1):
    """
    Diebold–Mariano test for equal predictive accuracy.
    e1, e2: forecast errors (y - yhat) aligned on the same dates.
    h: forecast horizon; for 1-step ahead, h=1.
    Uses squared error loss by default.
    """
    d = (e1**2 - e2**2)
    d_mean = d.mean()
    # Newey-West variance with lag = h-1 (for h=1, lag=0)
    # For simplicity in class: use sample variance / T (OK for h=1)
    var_d = d.var(ddof=1) / len(d)
    dm_stat = d_mean / np.sqrt(var_d)
    pval = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, pval


def clark_west_test(y, y_bench, y_aug):
    """
    Clark–West (2007) test for nested models, 1-step ahead.
    y: actuals
    y_bench: benchmark predictions (e.g., OLS)
    y_aug: augmented model predictions (e.g., GBDT)
    Returns CW statistic and p-value (H0: no improvement).
    """
    f_b = y_bench
    f_a = y_aug
    u = (y - f_b)**2 - ((y - f_a)**2 - (f_b - f_a)**2)
    cw_stat = u.mean() / (u.std(ddof=1) / np.sqrt(len(u)))
    pval = 1 - stats.norm.cdf(cw_stat)  # one-sided: augmented better than benchmark
    return cw_stat, pval


# Build aligned errors
errs = {name: (y_oos - preds[name][mask]) for name in preds.keys()}

# Example comparisons:
dm_rf_vs_ols = dm_test(errs['OLS'], errs['RF'])
dm_gbdt_vs_ols = dm_test(errs['OLS'], errs['GBDT'])
print(f"DM (RF vs OLS): stat={dm_rf_vs_ols[0]:.3f}, p={dm_rf_vs_ols[1]:.3f}")
print(f"DM (GBDT vs OLS): stat={dm_gbdt_vs_ols[0]:.3f}, p={dm_gbdt_vs_ols[1]:.3f}")

cw_gbdt_vs_ols = clark_west_test(y_oos, preds['OLS'][mask], preds['GBDT'][mask])
cw_rf_vs_ols = clark_west_test(y_oos, preds['OLS'][mask], preds['RF'][mask])
print(f"CW (GBDT vs OLS): stat={cw_gbdt_vs_ols[0]:.3f}, p(one-sided)={cw_gbdt_vs_ols[1]:.3f}")
print(f"CW (RF vs OLS):   stat={cw_rf_vs_ols[0]:.3f}, p(one-sided)={cw_rf_vs_ols[1]:.3f}")
