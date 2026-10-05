import numpy as np
import pandas as pd
import random
import os

n = 110000
types = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
t_probs = [0.35, 0.32, 0.22, 0.09, 0.02]

np.random.seed(42)
random.seed(42)

print(f"Generating {n:,} synthetic transactions (~15 MB)...")

t = np.random.choice(types, size=n, p=t_probs)
s = np.sort(np.random.randint(101, 125, size=n))
amt = np.round(np.random.exponential(scale=3500.0, size=n) + 5.0, 2)
o_org = np.round(np.random.exponential(scale=15000.0, size=n) + 50.0, 2)
n_org = np.maximum(0.0, np.round(o_org - amt, 2))
o_dst = np.round(np.random.exponential(scale=25000.0, size=n), 2)
n_dst = np.round(o_dst + amt, 2)

orig_cnt = np.random.poisson(lam=8, size=n)
orig_spend = np.round(orig_cnt * np.random.uniform(500, 4000, size=n), 2)
dest_cnt = np.random.poisson(lam=25, size=n)
dest_sum = np.round(dest_cnt * np.random.uniform(800, 5000, size=n), 2)
dest_max = np.round(np.random.uniform(200, 15000, size=n), 2)

# Fix CASH_IN math
cash_in = t == "CASH_IN"
n_org[cash_in] = np.round(o_org[cash_in] + amt[cash_in], 2)
n_dst[cash_in] = np.maximum(0.0, np.round(o_dst[cash_in] - amt[cash_in], 2))

# Inject 350 clear fraud cases (account-draining transfers & mule cashouts)
frauds = np.random.choice(n, size=350, replace=False)
for idx in frauds:
    t[idx] = "TRANSFER"
    d = float(random.randint(90000, 400000))
    amt[idx] = d
    o_org[idx] = d
    n_org[idx] = 0.0
    o_dst[idx] = 0.0
    n_dst[idx] = 0.0
    dest_cnt[idx] = 0
    dest_sum[idx] = 0.0
    dest_max[idx] = 0.0

df = pd.DataFrame({
    "step": s,
    "type": t,
    "amount": amt,
    "nameOrig": [f"C{random.randint(100000000, 999999999)}" for _ in range(n)],
    "oldbalanceOrg": o_org,
    "newbalanceOrig": n_org,
    "nameDest": [
        f"M{random.randint(100000000, 999999999)}"
        if x == "PAYMENT"
        else f"C{random.randint(100000000, 999999999)}"
        for x in t
    ],
    "oldbalanceDest": o_dst,
    "newbalanceDest": n_dst,
    "orig_tx_count": orig_cnt,
    "orig_prev_amount_sum": orig_spend,
    "dest_tx_count": dest_cnt,
    "dest_prev_amount_sum": dest_sum,
    "dest_max_amount": dest_max,
})

output_file = "test_batch_feed.csv"
df.to_csv(output_file, index=False)
size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Created: {output_file} ({size_mb:.2f} MB, {len(df):,} rows)")