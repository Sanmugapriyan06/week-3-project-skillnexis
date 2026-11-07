"""Week 3 - Python & Data Wrangling
Assignment: clean a messy dataset in Pandas (missing values, filter rows,
create new columns) + intro visualization with Matplotlib & Seaborn.
Run: python week3_solutions.py
To use your own file, set INPUT_CSV and delete the 'sample data' block."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "messy_data.csv"

# ---------- Sample messy data (replace with your dataset) ----------
rng = np.random.default_rng(7)
n = 60
raw = pd.DataFrame({
    "Order ID": np.arange(1, n + 1),
    " Customer Name ": rng.choice(["  asha ", "BEN", "chen", "Divya ", "eli", None], n),
    "Region": rng.choice(["North", "south", "SOUTH", "East", "west", "West "], n),
    "Order Date": rng.choice(["2026-01-15", "2026-02-03", "2026/03/10", "04-04-2026", "bad date", None], n),
    "Quantity": rng.choice([1, 2, 3, 4, 5, np.nan, -2], n),
    "Unit Price": rng.choice([250, 500, 1200, 1800, np.nan, 99999], n),
})
raw = pd.concat([raw, raw.iloc[[2, 9]]], ignore_index=True)
raw.to_csv(INPUT_CSV, index=False)

# ---------- 1. Load & inspect ----------
df = pd.read_csv(INPUT_CSV)
print("Raw shape:", df.shape)
print(df.isna().sum())

# ---------- 2. Clean ----------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.drop_duplicates()
for c in ["customer_name", "region"]:
    df[c] = df[c].str.strip().str.title()
df["customer_name"] = df["customer_name"].fillna("Unknown")
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")
df = df.dropna(subset=["order_date"])                       # unparseable dates

# invalid values -> NaN, then impute with median
df.loc[df["quantity"] <= 0, "quantity"] = np.nan
df.loc[df["unit_price"] > 10000, "unit_price"] = np.nan     # outlier/sentinel
df["quantity"] = df["quantity"].fillna(df["quantity"].median())
df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

# ---------- 3. Filter rows ----------
df = df[df["order_date"] >= "2026-01-01"]
high_value = df[df["unit_price"] >= 1000]

# ---------- 4. New columns ----------
df["revenue"] = df["quantity"] * df["unit_price"]
df["month"] = df["order_date"].dt.to_period("M").astype(str)
df["order_size"] = pd.cut(df["revenue"], [0, 1000, 4000, np.inf],
                          labels=["Small", "Medium", "Large"])
print("\nClean shape:", df.shape, "| missing:", int(df.isna().sum().sum()))
print(df.head())
print("\nHigh-value rows:", len(high_value))

# ---------- 5. Summaries ----------
print("\nRevenue by region:\n", df.groupby("region")["revenue"].sum().sort_values(ascending=False))
df.to_csv("cleaned_data.csv", index=False)

# ---------- 6. Visualization ----------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
df.groupby("region")["revenue"].sum().sort_values().plot.barh(ax=ax[0], color="steelblue")
ax[0].set_title("Total Revenue by Region")
sns.histplot(df["revenue"], bins=15, kde=True, ax=ax[1])
ax[1].set_title("Revenue Distribution")
sns.boxplot(data=df, x="order_size", y="unit_price", ax=ax[2])
ax[2].set_title("Unit Price by Order Size")
plt.tight_layout()
plt.savefig("week3_charts.png", dpi=120)
print("\nSaved cleaned_data.csv and week3_charts.png")
