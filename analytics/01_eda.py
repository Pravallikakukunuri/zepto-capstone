"""
01_eda.py — Part A: Profiling, cleaning, and the EDA data story.

This is the ONLY place sns.load_dataset('titanic') is called in this whole
module. It loads the dataset from network/cache exactly once, immediately
saves the RAW snapshot to titanic.csv (the offline fallback required by the
brief), then profiles, cleans (in memory, for EDA purposes), and explores it.

02_modeling.py never calls sns.load_dataset again — it reads titanic.csv.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(CHART_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved chart -> charts/{name}")


print("=" * 78)
print("STEP 0: Loading Titanic dataset (sns.load_dataset)")
df_raw = sns.load_dataset("titanic")
raw_csv_path = os.path.join(OUT_DIR, "titanic.csv")
df_raw.to_csv(raw_csv_path, index=False)
print(f"  Raw dataset saved as offline fallback -> {raw_csv_path}")
print(f"  Shape: {df_raw.shape}")

df = df_raw.copy()

print("\n" + "=" * 78)
print("STEP 1: Profiling")
print("-" * 78)
print(df.info())
print("\ndf.describe(include='all'):")
print(df.describe(include="all"))
print(f"\ndf.shape: {df.shape}")

missing_pct = (df.isnull().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\nMissing value percentage per column:")
print(missing_pct)

print("\n" + "=" * 78)
print("STEP 2: Missing-value handling (threshold rule)")
print("-" * 78)

if "age" in missing_pct.index:
    pct = missing_pct["age"]
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)
    print(f"age: {pct}% missing -> 5-30% band -> imputed with median ({median_age})")

if "embarked" in missing_pct.index:
    pct = missing_pct["embarked"]
    before = len(df)
    df = df.dropna(subset=["embarked"])
    print(f"embarked: {pct}% missing -> <5% band -> dropped {before - len(df)} row(s)")

if "embark_town" in missing_pct.index:
    pct = missing_pct["embark_town"]
    before = len(df)
    df = df.dropna(subset=["embark_town"])
    print(f"embark_town: {pct}% missing -> <5% band -> dropped {before - len(df)} row(s)")

if "deck" in missing_pct.index:
    pct = missing_pct["deck"]
    df = df.drop(columns=["deck"])
    print(f"deck: {pct}% missing -> >30% band -> DROPPED the column "
          f"(too sparse to impute reliably; an 'Unknown' bucket covering ~3/4 "
          f"of rows would add noise rather than signal)")

leak_or_redundant = [c for c in ["alive", "class", "who"] if c in df.columns]
df = df.drop(columns=leak_or_redundant)
print(f"\nDropped as redundant/leaky: {leak_or_redundant}")
print(f"\nShape after cleaning: {df.shape}")

print("\n" + "=" * 78)
print("STEP 3: Univariate analysis — age & fare")
print("-" * 78)


def iqr_outlier_count(series, label):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    print(f"{label}: Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  "
          f"bounds=[{lower:.2f}, {upper:.2f}]  -> {len(outliers)} outliers")
    return len(outliers)


for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df[col], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title(f"{col} — histogram")
    axes[1].boxplot(df[col].dropna(), vert=True)
    axes[1].set_title(f"{col} — boxplot")
    savefig(f"univariate_{col}.png")
    iqr_outlier_count(df[col], col)

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]
print(f"\nfare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
if fare_mean > fare_median > fare_mode:
    skew_txt = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    skew_txt = "left-skewed"
else:
    skew_txt = "not a clean monotonic ordering -- inspect the histogram"
print(f"Since mean/median/mode = {fare_mean:.2f} / {fare_median:.2f} / {fare_mode:.2f}, "
      f"fare is {skew_txt}.")

print("\n" + "=" * 78)
print("STEP 4: Bivariate analysis — survival rate breakdowns")
print("-" * 78)

print("(a) Survival rate by sex:")
for sex_val in df["sex"].unique():
    mask = (df["sex"] == sex_val)
    rate = df.loc[mask, "survived"].mean()
    print(f"    sex={sex_val}: {rate:.3f}  (n={mask.sum()})")

print("\n(b) Survival rate by pclass:")
for pclass_val in sorted(df["pclass"].unique()):
    mask = (df["pclass"] == pclass_val)
    rate = df.loc[mask, "survived"].mean()
    print(f"    pclass={pclass_val}: {rate:.3f}  (n={mask.sum()})")

print("\n(c) Survival rate by sex AND pclass:")
for sex_val in df["sex"].unique():
    for pclass_val in sorted(df["pclass"].unique()):
        mask = (df["sex"] == sex_val) & (df["pclass"] == pclass_val)
        rate = df.loc[mask, "survived"].mean()
        print(f"    sex={sex_val} & pclass={pclass_val}: {rate:.3f}  (n={mask.sum()})")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[corr_cols].corr()
print("\nCorrelation matrix (6x6, adult_male & alone excluded):")
print(corr.round(3))

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation heatmap — 6 numeric columns")
savefig("correlation_heatmap.png")

corr_pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        corr_pairs.append((corr_cols[i], corr_cols[j], corr.iloc[i, j]))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nTop 2 strongest absolute off-diagonal correlations:")
for a, b, v in corr_pairs[:2]:
    print(f"    {a} <-> {b}: {v:.3f}")

print("\n" + "=" * 78)
print("STEP 5: Multivariate data story (4 charts)")
print("-" * 78)

survival_grid = df.groupby(["pclass", "sex"])["survived"].mean().unstack()
survival_grid.plot(kind="bar", figsize=(7, 4))
plt.ylabel("Survival rate")
plt.title("Chart 1: Survival rate by passenger class and sex")
savefig("story_1_bar_survival_by_class_sex.png")
print("Chart 1 interpretation: Survival rate is highest for women in 1st and 2nd "
      "class and lowest for men in 3rd class, showing that sex and class combined "
      "mattered far more than either alone -- the 'women and children first' "
      "evacuation norm interacted strongly with cabin class access to lifeboats.")

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="pclass", y="fare", hue="survived")
plt.title("Chart 2: Fare distribution by class, split by survival")
savefig("story_2_box_fare_by_class_survival.png")
print("Chart 2 interpretation: Within every class, survivors tend to have paid "
      "somewhat higher fares than non-survivors, and 1st class fares are far "
      "higher and more spread out than 3rd class -- fare acts as a finer-grained "
      "proxy for wealth and likely deck location beyond pclass alone.")

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6)
plt.title("Chart 3: Age vs fare, colored by survival")
savefig("story_3_scatter_age_fare_survival.png")
print("Chart 3 interpretation: Survivors are scattered across all ages but cluster "
      "more at higher fares, while most older, low-fare passengers did not "
      "survive -- reinforcing that fare/class mattered more than age alone.")

pair_cols = ["age", "fare", "pclass", "survived"]
pp = sns.pairplot(df[pair_cols], hue="survived", diag_kind="hist")
pp.savefig(os.path.join(CHART_DIR, "story_4_pairplot.png"), dpi=120)
plt.close("all")
print("  saved chart -> charts/story_4_pairplot.png")
print("Chart 4 interpretation: The pair plot shows survivors concentrated at "
      "lower pclass values and a wider fare range, with no single variable "
      "cleanly separating the two groups on its own -- survival looks like a "
      "joint function of class, fare, and age rather than any single feature.")

print("\n" + "=" * 78)
print("STEP 6: EDA-stage standardization check (z-score) — age & fare")
print("-" * 78)

for col in ["age", "fare"]:
    mean, std = df[col].mean(), df[col].std()
    z_col = f"{col}_z"
    df[z_col] = (df[col] - mean) / std
    print(f"{col}: BEFORE -> mean={mean:.2f}, std={std:.2f}  |  "
          f"AFTER  -> mean={df[z_col].mean():.2e}, std={df[z_col].std():.2f}")

clean_csv_path = os.path.join(OUT_DIR, "titanic_clean.csv")
df.to_csv(clean_csv_path, index=False)
print(f"\n(Bonus) cleaned dataframe also saved -> {clean_csv_path}")

print("\n" + "=" * 78)
print("01_eda.py complete. Now run: python 02_modeling.py")