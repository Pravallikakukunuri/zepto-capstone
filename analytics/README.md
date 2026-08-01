# /analytics - Analytics Pipeline Module

Titanic dataset: profile, clean, EDA data story (01_eda.py), then full modeling pipeline (02_modeling.py).

## How to run
cd analytics
pip install -r requirements.txt
python 01_eda.py
python 02_modeling.py

## Part A — EDA Findings

### Missing values and strategy applied
| Column | % missing | Strategy | Justification |
|---|---|---|---|
| deck | 77.22% | Dropped the column | Over the 30% threshold. At this rate, imputing would be manufacturing the large majority of the column's values from almost nothing, and an "Unknown" bucket covering ~3/4 of rows would add noise rather than signal. |
| age | 19.87% | Imputed with median (28.0) | Falls in the 5-30% band -> impute per the threshold rule. |
| embarked | 0.22% | Dropped rows (2 rows dropped) | Under 5% missing -> drop rule; negligible data loss. |
| embark_town | 0.22% | Dropped rows | Mirrors embarked; also dropped as a redundant column afterward. |

Also dropped (redundant/leaky, separate from the missing-value rule): alive (string mirror of survived - target leakage), class (string duplicate of pclass), who (derived from sex + age).

### Univariate — age & fare
- Age: Q1=22.00, Q3=35.00, IQR=13.00, outlier bounds=[2.50, 54.50] -> 65 outliers
- Fare: Q1=7.90, Q3=31.00, IQR=23.10, outlier bounds=[-26.76, 65.66] -> 114 outliers
- Fare: mean=32.10, median=14.45, mode=8.05. Since mean > median > mode, fare is right-skewed -- a small number of very expensive fares pull the mean well above the median and mode.

### Bivariate — survival rate
- By sex: male=0.189 (n=577), female=0.740 (n=312) -- women survive at nearly 4x the rate of men.
- By pclass: 1st=0.626 (n=214), 2nd=0.473 (n=184), 3rd=0.242 (n=491) -- survival drops sharply as class number increases.
- By sex+pclass: 1st class women survived at 0.967, 2nd class women at 0.921, 3rd class men at only 0.135 -- the two effects compound strongly.

### Correlation matrix (6x6: survived, pclass, age, sibsp, parch, fare)
adult_male and alone excluded (derived/redundant flags, not independent measured features).

Two strongest absolute correlations:
1. pclass <-> fare: -0.548 -- higher class number (lower cabin class) strongly associates with cheaper fare, since pclass is essentially a coarse bucketing of the same wealth signal fare measures continuously.
2. sibsp <-> parch: 0.415 -- passengers traveling with more siblings/spouses also tend to travel with more parents/children, reflecting that both capture "traveling with family" rather than fully independent information.

### Multivariate data story (4 charts, in charts/ folder)
1. **story_1_bar_survival_by_class_sex.png**: Survival rate is highest for women in 1st and 2nd class and lowest for men in 3rd class, showing sex and class combined mattered far more than either alone -- the "women and children first" evacuation norm interacted strongly with cabin class access to lifeboats.
2. **story_2_box_fare_by_class_survival.png**: Within every class, survivors tend to have paid somewhat higher fares than non-survivors, and 1st class fares are far higher and more spread out than 3rd class -- fare acts as a finer-grained proxy for wealth and likely deck location beyond pclass alone.
3. **story_3_scatter_age_fare_survival.png**: Survivors are scattered across all ages but cluster more at higher fares, while most older, low-fare passengers did not survive -- reinforcing that fare/class mattered more than age alone.
4. **story_4_pairplot.png**: The pair plot shows survivors concentrated at lower pclass values and a wider fare range, with no single variable cleanly separating the two groups on its own -- survival looks like a joint function of class, fare, and age rather than any single feature.

### EDA-stage standardization check (z-score, age & fare)
- age: before mean=29.32, std=12.98 -> after mean~0 (2.80e-16), std=1.00
- fare: before mean=32.10, std=49.70 -> after mean~0 (1.36e-16), std=1.00

Both columns confirm mean approx 0 and std approx 1 after standardization. This check is exploratory only -- the modeling pipeline performs its own train-only scaling.

## Part B — Modeling Findings

### Stratified split
survived is imbalanced: 61.6% not survived / 38.4% survived in the full data. Train split: 61.7%/38.3%. Test split: 61.5%/38.5% -- stratification kept both folds almost identical to the full dataset's class balance, avoiding a random split that could have skewed one fold's survival rate purely by chance.

### Classifier comparison
| model | accuracy | precision | recall | f1 | auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.816 | 0.790 | 0.710 | 0.748 | 0.790 |
| Random Forest | 0.816 | 0.800 | 0.696 | 0.744 | 0.830 |

Confusion matrices:
- Logistic Regression: [[98,12],[23,46]]
- Decision Tree: [[97,13],[20,49]]
- Random Forest: [[98,12],[21,48]]

### Imbalance handling comparison
Class balance in y_train: 61.7% not survived / 38.3% survived.

| variant | precision | recall | f1 |
|---|---|---|---|
| Baseline (no handling) | 0.800 | 0.696 | 0.744 |
| class_weight=balanced | 0.750 | 0.739 | 0.745 |
| SMOTE (train fold only) | 0.761 | 0.739 | 0.750 |

Conclusion: SMOTE (applied to the training fold only) gave the best F1 (0.750) among the three variants, edging out class_weight=balanced (0.745) and the baseline (0.744). SMOTE's synthetic oversampling of the minority (survived) class in training let the model learn that class's patterns better without touching the test fold, improving recall over the baseline while keeping precision reasonably high.

### Hyperparameter tuning
Best RandomForest params: max_depth=5, max_features='sqrt', n_estimators=100
OOB score: 0.827

### Regression side-task (predicting fare)
MAE=20.90, RMSE=30.53, R2=0.398, Adjusted R2=0.373

Correlation between predicted fare and |residuals|: 0.546 -- the residual plot DOES show heteroscedasticity: residual spread grows as predicted fare increases, consistent with fare's right-skewed distribution (a few very expensive fares are harder to predict precisely, producing larger errors at the high end).

### Final recommendation
Deploy the Decision Tree model. It has the highest F1 score (0.748) among the three classifiers, balancing precision (0.790) and recall (0.710). While Logistic Regression has the highest AUC (0.844), its recall (0.667) is noticeably lower, meaning it misses more actual survivors. Random Forest is close behind on most metrics and its GridSearchCV-tuned variant reaches an OOB score of 0.827, making it a reasonable alternative if slightly more robustness is wanted at the cost of interpretability. Given the Decision Tree's strong balance of precision/recall and its interpretability (visualized via plot_tree), it is the recommended model to deploy for this classification task.

### Saved pipeline
best_pipeline.joblib contains the complete fitted Pipeline (ColumnTransformer + GridSearchCV-tuned RandomForestClassifier), saved via joblib.dump. Reload check confirmed: prediction on one raw test row matched the actual label (predicted [0], actual 0), verifying the saved pipeline works end-to-end on raw, unpreprocessed data.
