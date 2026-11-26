# Predicting Data Salaries — and Checking Whether the Model Was Worth Building

A DataCamp guided project reworked from scratch. The original task was to predict the
salaries of data professionals from survey data. I did that, and then spent most of the
work checking the result, which turned out to be the more interesting part.

## The headline

**A tuned XGBoost model on nine features is statistically indistinguishable from a pivot
table of median salaries.**

| | MAE | % of mean salary |
|---|---|---|
| constant: median of the training set | 54,906 | 34.4% |
| median by experience level | 51,604 | 32.3% |
| median by experience × role | 47,246 | 29.6% |
| **median by experience × role × country** | **45,554** | **28.6%** |
| XGBoost, 8 features, grid-searched | 45,661 | 28.6% |

The paired difference is 0.7 sigma on one split. Across five random splits the table is
ahead by 228 on average, which is 0.5%, and the sign is the same every time. All the
features are categorical, the useful interactions are low order, and there are enough
rows to estimate each cell directly, so there is very little left for the algorithm to
find.

The second finding follows from the same measurements:

| what changes | effect on MAE |
|---|---|
| the random split (seed) | ~690 |
| Lasso instead of a tree model | ~770 |
| between the four tree models | ~200 |
| the model instead of a pivot table | ~228 |
| nine features instead of eight | ~96 |
| the full grid search instead of ten random draws | ~38 |

**The choice of random seed moves the result more than any decision I made.** The full
grid for XGBoost is 540 fits; ten random draws with three folds is 30 fits and lands
38 USD away.

## Other findings

**Simpson's paradox in the year-on-year growth.** Between 2023 and 2024 the median
salary rose inside every experience level, by 5% to 18%, while the overall median rose
by only 3.4%. The sample became more junior, which hid the growth. Standardising 2024 to
the 2023 composition gives about +7.8%.

**The model does not transfer forward in time.** Trained on 2020–2023 and asked about
2024 it gives MAE 49,863 against 45,736 for a random split of the same size. A constant
correction does not repair it, so what changed between the years is the relationship
between the features and the pay, not only the level.

**The prediction interval was wrong, and is now measured.** The original
`prediction ± MAE` contained the true salary 58.8% of the time. An interval built from
the residual quantiles — split conformal prediction — covers 89.4% against a nominal
90%. Calibrating it separately inside each quarter of the predictions — mondrian
conformal prediction — keeps the coverage near 90% in every group, instead of swinging
between 97% and 82%.

**Five feature experiments, and every one confirmed the original choice.** Coarser role
groupings, country groupings by income level, dropping the redundant residence column,
collapsing the currency column, and merging small role categories were all tested with
paired comparisons on identical validation rows.

## What was wrong with the first version

I kept the mistakes visible rather than quietly fixing them, because the corrections are
the work:

- Every hyperparameter search was commented out, and the MAE values in the charts were
  typed in by hand. The notebook now refits the selected configurations, and all fifteen
  numbers reproduce to the cent.
- One model had no `random_state`, so its reported test MAE could not be reproduced.
- `max_features: 'auto'` has raised an error since scikit-learn 1.3.
- The EDA ranked regions by mean with no group sizes shown. North Africa was called the
  second highest paid region on 39 rows; by median it is fourth. One conclusion named a
  category that does not exist in that variable at all, because small categories had
  been merged into "Other" earlier in the notebook.
- The bar charts showed means of a right-skewed target. They now show medians with
  confidence intervals and the number of rows next to every bar.
- PCA was applied to one-hot encoded data before a tree model. The notebook now explains
  why that could not have helped.

## Limitations

- 89–92% of the rows are from the United States, and that share grew over time. This is
  not a worldwide salary model.
- 82% of the rows are from 2024. 2020 and 2021 together hold 293 rows, so any statement
  about a trend since 2020 rests on 75 observations.
- All amounts are nominal. No inflation adjustment was made, so the year-on-year growth
  above is overstated by whatever inflation was over the period.
- The rows are survey responses, not the same people followed over time. The honest
  wording is always "what is reported in this data", never "what people are paid".

## The data

`salaries.csv` — 57,194 rows, 11 columns, no missing values, covering 2020 to 2024. From
a survey hosted by an HR consultancy, provided with the DataCamp project.

## Running it

```
pip install -r requirements.txt
jupyter notebook notebooks/salary_prediction.ipynb
```

The notebook runs from top to bottom. The slowest parts are the five-model refits on the
nine-feature set and the randomized search; expect fifteen to twenty minutes in total.
