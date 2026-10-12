# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Javier Arce López
- **Student ID**: 20158
- **Email**: jarcel.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

Ames Housing sales from Dean De Cock (`AmesHousing.csv` in this repo). Each
row is one house sold in Ames, Iowa, with property attributes and the
recorded closing price.

The file is tabular, has a clear price target, and enough features that
preprocessing and engineering actually matter, while staying small enough to
commit and retrain from scratch. That also matches a concrete business use:
checking an asking price against comparable closes.

After dropping three partial-sale outliers (as documented by the dataset creator, De Cock), 2,927 sales remain: 1,938 in
training (2006–2008), 648 in validation (2009), and 341 in test (2010). After preparation, the model matrix has 159 columns.

## Business / real-life framing

A buyer's agent in Ames uses this model before a client writes an offer. The seller has named an asking price. The agent wants an estimate of what that house would close for, based on houses that have already sold, and using only what is known before the deal. This is, the house itself, and the month and year of the search. If the ask sits well above the estimate, the agent will tell the client the ask is aggressive, helping them better estimate the performance of the offer in the market. 

Three partial sales of houses above 4,000 sq ft are removed. Those prices are not comparisons a client could use. The two other houses above that size stay, because their prices match each other.

The split is ordered chronologically. An agent pricing a house in 2010 does not yet have the 2010 sales. The model is fit on 2006–2008, the penalty and the learning rate are chosen on 2009, and 2010 is the test. A random split would train on sales that had not closed yet, resulting in data leackage.

There is no class threshold. A large dollar miss is the mistake that leads to a bad offer, so models are compared by dollar RMSE after the log prediction is converted back to dollars. Dollar MAE is reported next to it so the typical miss is visible too.

Training uses log price because raw prices are strongly right-skewed, and the same dollar miss means something different on a cheap house and an expensive one. The reported errors are still in dollars, which is the unit of the offer.

## Data preparation & feature engineering

**Rows and date fixes.** Three Edwards partial sales above 4,000 sq ft
(orders 1499, 2181, 2182) are removed; their prices are not usable
comparables. Two other large houses stay. Three date typos are corrected on
the full file before the split (garage year 2207 → 2007; remodel before
build → build year; remodel after sale → sale year).

**Missing values.** Blanks that mean “no alley / basement / fireplace /
garage / fence / masonry” become `"None"` (and related areas or baths 0).
Real gaps are filled from 2006–2008 only: lot frontage by neighborhood
median (then overall training median), masonry area by veneer-type median,
basement exposure and finish among houses with a basement, electrical by
training mode, and two incomplete detached garages from other training
detached garages.

**Columns dropped.** Identifiers; `Sale Condition` / `Sale Type` (deal
outcome, unknown beforehand — condition is used only to spot the three
partials); near-constant fields; redundant area/car columns that duplicate
`Gr Liv Area`, basement totals, or `Garage Area`; raw year columns replaced
by ages; `Pool QC` / `Misc Feature` (almost always blank) while keeping
`Pool Area` and `Misc Val`.

**Features added.** `Age`, `RemodAge`, and `GarageAge` (0 if no garage) so
the model sees age at sale without inventing a build year for missing
garages. `Mo Sold` and `Yr Sold` stay: the agent knows when the client is
shopping. Year stays numeric so 2009–2010 can extend a trend; a year dummy
would be all zeros out of sample. Month is categorical (December is not
“12× January”).

**Encoding and scaling.** Ordered qualities become equally spaced integers
(an assumption). Remaining categories, including `MS SubClass` and month,
are one-hot with levels under 10 training rows collapsed to `Other`,
sorted A–Z, first level dropped. After that, 2009 and 2010 introduce no
unseen category. All 159 columns are standardized with a scaler fit on
training only. Neighborhood medians, rare-level map, dummy columns, and
scaler never see validation or test.

## Modeling: three implementations, one model

The shared model is linear regression on `log(SalePrice)` — continuous
target, same features for all three fits. Training minimizes error in log
space; reported metrics convert predictions back to dollars (RMSE primary,
MAE beside it). The naive baseline predicts the training geometric mean
(~$167,485) for every house.

**Tuning (validation 2009 only).** A small Ridge α grid picked α=30 (best
val dollar RMSE among OLS and Ridge). Lasso was swept as a reference; it
was slightly worse than Ridge on val and needs a proximal step the manual
PyTorch loop does not have, so it is not the shared three-way model. For
both PyTorch trainers, learning rate was chosen on the same val set with
α fixed at 30; lr=0.05 was best among finite runs (0.1 diverged). Test
2010 was not used to pick α or lr.

**Unregularized check.** Before Ridge, sklearn OLS and both PyTorch OLS
loops already agreed closely on the same split. After Ridge, coefficients
match sklearn within ~1e-11; the two PyTorch paths match each other to
machine precision.

**Results (dollar RMSE / MAE).** Val chose the model; test is the number
that matters.

| Model | Train RMSE | Val RMSE | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| Naive baseline | 81,753 | 81,985 | 74,697 | 53,098 |
| sklearn Ridge (α=30) | 17,765 | 19,156 | 21,153 | 15,150 |
| manual PyTorch Ridge | 17,765 | 19,156 | 21,153 | 15,150 |
| standard PyTorch Ridge | 17,765 | 19,156 | 21,153 | 15,150 |

The three graded fits agree. Versus OLS, Ridge improved val RMSE by about
$90 (19,246 → 19,156) but test RMSE got slightly worse (21,077 → 21,153).
Regularization was kept because selection used validation only; the test
reversal is noted under Limitations.

## Limitations & next steps

**2010 is incomplete.** Sales stop in July, so test error mixes model
quality with a shorter season. With a full later year, re-run the same
calendar protocol before trusting the dollar numbers for year-end pricing.

**No asking prices in the file.** The intended rule — flag an ask that
exceeds the estimate by more than typical dollar MAE — cannot be tested
here. Collect list vs close on new listings and score that rule directly.

**Ordinal scores assume equal steps.** “Good” vs “excellent” may not match
one integer step. Try learned embeddings or fewer bins if those fields
dominate residuals.

**Rare levels → `Other`.** Unusual neighborhoods or subclasses share one
coefficient. With more data, raise the frequency cutoff or use target
encoding fit on training folds only.

**Non-arms-length sales stay in the target.** Family and abnormal closes
pull the estimate down on purpose for a conservative check. For a
retail-only appraisal, filter to normal sales and refit — expect a higher
mean prediction and a different error profile.

**Ridge’s validation gain is small and reversed on test.** α=30 beat OLS by
about $90 RMSE on 2009 and lost about $76 on 2010. Treat regularization as
stabilization, not a proven accuracy lift; with more years of validation,
re-check whether α=0 is enough.

## Generative AI use disclosure

*TODO — complete in your own words.* Cursor was used for boilerplate,
notebook/report formatting, and sanity-checking the tuning section.
Dataset choice, business framing, split/metric decisions, feature
engineering choices, and the conclusions in this report are mine.
