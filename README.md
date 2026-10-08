# Assignment 1

Pick a tabular dataset, frame a realistic business problem it would serve,
and train the **same** simple model (linear regression for a continuous
target, logistic regression for a binary one) three ways: scikit-learn, a
from-scratch PyTorch loop (Session 5's manual approach), and the standard
`torch.nn.Module` + `torch.optim` workflow. Compare all three and ship the
result behind an interactive Gradio dashboard.

The point isn't the dataset or getting the best score. It's showing you
understand every step well enough to explain it later, without your code
in front of you, in the written comprehension check (see Grading below).
Pick a dataset interesting enough that feature engineering actually
matters, and be honest in REPORT.md about what didn't work and why.

## The task

1. **Pick a dataset.** Tabular, with a clear regression or classification
   target (or one you can build non-trivially from the data), interesting
   enough that feature engineering isn't trivial. Large enough to train on,
   small enough to commit to this repo and retrain from scratch: a rough
   guide is under ~20k rows / ~20MB, not a strict cutoff.
2. **Frame the business problem** *before* writing any pipeline code: what
   hypothetical (or real) decision does this model support? Write it in
   REPORT.md, and make sure it drives later decisions: does target
   definition need care, would a random split leak the future into
   training, which metric should set the decision threshold?
3. **Preprocess, engineer features, and split** with numpy/pandas/sklearn,
   consistent with what you decided in step 2.
4. **Train the same model three ways** on the same split:
   - scikit-learn linear or logistic regression
   - manual PyTorch loop like we saw in the PyTorch introduction notebook
   - standard PyTorch workflow

   Compare all three against each other and against a naive baseline. Save
   each trained model to disk: training is the only stage that trains
   anything, and the Gradio app must load these saved models, never
   retrain them.
5. **Tune sparingly.** The only hyperparameters worth touching are
   regularization strength and, for the two PyTorch versions, the learning
   rate. Put your effort into features, not a grid search.
6. **Write REPORT.md** (skeleton already in this repo): dataset, business
   framing, your process, a three-method comparison table, and honest
   limitations.
7. **Build the Gradio dashboard** from your trained models. It should
   never retrain anything at startup. At minimum, let you compare the
   three models with a prediction vs. actual plot, see feature/target
   distributions, and, for classification, move a threshold slider to
   watch the confusion matrix and a business-cost number change.
8. **Submit**: see Submission below.

Everything else (how you structure your code, what you name things beyond
what `main.py` requires, how you organize your pipeline) is your call to
make and be able to explain.

## Setup

Same environment workflow as Session 2's environment check.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS
```
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Fork this repo (https://github.com/ami232/daiist-assignment-1), then clone
**your fork**:

```bash
git clone https://github.com/<your-github-username>/<your-fork-name>.git
cd <your-fork-name>
uv sync
```

> If `uv sync` can't install PyTorch locally (older Intel Macs), use GitHub
> Codespaces on your fork instead: same environment, runs in the browser.

### Adding dependencies

The starting dependencies (numpy, pandas, scikit-learn, torch, gradio,
plotly, matplotlib, ...) cover everything the assignment needs, but if you
want something else:

```bash
uv add <package-name>
```

This updates `pyproject.toml` and `uv.lock`. Commit both so `uv sync`
reproduces your exact environment for anyone else, including CI (below).

## Running your work

```bash
uv run python main.py train    # runs train.py, or train.ipynb if you used a notebook
uv run python main.py app      # runs app.py, or app.ipynb if you used a notebook, opens in your browser
```

`main.py` looks for `<stage>.py` first, then `<stage>.ipynb`, so name your
training code `train.py` or `train.ipynb`, and your app `app.py` or
`app.ipynb`, at the repo root. Pick whichever format fits each stage (a
notebook for one and a script for the other is fine). Beyond that naming
requirement, how you build them is up to you, but each must run start to
finish with no manual steps.

A GitHub Actions workflow (`.github/workflows/verify.yml`) runs both
commands automatically on every push and pull request. It's a sanity check
that your submission runs, not a grading mechanism, and it fails if
`main.py train` errors or `main.py app` doesn't come up and respond within
its startup window.

## Submission

1. Push your finished branch to your fork.
2. Open a pull request from your fork to the original repo:
   https://github.com/ami232/daiist-assignment-1
3. Submit the PR link on Blackboard.
4. Also upload a zipped copy of your repo to Blackboard, as a backup in
   case your fork or the PR becomes unavailable.

## Grading

**Hard requirement:** your training pipeline and Gradio app must each run
end-to-end via `main.py` with no manual intervention. A submission that
fails this can't pass the assignment, regardless of everything else below.

| Component | Weight | What it checks |
|---|---|---|
| Dataset & business framing | 20% | Is the framing sensible, and does it actually drive concrete pipeline decisions (split strategy, threshold metric, etc.) rather than just narrating them? |
| Feature engineering & preprocessing | 30% | Quality and justification of what you built, not just its presence. This is the largest component, since most of your effort should go here. |
| Three-method comparison | 30% | Do scikit-learn / manual PyTorch / standard PyTorch agree on the same split? If not, is that investigated and explained honestly rather than hidden? |
| Gradio dashboard | 20% | All required views present, working off your pipeline's real artifacts, nothing retrained at startup. |
| **× Written Comprehension Check** | **0–100%** | Multiplies the subtotal |

REPORT.md isn't graded as its own row: it's where the other four
components' content lives, so its quality is already captured by them. The
four weighted rows above sum to 100% and form your subtotal; the Written
Comprehension Check then multiplies that subtotal. A strong submission
paired with a weak comprehension-check score gets scaled down accordingly,
since the check verifies the understanding this brief keeps asking for.

Coefficients above are a starting proposal sized to relative workload
(feature engineering and the three-method comparison carry the most work),
not settled policy. Expect them to be confirmed before the deadline.

## Generative AI use

Per the syllabus AI Policy: disclosed AI use is fine and must be stated in
REPORT.md's disclosure section. Within that policy, here's how it applies
to this assignment specifically:

- **Fine to use AI for**: boilerplate and common operations, like loading
  a dataset, saving a trained model, and especially plotting and
  presenting results in the Gradio dashboard.
- **Use your own judgement for**: the decisions that are the point of this
  assignment, like which features to design, how to frame the business
  problem, and the implications of your design choices. AI can write the
  code for a decision, but the decision itself has to be yours.
- **REPORT.md**: the ideas and findings in it must be your own. AI may
  help with formatting, not with generating the analysis or conclusions.

None of this changes what's expected of you: you have to be able to
explain every decision in your submission as if you made it yourself,
because you did. Using a tool to help write it doesn't transfer the
understanding requirement to the tool.

## Business case

A buyer's agent in Ames checks an asking price before a client writes an offer. The model estimates the closing price from houses already sold, using only the house and the month and year of the search. It does not set the offer. If the ask sits well above the estimate, the agent calls the ask aggressive.

The target is the recorded close, including family, abnormal, allocation, and adjoining-land sales, so the number is a conservative check rather than a fair-market appraisal. The split is 2006–2008 to train, 2009 to choose the penalty and learning rate, and 2010 to test, because a 2010 search cannot use 2010 closes. Models are compared by dollar RMSE after undoing the log, with dollar MAE beside it. There is no class threshold. Flagging an ask that beats the estimate by more than the typical dollar miss is the decision rule, but the file has no asking prices, so only the match to the eventual close can be measured.

The same account is in `REPORT.md`. The notebook implements the rows, the split, the target, and the columns that follow from it. The three models are not trained yet.

## Preprocessing

`train.ipynb` prepares the Ames sales and stops.

### Rows

De Cock flags five houses with living area above 4,000 sq ft. Three are partial sales in Edwards (orders 1499, 2181, and 2182) priced far below what the size would suggest. Those three are removed before the split. The two Northridge houses stay, including the abnormal sale at $745,000, because it matches the normal Northridge sale at $755,000.

Three date errors are corrected on the whole file. They are typos, not values learned from the training years:

- One garage year is 2207. That house was built in 2006 and sold in 2007, so the year is set to 2007.
- One house is marked remodeled in 2001 and built in 2002. The remodel year is set to 2002.
- One kept sale has a remodel year after the sale year (two of the dropped partials had the same glitch). A later year is not known at the sale, so it is set back to the sale year.

### Split and target

- Train: sales from 2006–2008 (1,938 rows).
- Validation: 2009 (648 rows), held out for the penalty and the learning rate.
- Test: 2010 (341 rows). Sales stop in July, so this year is incomplete.

Training uses `log(SalePrice)`. Raw price skew is about 1.74; log price skew is about 0. Dollar error is computed after exponentiating. The naive baseline is the training geometric mean, about $167,485.

Neighborhood medians, rare levels, dummy columns, and the scaler are fit on 2006–2008 only.

### What a blank means

A blank in alley, basement quality and type, fireplace quality, garage type and finish, fence, and masonry type means that part of the house is absent. Those cells become `"None"`. No veneer with a missing area becomes area 0. No basement becomes basement area 0 and basement baths 0. No garage becomes garage area 0.

Real gaps are filled from the training years:

- Lot frontage: median of the same neighborhood, then the overall training median.
- Masonry area when a veneer type is present: training median of that type.
- Basement exposure and second finish type, when a basement exists: most common value among training houses with a basement.
- The one missing electrical system: most common training value.
- Two detached garages with incomplete records: median or most common value among training detached garages.

### Columns left out

- `Order` and `PID` are identifiers.
- `Sale Condition` and `Sale Type` describe the deal, which is not known beforehand. Sale condition is used only to find the three partial sales.
- `Street`, `Utilities`, `Condition 2`, `Roof Matl`, and `Heating` are almost constant.
- `Exterior 2nd` repeats the exterior. `Pool QC` and `Misc Feature` are almost entirely blank; `Pool Area` and `Misc Val` stay.
- `1st Flr SF`, `2nd Flr SF`, and `Low Qual Fin SF` add up to `Gr Liv Area`.
- `Bsmt Unf SF` is the leftover of the basement total minus the finished areas.
- `Garage Cars` repeats `Garage Area`.
- `Year Built`, `Year Remod/Add`, and `Garage Yr Blt` are replaced by ages.

### Features added

- `Age` = year sold − year built.
- `RemodAge` = year sold − year remodeled.
- `GarageAge` = year sold − garage year, or 0 when there is no garage. Keeping the raw year would require inventing a construction date for those houses.

`Mo Sold` and `Yr Sold` stay, because the month and year of the estimate are known. Year stays a number so 2009 and 2010 can extend a trend. A year indicator could not: those years never appear in training, so the indicator would be 0. Month is not a number, because December is not twelve times January.

### Encoding

Ordered fields become integers one step apart. Equal spacing is an assumption. A higher number is the more standard or more complete state, except land slope, where a higher number is steeper.

| Field | Order |
|---|---|
| Quality fields | None 0, Po 1, Fa 2, TA 3, Gd 4, Ex 5 |
| Basement exposure | None 0, No 1, Mn 2, Av 3, Gd 4 |
| Basement finish type | None 0, Unf 1, LwQ 2, Rec 3, BLQ 4, ALQ 5, GLQ 6 |
| Function | Sal 0, Sev 1, Maj2 2, Maj1 3, Mod 4, Min2 5, Min1 6, Typ 7 |
| Garage finish | None 0, Unf 1, RFn 2, Fin 3 |
| Lot shape | IR3 0, IR2 1, IR1 2, Reg 3 |
| Land slope | Gtl 0, Mod 1, Sev 2 |
| Paved drive | N 0, P 1, Y 2 |
| Fence | None 0, MnWw 1, GdWo 2, MnPrv 3, GdPrv 4 |
| Central air | N 0, Y 1 |

Quality fields are exterior, basement, heating, kitchen, fireplace, and garage quality, plus exterior and basement condition.

The remaining categories are one-hot encoded, including `MS SubClass` (it is a code, not a measurement) and month. A level with fewer than 10 training rows is renamed `Other`. Levels are sorted A to Z and the first is left out, so the intercept stands for it. After that grouping, 2009 and 2010 contain no category that training never saw.

Every column, including the indicators, is standardized with a scaler fit on the training years. The matrix has 159 columns.
