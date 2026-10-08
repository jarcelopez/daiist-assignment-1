# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Javier Arce López
- **Student ID**: 20158
- **Email**: jarcel.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

## Business / real-life framing

A buyer's agent in Ames uses this model before a client writes an offer. The seller has named an asking price. The agent wants an estimate of what that house would close for, based on houses that have already sold, and using only what is known before the deal: the house itself, and the month and year of the search. The model does not choose the offer. If the ask sits well above the estimate, the agent tells the client the ask is aggressive.

The target is the recorded closing price. Family, abnormal, allocation, and adjoining-land sales stay, because those payments happened. The estimate therefore sits a bit low for an ordinary retail purchase. The agent uses it as a conservative check, not as an appraisal of fair market value.

Three partial sales of houses above 4,000 sq ft are removed. Those prices are not comparisons a client could use. The two other houses above that size stay, because their prices match each other.

The split follows the calendar. An agent pricing a house in 2010 does not yet have the 2010 sales. The model is fit on 2006–2008, the penalty and the learning rate are chosen on 2009, and 2010 is the test. A random split would train on sales that had not closed yet. `Sale Condition` and `Sale Type` are not features: they describe how the deal was done. Month and year are features: the agent knows when the client is shopping.

There is no class threshold. A large dollar miss is the mistake that leads to a bad offer, so models are compared by dollar RMSE after the log prediction is converted back to dollars. Dollar MAE is reported next to it so the typical miss is visible too. A practical rule, once that error is known, is to flag an asking price that exceeds the estimate by more than the typical dollar miss. This file has closing prices only, so that rule cannot be tested. What can be tested is whether the estimate matches the eventual close.

Training uses log price because raw prices are strongly right-skewed, and the same dollar miss means something different on a cheap house and an expensive one. The reported errors are still in dollars, which is the unit of the offer.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
