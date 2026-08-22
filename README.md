# MSc Project

This repository contains the code and input files needed to reproduce the results and analyses presented in my MSc Project; Bayesian Inference and Prior Sensitivity: Modelling Recent Fertility Dynamics in South Korea.

## Main files

- `fit_changepoint.Rmd`: main analysis document.
- `data/fertility.csv`: South Korean annual total fertility rate input file.
- `stan/changepoint.stan`: changepoint model.
- `stan/nullmodel.stan`: Phase-II-only model; used for comparison.
- `bayesTFR_output/`: saved output from `bayesTFR` package; used to extract empirical Bayes hyperparameter medians.

The `bayesTFR_output/` folder contains the fitted output from the preliminary analysis using `bayesTFR` package. The main R Markdown file loads this saved output and extracts posterior medians of the global Phase III hyperparameters used to construct the empirical Bayes prior. The chains from `bayesTFR` are not rerun because they are slow and are only needed to obtain these four hyperparameter medians in the dissertation.

The fertility data were initially obtained from KOSIS using its OpenAPI service. The cleaned CSV is included directly so the analysis can be reproduced without requiring a personal API key.

To run the analysis, open `fit_changepoint.Rmd` in RStudio or VS Code and knit/run the chunks from the project root folder.
