# Assignment Overview

## Epigenetic Age Prediction from DNA Methylation

## Here introduction

## The Dataset before pre-processing

Source You will work with GSE40279 (Hannum et al., 2013), a publicly available dataset measuring whole-blood DNA methylation using the Illumina 450K microarray. The original dataset comprises 656 samples from individuals aged 19–101 years. The 450K array measures beta values — the proportion of methylated reads at each CpG site, ranging from 0 (fully unmethylated) to 1 (fully methylated) — at approximately 480,000 CpG positions.

## The Dataset after pre-processing
| Files   | Samples | Purpose | 
| ----- | :---: | :---: | 
| development_data .csv |  456 | Used in preprocessing, feature selection, training, cross-validation tuning. |
| evaluation_data. csv | 100  |  Final evaluation. |


## Structure of repository

<ul>
<li> src/ —functions.py (All pipeline functions) </li>
<li>notebooks/ — Jupyter notebook (data_exploration.ipynb with all the Tasks execution)</li>
<li>data/ — dataset files (.gitignore for data commitment)</li>
<li>models/ — saved best model (best_model.okl)</li>
<li>figures/ — generated plots (Savied plots that were generated from data_exploration.ipynb)</li>
</ul>


## Libraries of use

```
numpy, 
pandas,
scikit-learn,
scipy,
mrmr-selection,
optuna,
matplotlib,
pickle
```

## Selected model

## Assignment Tasks 
[✓]  Task 1  
[✓] Task 2  
[✓] Task 3  
[✓] Task 4  
[X] Bonus A   
[X] Bonus B  

