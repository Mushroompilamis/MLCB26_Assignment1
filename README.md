# Assignment Overview

## Epigenetic Age Prediction from DNA Methylation

## Ιntroduction
Understanding human aging can result in multiple implications while measuring it from molecular profiles. Its implications vary in many fields, such as disease prevention and treatment, forensics, amelioration of living expectancy and extension of life. Chronological age, additionally has been linked to changes in DNA methylation, meaning the addition of methyl groups to cytosine bases at CpG dinucleotides, which changes systematically with age across the human genome. However, the methylome has not yet been used to measure and compare human aging rates, but its signal can be signal can be measured with microarray technology and used to train regression models that predict chronological age from methylation profiles. Such models are called epigenetic clocks, and have broad applications in biology and medicine: measuring biological ageing rate, identifying accelerated ageing in disease, and understanding the epigenetic basis of lifespan.   
Therefore, in this assignment, a complete machine learning pipeline for epigenetic age prediction from DNA methylation data was created, to better understand how the methylome ages and to determine whether human aging rates can be quantified and compared.   
The assignment is separated into 4 different implementation steps: a) the dataset and preprocessing infrastructure establishment b) which is fed into different baseline models to predict age from DNA data by, c)  feature selection in which by using the model with better results in RMSE, MAE, R², and Pearson r with 95% confidence intervals (CI) and does not over or underfit of the previous steps and then d), by using the selected feature set from the previous step, the final tuning and evaluation of the model of use occurs.

## Dataset before pre-processing

΅Working with GSE40279 (Hannum et al., 2013), a publicly available dataset measuring whole-blood DNA methylation using the Illumina 450K microarray. The original dataset comprises 656 samples from individuals aged 19–101 years. The 450K array measures beta values — the proportion of methylated reads at each CpG site, ranging from 0 (fully unmethylated) to 1 (fully methylated) — at approximately 480,000 CpG positions.

## Dataset after pre-processing
| **Files**   | **Samples** | **Purpose** | 
| ----- | :---: | :---: | 
| development_data .csv |  456 | Used in preprocessing, feature selection, training, cross-validation tuning. |
| evaluation_data. csv | 100  |  Final evaluation. |

### Structure of data
a) Each row is a sample (labelled by GEO sample ID). Each column is either a CpG feature  (named cgXXXXXXXX), an age column (integer years, the regression target), a sex column  (F/M), or an ethnicity column. The 1000 CpG features included in the datasets were selected from the full 480K array by absolute Spearman correlation with age across all samples.   
b) To simulate real-world data quality, some of the CpG values have been randomly set to NaN.

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

