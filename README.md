# Assignment Overview

## Epigenetic Age Prediction from DNA Methylation
 
## Ιntroduction
Understanding human aging can result in multiple implications while measuring it based on molecular profiles. These range across several fields, such as disease prevention and treatment, forensics, and the improvement of life expectancy and quality of life. Chronological age, additionally, has been linked to changes in DNA methylation, which refers to the addition of methyl groups to cytosine bases at CpG dinucleotides, that change systematically with age across the human genome.          
Regarding methylome, it can be measured using microarray technologies and used to train regression models that predict chronological age from DNA methylation profiles. These models are known as epigenetic clocks and have broad applications in biology and medicine, such as estimating biological aging, detecting accelerated aging in disease, and studying the mechanisms of lifespan.        
In this assignment, a complete machine learning pipeline for epigenetic age prediction from DNA methylation data was developed, with the aim of better understanding how the methylome changes with age and whether human aging rates can be quantified and compared.      
The assignment is separated into 4 main different implementation steps:     
`a) the preparation of the dataset and preprocessing pipeline,`     
`b) the training of baseline models to predict age from DNA methylation data,`     
`c) feature selection based on model performance (RMSE, MAE, R², and Pearson r with 95% confidence intervals) while ensuring no overfitting or underfitting,` and     
`d) the final tuning and evaluation of the selected model using the optimal feature set.`      

---

## Dataset (Before Preprocessing)

The `GSE40279` (Hannum et al., 2013), a publicly available dataset of whole-blood DNA methylation profiles measured using the Illumina 450K microarray, was used.       

Dataset  -->  `656 samples` from individuals aged `19–101 years`, with total features (positions) an approximation of `480,000 CpG positions`.        
Each `CpG` site --> Represented by a beta value `(range 0-1)`, indicating if having full or no methylation.      

## Dataset (After Preprocessing)
| **Files**   | **Samples** | **Purpose** | 
| ----- | :---: | :---: | 
| development_data .csv |  456 | Training, preprocessing, feature selection, cross-validation |
| evaluation_data. csv | 100  |  Final evaluation |

### Data Structure
a) Each `row represents a sample` (GEO ID)  
b) Columns include:  
  ▻CpG features `(cgXXXXXX)`  
  ▻`age` (target variable)  
  ▻`sex` (categorical)  
  ▻`ethnicity` (categorical)  

#### Data Construction
From the original ~480K CpGs, the 1000 CpG positions were selected by absolute Spearman correlation with age across the dataset.

#### Missing Values
To stimulate real world conditions, some CpG values were randomly set to NaN.

---

## Structure of repository
<ul>
<li> src/ —functions.py (All pipeline functions) </li>
<li>notebooks/ — Jupyter notebook (data_exploration.ipynb and model_analysis.ipynb (final) with all the Tasks execution)</li>
<li>data/ — dataset files (.gitignore for data commitment)</li>
<li>models/ — saved best model (best_model.okl)</li>
<li>figures/ — generated plots (Saved plots that were generated from data_exploration.ipynb)</li>
</ul>

---

## Libraries of use

```
numpy, pandas,scikit-learn,scipy,mrmr-selection,matplotlib,pickle
```

---

## Best Selected model
Following hyperparameter tuning and final evaluation on the locked evaluation set, `ElasticNet` had the lowest bootstrap RMSE mean **(5.0549)** among the three tuned models (ElasticNet, SVR, BayesianRidge).   

*ElasticNet --> Selected as the final model*

---

## Assignment Tasks 
[✓]  Task 1  
[✓] Task 2  
[✓] Task 3  
[✓] Task 4  
[X] Bonus A   
[X] Bonus B  

