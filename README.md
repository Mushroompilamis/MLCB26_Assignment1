# Assignment Overview

## Epigenetic Age Prediction from DNA Methylation

## Ιntroduction
Understanding human aging can result in multiple implications while measuring it from molecular profiles. Its implications vary in many fields, such as disease prevention and treatment, forensics, amelioration of living expectancy and extension of life. Chronological age, additionally has been linked to changes in DNA methylation, meaning the addition of methyl groups to cytosine bases at CpG dinucleotides, which changes systematically with age across the human genome. However, the methylome has not yet been used to measure and compare human aging rates, but its signal can be signal can be measured with microarray technology and used to train regression models that predict chronological age from methylation profiles. Such models are called epigenetic clocks, and have broad applications in biology and medicine: measuring biological ageing rate, identifying accelerated ageing in disease, and understanding the epigenetic basis of lifespan.   
Therefore, in this assignment, a complete machine learning pipeline for epigenetic age prediction from DNA methylation data was created, to better understand how the methylome ages and to determine whether human aging rates can be quantified and compared.   
The assignment is separated into 4 different implementation steps:     
`a) the dataset and preprocessing infrastructure establishment`,       
`b) which is fed into different baseline models to predict age from DNA data by,`      
`c)  feature selection in which by using the model with better results in RMSE, MAE, R², and Pearson r with 95% confidence intervals (CI) and does not over or underfit of the previous steps and then`       
`d), by using the selected feature set from the previous step, the final tuning and evaluation of the model of use occurs.`   

---

## Dataset (Before Preprocessing)

The `GSE40279` (Hannum et al., 2013), a publicly available dataset of whole-blood DNA methylation profiles measured using the Illumina 450K microarray, was used.       

Dataset  -->  `656 samples` from individuals aged `19–101 years`, with total features (positions) an approximation of `480,000 CpG positions`.        
Each `CpG` site --> Represented by a beta value `(range 0-1)`, indicatin if having full or no methylation.      

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
From the original ~480K CpGs, the 1000 CpG positions were selected by absolutre Spearman correlation with age across the dataset.

#### Missing Values
To stimulate real world conditions, some CpG values were randomly set to NaN.

---

## Structure of repository
<ul>
<li> src/ —functions.py (All pipeline functions) </li>
<li>notebooks/ — Jupyter notebook (data_exploration.ipynb with all the Tasks execution)</li>
<li>data/ — dataset files (.gitignore for data commitment)</li>
<li>models/ — saved best model (best_model.okl)</li>
<li>figures/ — generated plots (Savied plots that were generated from data_exploration.ipynb)</li>
</ul>

---

## Libraries of use

```
numpy, pandas,scikit-learn,scipy,mrmr-selection,matplotlib,pickle
```

---

## Best Selected model

---

## Assignment Tasks 
[✓]  Task 1  
[✓] Task 2  
[✓] Task 3  
[✓] Task 4  
[X] Bonus A   
[X] Bonus B  

