#Libraries
#######################################################################################
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import ElasticNet,BayesianRidge
from sklearn.model_selection import RandomizedSearchCV,KFold
from sklearn.svm import SVR
from scipy.stats import pearsonr, alpha
from scipy.stats import spearmanr
from scipy.stats import loguniform,uniform
from mrmr import mrmr_regression
import optuna
from mrmr import mrmr_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score,f1_score,matthews_corrcoef,roc_auc_score,average_precision_score,confusion_matrix,roc_curve)


#######################################################################################
#######################################################################################
###########TASK 1###########
# 1.2 Preprocessing
#####################################################################################
#Loading the files of dev and eval
def loading_files(dev_path,eval_path):
    dev_data = pd.read_csv(dev_path, index_col=0)
    print("Development set", dev_data.shape, dev_data.head())
    print("\n")
    eval_data = pd.read_csv(eval_path, index_col=0)
    print("Evaluation set (locked)", eval_data.shape, eval_data.head())
    print("Samples of each data:",len(dev_data), len(eval_data))

    return dev_data,eval_data

#######################################################################################
#Function for splitting
def split_data(dev_data, eval_data):
    X=dev_data.drop(columns=["age"])
    y=dev_data["age"]
    #Bins Creation for stratification
    y_bins=pd.qcut(y,5,duplicates='drop')

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y_bins)

    train_data=X_train.copy()
    train_data["age"]=y_train

    val_data=X_val.copy()
    val_data["age"]=y_val

    #Percentage Split
    total_data=len(train_data) + len(val_data)
    train_pre= len(train_data)/ total_data*100
    val_pre= len(val_data)/ total_data*100

    #Creating table of the data split information for better visualization
    table_summary = pd.DataFrame({
    "Split": ["Train", "Validation"],
    "Samples": [len(train_data), len(val_data)],
    "Percentage": [train_pre, val_pre]})
    print(table_summary)
    return train_data,val_data, eval_data

#######################################################################################
# Feature Scaring & Categorical Features process setting
def feature_gathering(data):
    metadata=[i for i in["sex","ethnicity"]if i in data.columns]
    cpg=[i for i in data.columns if i.startswith("cg")]
    age="age"
    return metadata,cpg,age

def feature_set(data,sets="all"):
    metadata,cpg, _= feature_gathering(data)
    if sets=="metadata":
        return metadata
    elif sets=="cpg":
        return cpg
    elif sets=="all":
        return metadata + cpg
    else:
        raise ValueError(f"Invalid set: {sets}")

def split_feature(data,features,age="age"):
    X=data[features].copy()
    y=data[age].copy()
    return X,y

#######################################################################################
#Building the processor that will be applied across all splits
#fit on training
#Application on val and later on eval
#Filling the missing values with median
def  preprocessor_pipeline(cpg,metadata):
    transformers = []
    if len(cpg)>0:
        #Feature scaling: StandardScaler to CpG features inside a sklearn Pipeline
        cpq_pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("cpg",cpq_pipeline,cpg))

    if len(metadata)>0:
        #Categorical features: Metadata columns are categorical , use of encoding for numerical representation
         metadata_pipeline=Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),
             ("encoder", OneHotEncoder(handle_unknown="ignore",sparse_output=False))
         ])
         transformers.append(("metadata",metadata_pipeline,metadata))

    if len(transformers) == 0:
        raise ValueError("No valid feature found")

    preprocessor=ColumnTransformer(transformers=transformers)
    return preprocessor
#######################################################################################
# By one and all feature matrix
def feature_matrices(train_data, val_data, eval_data):
    # Taking the features from previous function
    metadata_features, cpg_features, target = feature_gathering(train_data)
    all_features = metadata_features + cpg_features

    feature_groups = {"metadata": metadata_features, "cpg": cpg_features, "all": all_features }
    results = {}
    summary_rows = []

    for char, features in feature_groups.items():
        if len(features) == 0:
            print(f"Skipping {char}: no features found.")
            continue

        X_train, y_train = split_feature(train_data, features, age=target)
        X_val, y_val = split_feature(val_data, features, age=target)
        X_eval, y_eval = split_feature(eval_data, features, age=target)

       # Preprocessor for each possible group build
        if char == "metadata":
            preprocessor = preprocessor_pipeline(cpg=[], metadata=features)
        elif char == "cpg":
            preprocessor = preprocessor_pipeline(cpg=features, metadata=[])
        else:
            preprocessor = preprocessor_pipeline(cpg=cpg_features, metadata=metadata_features)

        # Fit on train only
        train_processed = preprocessor.fit_transform(X_train)
        val_processed = preprocessor.transform(X_val)
        eval_processed = preprocessor.transform(X_eval)

        # Store everything
        results[char] = { "features": features,
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_eval": X_eval, "y_eval": y_eval,
            "X_train_processed": train_processed, "X_val_processed": val_processed, "X_eval_processed": eval_processed,
            "preprocessor": preprocessor
        }

        # Summary row
        summary_rows.append({
            "feature_set": char,
            "train": X_train.shape,
            "train_processed": train_processed.shape,
            "val": X_val.shape,
            "val_processed": val_processed.shape,
            "evaluation": X_eval.shape,
            "evaluation_processed": eval_processed.shape
        })

    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df

#######################################################################################
###############################
#1.3 Exploratory analysis
# Creating the dataset summary function with addition information and checking of missing values
def data_summary(train_data, val_data, eval_data, results=None, features="all"):
    rows = []

    processed_map = {}
    processed_missing_map = {}

    if results is not None and features in results:
        processed_map = {
            "Train": results[features]["X_train_processed"].shape,
            "Validation": results[features]["X_val_processed"].shape,
            "Evaluation": results[features]["X_eval_processed"].shape
        }

        processed_missing_map = {
            "Train": pd.DataFrame(results[features]["X_train_processed"]).isna().sum().sum(),
            "Validation": pd.DataFrame(results[features]["X_val_processed"]).isna().sum().sum(),
            "Evaluation": pd.DataFrame(results[features]["X_eval_processed"]).isna().sum().sum()
        }

    for name, data in [("Train", train_data), ("Validation", val_data), ("Evaluation", eval_data)]:
        rows.append({
            "dataset": name,
            "samples": len(data),
            "age_mean ± std": f"{data['age'].mean():.1f} ± {data['age'].std():.1f}",
            "age_range": f"{data['age'].min():.0f} - {data['age'].max():.0f}",
            "sex_balance": data["sex"].value_counts().to_dict() if "sex" in data.columns else {},
            "ethnicity_balance": data["ethnicity"].value_counts().to_dict() if "ethnicity" in data.columns else {},
            "total_missing_values_raw": int(data.isna().sum().sum()),
            "processed_shape": processed_map.get(name, None),
            "total_missing_values_processed": processed_missing_map.get(name, None)
        })
    return pd.DataFrame(rows)

#######################################################################################
#Age distribution plot
def age_distribution_plot(age):
    plt.figure(figsize=(10,5))
    plt.hist(age,bins=30, color="blue", edgecolor="black")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Histogram of age distribution (dev_set)")
    plt.savefig("../figures/age_distribution_dev.png",dpi=250,bbox_inches="tight")
    plt.show()
    plt.close()

#Checking distribution per split
def age_split_plot(train_data, val_data, eval_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, data) in zip(axes, [("Train", train_data), ("Validation", val_data), ("Evaluation", eval_data)]):
        ax.hist(data["age"], bins=20, color="blue", edgecolor="black")
        ax.set_title(name)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("../figures/age_histogram_split_data.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

#######################################################################################
#######################################################################################
###########TASK 2###########
# 2.1  OLS LinearRegression across feature sets

#OlS Linear Regression for the train_data

def ols_model(X_train,y_train,X_val):
    model=LinearRegression()
    model.fit(X_train, y_train)
    y_predict=model.predict(X_val)
    return model,y_predict
#######################################################################################

#Evaluation function run to each model separately on the validation set using bootstrap resampling of the validation set(1000 resamples, seed=42)
def evaluation_of_model(y_true, y_predict, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    # RMSE, MAE, R², and Pearson r
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    pearson_corr_scores = []
    indices = np.arange(len(y_true))

    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        y_true_sample = y_true[sample_idx]
        y_pred_sample = y_predict[sample_idx]

        rmse = np.sqrt(mean_squared_error(y_true_sample, y_pred_sample))
        mae = mean_absolute_error(y_true_sample, y_pred_sample)
        r2 = r2_score(y_true_sample, y_pred_sample)
        r, _ = pearsonr(y_true_sample, y_pred_sample)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        pearson_corr_scores.append(r)

    rmse_scores = np.array(rmse_scores)
    mae_scores = np.array(mae_scores)
    r2_scores = np.array(r2_scores)
    pearson_corr_scores = np.array(pearson_corr_scores)

    full_rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    full_mae = mean_absolute_error(y_true, y_predict)
    full_r2 = r2_score(y_true, y_predict)
    full_r, _ = pearsonr(y_true, y_predict)

    # Addind and with 95% confidence intervals (CI).
    final_results = {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(ddof=1),
        "rmse_ci": np.percentile(rmse_scores, [2.5, 97.5]),
        "rmse_full": full_rmse,
        "rmse_scores": rmse_scores,

        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(ddof=1),
        "mae_ci": np.percentile(mae_scores, [2.5, 97.5]),
        "mae_full": full_mae,

        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(ddof=1),
        "r2_ci": np.percentile(r2_scores, [2.5, 97.5]),
        "r2_full": full_r2,
        "r2_scores": r2_scores,

        "pearson_mean": pearson_corr_scores.mean(),
        "pearson_std": pearson_corr_scores.std(ddof=1),
        "pearson_ci": np.percentile(pearson_corr_scores, [2.5, 97.5]),
        "pearson_full": full_r,
    }
    print("Evaluation completed")
    return final_results

# Train OLS LinearRegression model per feature set
def ols_all_featuresets(results, n_bootstrap=1000, seed=42):
    output = {}
    summary_rows = []

    for featureset in ["metadata", "cpg", "all"]:
        if featureset not in results:
            continue

        X_train = results[featureset]["X_train_processed"]
        X_val = results[featureset]["X_val_processed"]
        y_train = results[featureset]["y_train"]
        y_val = results[featureset]["y_val"]

        model, y_pred = ols_model(X_train, y_train, X_val)
        metrics = evaluation_of_model(y_val, y_pred, n_bootstrap=n_bootstrap, seed=seed)

        output[featureset] = {
            "model": model,
            "y_pred": y_pred,
            "metrics": metrics
        }

        summary_rows.append({
            "feature_set": featureset,
            "RMSE": round(metrics["rmse_full"], 4),
            "RMSE_95CI": f"{metrics['rmse_ci'][0]:.4f} - {metrics['rmse_ci'][1]:.4f}",
            "MAE": round(metrics["mae_full"], 4),
            "MAE_95CI": f"{metrics['mae_ci'][0]:.4f} - {metrics['mae_ci'][1]:.4f}",
            "R2": round(metrics["r2_full"], 4),
            "R2_95CI": f"{metrics['r2_ci'][0]:.4f} - {metrics['r2_ci'][1]:.4f}",
            "Pearson_r": round(metrics["pearson_full"], 4),
            "Pearson_95CI": f"{metrics['pearson_ci'][0]:.4f} - {metrics['pearson_ci'][1]:.4f}",
        })
        print(f"Train OLS LinearRegression for {featureset}")
    summary_df = pd.DataFrame(summary_rows)
    return output, summary_df

#######################################################################################
###########################################
#2.2  Three regression models at default hyperparameters

#MODELS with default parameters

#ElasticNet — L1+L2 regularised linear regression
def model_elasticnet(X_train, y_train, X_val):
    print("############### Training ElasticNet ###############")
    model = ElasticNet()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_val)
    return model, y_predict

# SVR — Support Vector Regression with RBF kernel
def svr_model(X_train, y_train, X_val):
    print("############### Training SVR ###############")
    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)
    y_predict = model.predict(X_val)
    return model, y_predict

# BayesianRidge — Bayesian linear regression
def bayesianridge_model(X_train, y_train, X_val):
    print("############### Training BayesianRidge ###############")
    model = BayesianRidge()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_val)
    return model, y_predict


def report_table(results):
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "RMSE mean": f"{res['rmse_mean']:.4f}",
            "RMSE 95% CI": f"{res['rmse_ci'][0]:.4f} - {res['rmse_ci'][1]:.4f}",
            "MAE mean": f"{res['mae_mean']:.4f}",
            "MAE 95% CI": f"{res['mae_ci'][0]:.4f} - {res['mae_ci'][1]:.4f}",
            "R^2 mean": f"{res['r2_mean']:.4f}",
            "R^2 95% CI": f"{res['r2_ci'][0]:.4f} - {res['r2_ci'][1]:.4f}",
            "Pearson r mean": f"{res['pearson_mean']:.4f}",
            "Pearson r 95% CI": f"{res['pearson_ci'][0]:.4f} - {res['pearson_ci'][1]:.4f}"
        })

    df = pd.DataFrame(rows)
    print("Report table of model performance:")
    print(df.to_string(index=False))
    return df


def bootstrap_boxplots(results, path="../figures/bootstrap_boxplots_rmse_r2.png"):
    os.makedirs("../figures", exist_ok=True)

    model_names = list(results.keys())
    rmse_data = [results[m]["rmse_scores"] for m in model_names]
    r2_data = [results[m]["r2_scores"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].boxplot(rmse_data, patch_artist=True)
    axes[0].set_title("Bootstrap RMSE Distribution")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xlabel("Model")
    axes[0].set_xticks(range(1, len(model_names) + 1))
    axes[0].set_xticklabels(model_names, rotation=20)

    axes[1].boxplot(r2_data, patch_artist=True)
    axes[1].set_title("Bootstrap R² Distribution")
    axes[1].set_ylabel("R²")
    axes[1].set_xlabel("Model")
    axes[1].set_xticks(range(1, len(model_names) + 1))
    axes[1].set_xticklabels(model_names, rotation=20)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

#Comparison
def compare_with_ols(model_results, ols_results):
    rows = []
    ols_rmse = ols_results["rmse_full"]
    ols_r2 = ols_results["r2_full"]

    for name, res in model_results.items():
        beats_ols = (res["rmse_full"] < ols_rmse) and (res["r2_full"] > ols_r2)
        rows.append({
            "Model": name,
            "RMSE": round(res["rmse_full"], 4),
            "R²": round(res["r2_full"], 4),
            "Beats OLS ": beats_ols
        })

    return pd.DataFrame(rows)

#Checking for overfitting
def train_vs_val_all_models(models, X_train, y_train, X_val, y_val):
    rows = []
    for name, model in models.items():
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        rows.append({
            "Model": name,
            "Train RMSE": train_rmse,
            "Val RMSE": val_rmse,
            "Train R^2": train_r2,
            "Val R^2": val_r2
        })

    return pd.DataFrame(rows)

#######################################################################################
#######################################################################################
###########TASK 3###########
#3.1  Stability Selection
def stability_selection(train_data,resamples=50,samples_frac=0.80,top_k=200,seed=42):
    cpg_cols=[j for j in train_data.columns if j.startswith("cg")]
    counts=pd.Series(0, index=cpg_cols,dtype=int)
    for i in range(resamples):
        sub=train_data.sample(frac=samples_frac,replace=False,random_state=seed + i)
        #CpG within each subsample
        imputer=SimpleImputer(strategy="median")
        sub_cpg=pd.DataFrame(imputer.fit_transform(sub[cpg_cols]),columns=cpg_cols,index=sub.index)
        sub_age=sub["age"]

        correlations={}
        for j in cpg_cols:
            corr, _ = spearmanr(sub_cpg[j], sub_age)
            correlations[j] = abs(corr)
        top_features=sorted(correlations, key=correlations.get, reverse=True)[:top_k]
        counts[top_features]+=1

        if (i+1) % 10 == 0:
            print(f"Resample regeneration: {i+1}/{resamples} completed")
    stable=counts[counts > (resamples/2)]
    print(f"The stable features are: {len(stable)}")
    return stable.sort_values(ascending=False),counts.sort_values(ascending=False)

def frequencyplot(counts, path="../figures/stability_selection.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(counts.values, bins=20, edgecolor="black")
    plt.axvline(25, color="red", linestyle="--", label="threshold of 25")
    plt.xlabel("Selection  Frequency (out of 50 resamples) ")
    plt.ylabel("Number of CpGs")
    plt.title("Selection-frequency distribution across CpG features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

#######################################################################################
#3.2   Minimum Redundancy Maximum Relevance  (mRMR)
def choosing_k_value(train_data, val_data, k_values=None, path="../figures/mrmr_k_selection.png"):
    if k_values is None:
        k_values = [40,60,80,100,120,150,180,200]

    cpg_cols = [i for i in train_data.columns if i.startswith("cg")]
    results = []

    for k in k_values:
        print(f"Testing k: {k}")
        mrmr_features = mrmr_regression(
            X=train_data[cpg_cols],
            y=train_data["age"],
            K=k
        )
        # preprocess CpGs
        preprocessor = preprocessor_pipeline(cpg=mrmr_features, metadata=[])
        X_train_processed = preprocessor.fit_transform(train_data[mrmr_features])
        X_val_processed = preprocessor.transform(val_data[mrmr_features])

        # model of choice BR
        model = BayesianRidge()
        model.fit(X_train_processed, train_data["age"])
        y_val_predict = model.predict(X_val_processed)
        rmse = mean_squared_error(val_data["age"], y_val_predict) ** 0.5
        r2=r2_score(val_data["age"], y_val_predict)
        results.append({"K": k,"Validation RMSE": rmse,"Validation R^2": r2})
        print(f"K: {k} with Validation RMSE: {rmse:.4f} and Validation R^2: {r2:.4f}")

    #dataframe of results
    results = pd.DataFrame(results)
    best_row = results.loc[results["Validation RMSE"].idxmin()]
    b_k = best_row["K"]
    b_rmse = best_row["Validation RMSE"]
    b_r2 = best_row["Validation R^2"]
    print(f"\nThe best chosen K is: {b_k} with Validation RMSE: {b_rmse:.4f} and Validation R^2:{b_r2:.4f}")

    # Plot k vs RMSE
    plt.figure(figsize=(7, 4))
    plt.plot(results["K"], results["Validation RMSE"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Validation RMSE")
    plt.title("mRMR: Selection of optimal K")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return b_k, results


def mrmr_function(train_data, k):
    k=int(k)
    cpg_cols = [i for i in train_data.columns if i.startswith("cg")]
    imputer=SimpleImputer(strategy="median")
    X_train_cpg =pd.DataFrame(imputer.fit_transform(train_data[cpg_cols]),columns=cpg_cols,index=train_data.index)
    y_train = train_data["age"]

    mrmr_features = mrmr_regression(X=X_train_cpg, y=y_train, K=k)
    print(f"Selected mRMR {k} features.")

    top10_sfeatures = []
    for rank, feat in enumerate(mrmr_features[:10], start=1):
        corr, _ = spearmanr(X_train_cpg[feat], train_data["age"])
        top10_sfeatures.append({
            "Rank": rank,
            "Feature": feat,
            "Proxy importance (Spearman r)": abs(corr)
        })
    top10 = pd.DataFrame(top10_sfeatures)
    print("\nTop 10 selected features:")
    print(top10.to_string(index=False))
    return mrmr_features, top10

#######################################################################################
def overlapping_plot(stability_features, mrmr_features, path="../figures/feature_overlap.png"):
    set_stability = set(stability_features)
    set_mrmr = set(mrmr_features)
    stability = set_stability - set_mrmr
    overlap = set_stability & set_mrmr
    mrmr = set_mrmr - set_stability

    print("Stability: ", len(stability))
    print("Overlap: ", len(overlap))
    print("mRMR: ", len(mrmr))

    overlap_df = pd.DataFrame({
        "Group": ["Stability", "Overlap", "mRMR results"],
        "Count": [len(stability), len(overlap), len(mrmr)]
    })

    #####################
    plt.figure(figsize=(6, 5))
    plt.bar(overlap_df["Group"], overlap_df["Count"], edgecolor="black")
    plt.title("Overlap between the stability-selected and mRMR-selected feature sets")
    plt.ylabel("Number of Features")
    plt.xlabel("Feature Group")
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.show()
    plt.close()
    #####################

    return overlap_df

#######################################################################################
#3.3  Method comparison and selection
def feature_comparison(train_data,val_data,stable_features,mrmr_features,b_k):
    stable_cpg = list(stable_features.index) if hasattr(stable_features, "index") else list(stable_features)
    mrmr_cpg = list(mrmr_features)
    results = []

    for method, feature in [("Stability Selection", stable_cpg),("mRMR", mrmr_cpg)]:
        # preprocessing only on selected CpGs
        preprocessor = preprocessor_pipeline(cpg=feature, metadata=[])

        X_train_processed = preprocessor.fit_transform(train_data[feature])
        X_val_processed = preprocessor.transform(val_data[feature])

        y_train = train_data["age"].values
        y_val = val_data["age"].values

        # model of use BR
        model, y_val_predict = bayesianridge_model(X_train_processed,y_train,X_val_processed)

        rmse = mean_squared_error(y_val, y_val_predict) ** 0.5
        r2 = r2_score(y_val, y_val_predict)
        #######
        results.append({
            "Method": method,
            "Number of Features": len(feature),
            "Chosen K": b_k if method == "mRMR" else "-",
            "Validation RMSE": rmse,
            "Validation R^2": r2
        })

        print(f"\n  {method} ")
        print(f"Number of selected features : {len(feature)}")
        if method == "mRMR":
         print(f"Chosen K  : {b_k}")
        print(f"Validation RMSE  : {rmse:.4f}")
        print(f"Validation R^2 : {r2:.4f}")

    comparison_df = pd.DataFrame(results)

    # selecting the best method based on RMSE (lower one)
    best = comparison_df.loc[comparison_df["Validation RMSE"].idxmin()]
    selected_method = best["Method"]
    if selected_method == "Stability Selection":
        selected_features = stable_cpg
    else:
        selected_features = mrmr_cpg

    print("\n" + "=" * 60)
    print("Feature selection summary-comparison")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    print(f"Selected method (later use): {selected_method}")
    print(f"Number of features kept : {len(selected_features)}")
    print(f"Best validation RMSE : {best['Validation RMSE']:.4f}")
    print("=" * 60)

    return comparison_df, selected_features, selected_method

#######################################################################################
#######################################################################################
###########TASK 4###########
# 4.1  Hyperparameter tuning
def model_tuning(dev_data,selected_features,trails=40,seed=42):
    print("\nHyperparameter tuning on full dev_data")
    print(len(selected_features))

    X_dev = dev_data[selected_features]
    y_dev = dev_data["age"].values
    #Randomize search - 5-fold cross-val
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    #For each model pipeline
    #EN
    elastic_pip = Pipeline([("preprocessor", preprocessor_pipeline(cpg=selected_features, metadata=[])),("model", ElasticNet())])
    elastic_search = RandomizedSearchCV(
        estimator=elastic_pip,
        param_distributions={
            "model__alpha": loguniform(0.001, 10),
            "model__l1_ratio": uniform(0.1, 0.9)
        },
        n_iter=trails,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=-1
    )

    #SVR
    svr_pip = Pipeline([("preprocessor", preprocessor_pipeline(cpg=selected_features, metadata=[])),("model", SVR())])
    svr_search = RandomizedSearchCV(
        estimator=svr_pip,
        param_distributions={
            "model__C": loguniform(0.1, 500),
            "model__epsilon": [0.01, 0.1, 0.5, 1.0],
            "model__kernel": ["rbf", "linear"]
        },
        n_iter=trails,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=-1
    )

    #BR
    bayes_pip = Pipeline([("preprocessor", preprocessor_pipeline(cpg=selected_features, metadata=[])),("model", BayesianRidge())])
    bayes_search = RandomizedSearchCV(
        estimator=bayes_pip,
        param_distributions={
            "model__alpha_1": loguniform(1e-7, 1e-3),
            "model__alpha_2": loguniform(1e-7, 1e-3),
            "model__lambda_1": loguniform(1e-7, 1e-3),
            "model__lambda_2": loguniform(1e-7, 1e-3)
        },
        n_iter=trails,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=-1
    )

    # Fit searches
    print("____Tuning____")
    print("ElasticNet")
    elastic_search.fit(X_dev, y_dev)
    #print("Best ElasticNet params:", elastic_search.best_params_)
    #print("Best ElasticNet CV RMSE:", -elastic_search.best_score_)
    print("SVR")
    svr_search.fit(X_dev, y_dev)
    #print("Best SVR params:", svr_search.best_params_)
    #print("Best SVR CV RMSE:", -svr_search.best_score_)
    print("BayesianRidge")
    bayes_search.fit(X_dev, y_dev)
    #print("Best BayesianRidge params:", bayes_search.best_params_)
    #print("Best BayesianRidge CV RMSE:", -bayes_search.best_score_)

    best_models = {
        "ElasticNet": elastic_search.best_estimator_,
        "SVR": svr_search.best_estimator_,
        "BayesianRidge": bayes_search.best_estimator_
    }

    # dataset of best scores
    tuning_results = pd.DataFrame([
        {
            "Model": "ElasticNet",
            "Best RMSE": -elastic_search.best_score_,
            "Best Params": elastic_search.best_params_
        },
        {
            "Model": "SVR",
            "Best RMSE": -svr_search.best_score_,
            "Best Params": svr_search.best_params_
        },
        {
            "Model": "BayesianRidge",
            "Best RMSE": -bayes_search.best_score_,
            "Best Params": bayes_search.best_params_
        }
    ])

    return best_models, tuning_results


#######################################################################################
# 4.2  Final model evaluation
#Evaluation
def evaluation(model, eval_data, best_features, bootstrap=1000, seed=42):
    print("\nEvaluating with evaluation data")
    X_eval = eval_data[best_features]
    y_eval = eval_data["age"].values
    y_eval_predict = model.predict(X_eval)
    rng = np.random.RandomState(seed)
    indices = np.arange(len(y_eval))
    ##########
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    pearson_scores = []
    for _ in range(bootstrap):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        y_sample = y_eval[sample_idx]
        y_pred_sample = y_eval_predict[sample_idx]
    ####################################
        rmse = mean_squared_error(y_sample, y_pred_sample) ** 0.5
        mae = mean_absolute_error(y_sample, y_pred_sample)
        r2 = r2_score(y_sample, y_pred_sample)
        r, _ = pearsonr(y_sample, y_pred_sample)
        if np.std(y_sample) == 0 or np.std(y_pred_sample) ==0:
         r = np.na
        else:
         r, _ =pearsonr(y_sample, y_pred_sample)
    ####################################
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        pearson_scores.append(r)
    rmse_scores = np.array(rmse_scores)
    mae_scores = np.array(mae_scores)
    r2_scores = np.array(r2_scores)
    pearson_scores = np.array(pearson_scores)

    ####################################
    results = {
        "RMSE_mean": np.nanmean(rmse_scores),
        "RMSE_std": np.nanstd(rmse_scores, ddof=1),
        "RMSE_CI": np.nanpercentile(rmse_scores, [2.5, 97.5]),
        "rmse_scores": rmse_scores,

        "MAE_mean": np.nanmean(mae_scores),
        "MAE_std": np.nanstd(mae_scores, ddof=1),
        "MAE_CI": np.nanpercentile(mae_scores, [2.5, 97.5]),
        "mae_scores": mae_scores,

        "R2_mean": np.nanmean(r2_scores),
        "R2_std": np.nanstd(r2_scores, ddof=1),
        "R2_CI": np.nanpercentile(r2_scores, [2.5, 97.5]),
        "r2_scores": r2_scores,

        "Pearson_mean": np.nanmean(pearson_scores),
        "Pearson_std": np.nanstd(pearson_scores, ddof=1),
        "Pearson_CI": np.nanpercentile(pearson_scores, [2.5, 97.5]),
        "pearson_scores": pearson_scores
    }

    print("\nEvaluation of all three tuned models on the evaluation set using bootstrap resampling:")
    print(f"RMSE: mean={results['RMSE_mean']:.4f}, std={results['RMSE_std']:.4f}, 95% CI=({results['RMSE_CI'][0]:.4f}, {results['RMSE_CI'][1]:.4f})")
    print(f"MAE: mean={results['MAE_mean']:.4f}, std={results['MAE_std']:.4f}, 95% CI=({results['MAE_CI'][0]:.4f}, {results['MAE_CI'][1]:.4f})")
    print(f"R²: mean={results['R2_mean']:.4f}, std={results['R2_std']:.4f}, 95% CI=({results['R2_CI'][0]:.4f}, {results['R2_CI'][1]:.4f})")
    print(f"Pearson r : mean={results['Pearson_mean']:.4f}, std={results['Pearson_std']:.4f}, 95% CI=({results['Pearson_CI'][0]:.4f}, {results['Pearson_CI'][1]:.4f})")

    return results

#######################################################################################
# Table asked
def query_table(model_name, stage_results):
    rows = []

    for stage, res in stage_results.items():
        # taking keys from all tasks
        rmse_mean = res["RMSE_mean"] if "RMSE_mean" in res else res["rmse_mean"]
        rmse_ci = res["RMSE_CI"] if "RMSE_CI" in res else res["rmse_ci"]

        mae_mean = res["MAE_mean"] if "MAE_mean" in res else res["mae_mean"]

        r2_mean = res["R2_mean"] if "R2_mean" in res else res["r2_mean"]

        pearson_mean = res["Pearson_mean"] if "Pearson_mean" in res else res["pearson_mean"]

        rows.append({
            "Model": model_name,
            "Stage": stage,
            "RMSE (mean)": f"{rmse_mean:.3f}",
            "95% CI": f"[{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]",
            "MAE": f"{mae_mean:.3f}",
            "R²": f"{r2_mean:.3f}",
            "Pearson r": f"{pearson_mean:.3f}"
        })

    df = pd.DataFrame(rows)
    return df


#######################################################################################
#boxplots
def bootstrap_boxplots_4_2(results_dict, path="../figures/final_bootstrap_boxplots.png"):
    model_names = list(results_dict.keys())
    rmse_data = [results_dict[m]["rmse_scores"] for m in model_names]
    mae_data = [results_dict[m]["mae_scores"] for m in model_names]
    r2_data = [results_dict[m]["r2_scores"] for m in model_names]
    pearson_data = [results_dict[m]["pearson_scores"] for m in model_names]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    #Boxplot info (RMSE,MAE,R^2,Pearson r)
    axes[0, 0].boxplot(rmse_data)
    axes[0, 0].set_title("Bootstrap RMSE Distribution")
    axes[0, 0].set_xticks(range(1, len(model_names) + 1))
    axes[0, 0].set_xticklabels(model_names, rotation=20)
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 1].boxplot(mae_data)
    axes[0, 1].set_title("Bootstrap MAE Distribution")
    axes[0, 1].set_xticks(range(1, len(model_names) + 1))
    axes[0, 1].set_xticklabels(model_names, rotation=20)
    axes[0, 1].set_ylabel("MAE")
    axes[1, 0].boxplot(r2_data)
    axes[1, 0].set_title("Bootstrap R^2 Distribution")
    axes[1, 0].set_xticks(range(1, len(model_names) + 1))
    axes[1, 0].set_xticklabels(model_names, rotation=20)
    axes[1, 0].set_ylabel("R^2")
    axes[1, 1].boxplot(pearson_data)
    axes[1, 1].set_title("Bootstrap Pearson r Distribution")
    axes[1, 1].set_xticks(range(1, len(model_names) + 1))
    axes[1, 1].set_xticklabels(model_names, rotation=20)
    axes[1, 1].set_ylabel("Pearson r")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

#######################################################################################
def scatter_plot_realvspredicted(models_dict, eval_data, best_features, path="../figures/realvspredicted_scatter_plot.png"):
    X_eval = eval_data[best_features]
    y_eval = eval_data["age"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, (model_name, model) in zip(axes, models_dict.items()):
        y_predict = model.predict(X_eval)

        axis.scatter(y_eval, y_predict, edgecolor="black",alpha=0.7)
        min_val = min(y_eval.min(), y_predict.min())
        max_val = max(y_eval.max(), y_predict.max())
        axis.plot([min_val, max_val],[min_val, max_val],color="yellow",linestyle="--",linewidth=2,label="Ideal fit")
        axis.set_title(model_name)
        axis.set_xlabel("Actual Age")
        axis.set_ylabel("Predicted Age")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

#######################################################################################
def fs_eval_models(train_data, val_data, selected_features):
    X_tr = train_data[selected_features]
    X_v = val_data[selected_features]
    y_train = train_data["age"].values
    y_val = val_data["age"].values
    #################
    preprocessor = preprocessor_pipeline(cpg=selected_features, metadata=[])
    X_train_processed = preprocessor.fit_transform(X_tr)
    X_val_processed = preprocessor.transform(X_v)

   #Models
    elastic_model, elastic_predict = model_elasticnet(X_train_processed, y_train, X_val_processed)
    svr_rbf_k_model, svr_predict = svr_model(X_train_processed, y_train, X_val_processed)
    bayes_model, bayes_predict = bayesianridge_model(X_train_processed, y_train, X_val_processed)

    elastic_fs = evaluation_of_model(y_val, elastic_predict)
    svr_fs = evaluation_of_model(y_val, svr_predict)
    bayes_fs = evaluation_of_model(y_val, bayes_predict)

    return elastic_fs, svr_fs, bayes_fs

#######################################################################################
#4.3  Model selection and final model
def selection_save_final(tuned_results,tuned_models,model_dir="../models", model_filename="best_model.pkl"):
    rmse_summary = {
        model_name: result_dict["RMSE_mean"]
        for model_name, result_dict in tuned_results.items()
    }
    selected_name = min(rmse_summary, key=rmse_summary.get)
    selected_model = tuned_models[selected_name]

    print("\nFinal model selection")
    print("-" * 40)
    for model_name, rmse_value in rmse_summary.items():
        print(f"{model_name:<15} RMSE(mean): {rmse_value:.4f}")
    print("-" * 40)
    print(f"Selected best model: {selected_name}")
    print(f"Best RMSE(mean): {rmse_summary[selected_name]:.4f}")

    #######################
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_filename)

    with open(path, "wb") as f:
        pickle.dump(selected_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model saved")
    with open(path, "rb") as f:
        reloaded_model = pickle.load(f)
    print("Model loaded")


    return selected_name, reloaded_model, path

#######################################################################################
def plot_best_model_real_predict(y_true, y_predict, model_name, dir="../figures"):
    y_true = np.asarray(y_true)
    y_predict = np.asarray(y_predict)

    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"predicted_vs_actual_{model_name}.png")

    min_val = min(y_true.min(), y_predict.min())
    max_val = max(y_true.max(), y_predict.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_predict, alpha=0.7, edgecolor="black")
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Ideal fit")
    plt.xlabel("Real Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Predicted vs Real Age (in years) - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()

#######################################################################################
###########BONUS A###########
#Optuna Hyperparameter Optimization
#Optuna Tune Model
def optuna_tune_model(model_name, X_train, y_train, selected_features, n_trials=40, cv=5, seed=42):

    X_train = X_train[selected_features].copy()
    y_train = np.asarray(y_train)

    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)

    def objective(trial):
        if model_name == "ElasticNet":
            alpha = trial.suggest_float("alpha", 0.001, 10, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed)

        elif model_name == "SVR":
            C = trial.suggest_float("C", 0.1, 500, log=True)
            epsilon = trial.suggest_categorical("epsilon", [0.01, 0.1, 0.5, 1.0])
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])

            model = SVR(C=C, epsilon=epsilon, kernel=kernel)

        elif model_name == "BayesianRidge":
            alpha_1 = trial.suggest_float("alpha_1", 1e-7, 1e-3, log=True)
            alpha_2 = trial.suggest_float("alpha_2", 1e-7, 1e-3, log=True)
            lambda_1 = trial.suggest_float("lambda_1", 1e-7, 1e-3, log=True)
            lambda_2 = trial.suggest_float("lambda_2", 1e-7, 1e-3, log=True)

            model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,lambda_2=lambda_2)

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor_pipeline(cpg=selected_features, metadata=[])),
            ("model", model)
        ])

        scores = []
        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_tr = y_train[train_idx]
            y_va = y_train[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_va)
            rmse = mean_squared_error(y_va, y_pred) ** 0.5
            scores.append(rmse)

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params

    if model_name == "ElasticNet":
        best_model = ElasticNet(
            alpha=best_params["alpha"],
            l1_ratio=best_params["l1_ratio"],
            random_state=seed
        )

    elif model_name == "SVR":
        best_model = SVR(
            C=best_params["C"],
            epsilon=best_params["epsilon"],
            kernel=best_params["kernel"]
        )

    elif model_name == "BayesianRidge":
        best_model = BayesianRidge(
            alpha_1=best_params["alpha_1"],
            alpha_2=best_params["alpha_2"],
            lambda_1=best_params["lambda_1"],
            lambda_2=best_params["lambda_2"]
        )

    best_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline(cpg=selected_features, metadata=[])),
        ("model", best_model)
    ])

    best_pipeline.fit(X_train, y_train)

    return best_pipeline, study

#######################################################################################
#Tuning and for the 3 models
def optuna_tune_all_models(dev_data, selected_features, n_trials=40, cv=5, seed=42):
    X_dev = dev_data[selected_features].copy()
    y_dev = dev_data["age"].values

    best_models_optuna = {}
    studies = {}
    rows = []

    for model_name in ["ElasticNet", "SVR", "BayesianRidge"]:
        print(f"\nOptuna tuning: {model_name}")
        best_pipeline, study = optuna_tune_model(
            model_name=model_name,
            X_train=X_dev,
            y_train=y_dev,
            selected_features=selected_features,
            n_trials=n_trials,
            cv=cv,
            seed=seed
        )

        best_models_optuna[model_name] = best_pipeline
        studies[model_name] = study

        rows.append({
            "Model": model_name,
            "Best RMSE": study.best_value,
            "Best Params": study.best_params,
            "Trials": n_trials
        })

    optuna_results_df = pd.DataFrame(rows)
    return best_models_optuna, optuna_results_df, studies

#######################################################################################
#Head to head function of RandomSearch vs Optuna
def compare_randomsearch_vs_optuna(random_results_df, optuna_results_df):
    random_map = dict(zip(random_results_df["Model"], random_results_df["Best RMSE"]))
    optuna_map = dict(zip(optuna_results_df["Model"], optuna_results_df["Best RMSE"]))

    rows = []
    for model_name in ["ElasticNet", "SVR", "BayesianRidge"]:
        random_rmse = float(random_map[model_name])
        optuna_rmse = float(optuna_map[model_name])
        delta = optuna_rmse - random_rmse

        if optuna_rmse < random_rmse:
            winner = "Optuna"
        elif random_rmse < optuna_rmse:
            winner = "RandomSearch"
        else:
            winner = "Tie"

        rows.append({
            "Model": model_name,
            "RandomSearch RMSE": round(random_rmse, 4),
            "Optuna RMSE": round(optuna_rmse, 4),
            "Delta (Optuna - Random)": round(delta, 4),
            "Winner": winner
        })

    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))
    return comparison_df

#######################################################################################
# Function for the figure that shows Optuna optimisation history (trial RMSE vs trial number) for one model
def plot_optuna_history(study, model_name, path="../figures/optuna_history_elastincnet.png"):
    os.makedirs("../figures", exist_ok=True)

    trial_numbers = []
    trial_values = []

    for trial in study.trials:
        if trial.value is not None:
            trial_numbers.append(trial.number)
            trial_values.append(trial.value)

    plt.figure(figsize=(8, 5))
    plt.axhline(study.best_value, color='yellow', linestyle='--', label=f'Best RMSE = {study.best_value:.4f}')
    plt.plot(trial_numbers, trial_values, marker="o")
    plt.xlabel("Trial Number")
    plt.ylabel("CV RMSE")
    plt.title(f"Optuna Optimization History - {model_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

#######################################################################################
#Bonus B: Sex Prediction from DNA Methylation

def sex_label(train_data, val_data, eval_data):
    train_data = train_data.copy()
    val_data = val_data.copy()
    eval_data = eval_data.copy()

    train_data["sex_label"] = train_data["sex"].map({"F": 0, "M": 1})
    val_data["sex_label"] = val_data["sex"].map({"F": 0, "M": 1})
    eval_data["sex_label"] = eval_data["sex"].map({"F": 0, "M": 1})

    print("Sex labels created: F=0, M=1")
    return train_data, val_data, eval_data

#######################################################################################
def sex_features(train_data, k):
    k = int(k)
    cpg_cols = [i for i in train_data.columns if i.startswith("cg")]

    imputer = SimpleImputer(strategy="median")
    X_train_cpg = pd.DataFrame(
        imputer.fit_transform(train_data[cpg_cols]),
        columns=cpg_cols,
        index=train_data.index
    )

    y_train = train_data["sex_label"]

    print(f"Running mRMR classification --> K={k}")
    sex_features = mrmr_classif(X=X_train_cpg, y=y_train, K=k)
    print(f"Selected {len(sex_features)} sex-related CpG features")

    return sex_features

#######################################################################################
def plot_sexvsage_overlap(age_features, sex_features, path="../figures/sex_age_overlap.png"):
    os.makedirs("../figures", exist_ok=True)

    age_set = set(age_features)
    sex_set = set(sex_features)

    overlap = age_set & sex_set
    only_age = age_set - sex_set
    only_sex = sex_set - age_set

    print(f"Age: {len(only_age)}")
    print(f"Overlap: {len(overlap)}")
    print(f"Sex: {len(only_sex)}")

    plt.figure(figsize=(6, 5))
    plt.bar(["Age", "Overlap", "Sex"],[len(only_age), len(overlap), len(only_sex)],edgecolor="black")
    plt.title("Overlap between age-selected & sex-selected CpGs")
    plt.ylabel("Number of CpGs")
    plt.xlabel("Feature group")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

#######################################################################################

def prepare_sex_classification_data(train_data, eval_data, sex_features):
    X_train = train_data[sex_features].copy()
    X_eval = eval_data[sex_features].copy()

    y_train = train_data["sex_label"].values
    y_eval = eval_data["sex_label"].values

    preprocessor = preprocessor_pipeline(cpg=sex_features, metadata=[])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_eval_processed = preprocessor.transform(X_eval)

    return X_train_processed, X_eval_processed, y_train, y_eval, preprocessor
#######################################################################################
def training_classifiers(X_train, y_train):
    print("Training Logistic Regression")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    print("Training Gaussian Naive Bayes")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    print("Classification models --> Trained ")
    return lr, gnb
#######################################################################################
def evaluate_classifier(model, X_eval, y_eval, bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)

    acc_scores = []
    f1_scores = []
    mcc_scores = []
    roc_scores = []
    pr_scores = []

    indices = np.arange(len(y_eval))

    for _ in range(bootstrap):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        X_sample = X_eval[sample_idx]
        y_sample = y_eval[sample_idx]

        y_pred = model.predict(X_sample)
        y_prob = model.predict_proba(X_sample)[:, 1]

        acc_scores.append(accuracy_score(y_sample, y_pred))
        f1_scores.append(f1_score(y_sample, y_pred, zero_division=0))
        mcc_scores.append(matthews_corrcoef(y_sample, y_pred))

        if len(np.unique(y_sample)) < 2:
            roc_scores.append(np.nan)
            pr_scores.append(np.nan)
        else:
            roc_scores.append(roc_auc_score(y_sample, y_prob))
            pr_scores.append(average_precision_score(y_sample, y_prob))

    acc_scores = np.array(acc_scores)
    f1_scores = np.array(f1_scores)
    mcc_scores = np.array(mcc_scores)
    roc_scores = np.array(roc_scores)
    pr_scores = np.array(pr_scores)

    y_pred_full = model.predict(X_eval)
    y_prob_full = model.predict_proba(X_eval)[:, 1]

    results = {
        "accuracy_mean": np.nanmean(acc_scores),
        "accuracy_std": np.nanstd(acc_scores, ddof=1),
        "accuracy_ci": np.nanpercentile(acc_scores, [2.5, 97.5]),
        "accuracy_full": accuracy_score(y_eval, y_pred_full),
        "accuracy_scores": acc_scores,

        "f1_mean": np.nanmean(f1_scores),
        "f1_std": np.nanstd(f1_scores, ddof=1),
        "f1_ci": np.nanpercentile(f1_scores, [2.5, 97.5]),
        "f1_full": f1_score(y_eval, y_pred_full, zero_division=0),
        "f1_scores": f1_scores,

        "mcc_mean": np.nanmean(mcc_scores),
        "mcc_std": np.nanstd(mcc_scores, ddof=1),
        "mcc_ci": np.nanpercentile(mcc_scores, [2.5, 97.5]),
        "mcc_full": matthews_corrcoef(y_eval, y_pred_full),
        "mcc_scores": mcc_scores,

        "roc_auc_mean": np.nanmean(roc_scores),
        "roc_auc_std": np.nanstd(roc_scores, ddof=1),
        "roc_auc_ci": np.nanpercentile(roc_scores, [2.5, 97.5]),
        "roc_auc_full": roc_auc_score(y_eval, y_prob_full),
        "roc_auc_scores": roc_scores,

        "pr_auc_mean": np.nanmean(pr_scores),
        "pr_auc_std": np.nanstd(pr_scores, ddof=1),
        "pr_auc_ci": np.nanpercentile(pr_scores, [2.5, 97.5]),
        "pr_auc_full": average_precision_score(y_eval, y_prob_full),
        "pr_auc_scores": pr_scores
    }

    print("Completed")
    print(f"Accuracy : {results['accuracy_full']:.4f} (95% CI: {results['accuracy_ci'][0]:.4f} - {results['accuracy_ci'][1]:.4f})")
    print(f"F1       : {results['f1_full']:.4f} (95% CI: {results['f1_ci'][0]:.4f} - {results['f1_ci'][1]:.4f})")
    print(f"MCC      : {results['mcc_full']:.4f} (95% CI: {results['mcc_ci'][0]:.4f} - {results['mcc_ci'][1]:.4f})")
    print(f"ROC-AUC  : {results['roc_auc_full']:.4f} (95% CI: {results['roc_auc_ci'][0]:.4f} - {results['roc_auc_ci'][1]:.4f})")
    print(f"PR-AUC   : {results['pr_auc_full']:.4f} (95% CI: {results['pr_auc_ci'][0]:.4f} - {results['pr_auc_ci'][1]:.4f})")

    return results
#######################################################################################
def sex_classification_table(results_dict, features_label="mRMR-selected CpGs"):
    rows = []

    for model_name, res in results_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy (95% CI)": f"{res['accuracy_mean']:.3f} [{res['accuracy_ci'][0]:.3f}, {res['accuracy_ci'][1]:.3f}]",
            "F1 (95% CI)": f"{res['f1_mean']:.3f} [{res['f1_ci'][0]:.3f}, {res['f1_ci'][1]:.3f}]",
            "MCC (95% CI)": f"{res['mcc_mean']:.3f} [{res['mcc_ci'][0]:.3f}, {res['mcc_ci'][1]:.3f}]",
            "ROC-AUC (95% CI)": f"{res['roc_auc_mean']:.3f} [{res['roc_auc_ci'][0]:.3f}, {res['roc_auc_ci'][1]:.3f}]",
            "PR-AUC (95% CI)": f"{res['pr_auc_mean']:.3f} [{res['pr_auc_ci'][0]:.3f}, {res['pr_auc_ci'][1]:.3f}]",
            "Feature": features_label
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
#######################################################################################
def plot_confusion_matrices(lr, gnb, X_eval, y_eval, path="../figures/confusion_matrices_sex.png"):
    os.makedirs("../figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model, name in zip(axes,[lr, gnb],["Logistic Regression", "Gaussian NB"]):
        y_pred = model.predict(X_eval)
        matrix = confusion_matrix(y_eval, y_pred, labels=[0, 1])

        ax.imshow(matrix, cmap="Reds")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred F", "Pred M"])
        ax.set_yticklabels(["Actual F", "Actual M"])
        ax.set_title(f"Confusion Matrix - {name}")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, matrix[i, j],color="black", ha="center", va="center", fontsize=20,fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
#######################################################################################
def plot_roc_curves(lr, gnb, X_eval, y_eval, path="../figures/roc_curves_sex.png"):
    os.makedirs("../figures", exist_ok=True)

    plt.figure(figsize=(7, 6))

    models = [lr, gnb]
    names = ["Logistic Regression", "Gaussian NB"]

    for model, name in zip(models, names):
        y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_eval, y_prob)
        auc = roc_auc_score(y_eval, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1],color='red', linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Sex Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    #######################################################################################
def plot_top20_sex_cpgs(train_data, path="../figures/top20_sex_cpgs.png"):
        os.makedirs("../figures", exist_ok=True)

        cpg_cols = [i for i in train_data.columns if i.startswith("cg")]
        y = train_data["sex_label"].values

        print("Calculating |point-biserial r| for CpGs vs sex")
        correlations = {}

        for i in cpg_cols:
            x = train_data[i].fillna(train_data[i].median()).values
            r, _ = pearsonr(x, y)
            correlations[i] = abs(r)

        top20 = sorted(correlations, key=correlations.get, reverse=True)[:20]
        top20_vals = [correlations[i] for i in top20]
        print(top20[0], f"r={top20_vals[0]}")

        plt.figure(figsize=(10, 6))
        plt.barh(top20[::-1], top20_vals[::-1], edgecolor="black")
        plt.xlabel("|Point-biserial r| with sex")
        plt.title("Top 20 sex-discriminative CpGs")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
#######################################################################################
def classifier_boxplots(lr_scores, gnb_scores, path="../figures/classifier_boxplots_sex.png"):
    os.makedirs("../figures", exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    metrics = [
        ("accuracy_scores", "Accuracy", "Accuracy"),
        ("f1_scores", "F1 Score", "F1"),
        ("mcc_scores", "MCC", "MCC"),
        ("roc_auc_scores", "ROC-AUC", "ROC-AUC"),
        ("pr_auc_scores", "PR-AUC", "PR-AUC")
    ]

    flat_axes = axes.flatten()

    for ax, (key, title, ylabel) in zip(flat_axes, metrics):
        data = [lr_scores[key], gnb_scores[key]]
        labels = ["Logistic Regression", "Gaussian NB"]

        ax.boxplot(data, labels=labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    # Hide the unused last subplot
    flat_axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()