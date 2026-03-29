import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet,BayesianRidge
from sklearn.svm import SVR

###########TASK 1###########
# 1.2 Preprocessing

# Checking the missing values in the data
def check_missing_values(data):
    #Total missing values of data
    total_missing_values=data.isna().sum().sum()
    print("Total missing:",total_missing_values)
    #highlighting columns with most missing data
    miss_per_col= data.isna().sum()
    miss_per_col=miss_per_col[miss_per_col>0]
    sorted_miss= miss_per_col.sort_values(ascending=False)
    print("\nTop five missing:", sorted_miss.head(5))

# Feature Scaring & Categorical Features process setting
###############################
def feature_gather(data):
    metadata=[i for i in["sex","ethnicity"]if i in data.columns]
    cpg=[i for i in data.columns if i.startswith("cg")]
    age="age"
    return metadata,cpg,age

def feature_set(data,sets="all"):
    metadata = [i for i in ["sex", "ethnicity"] if i in data.columns]
    cpg = [i for i in data.columns if i.startswith("cg")]
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

#Building the processor that will be applied across all splits
#fit on training
#Application on val and later on eval
def  preprocessor_pipeline(cpg,metadata):
    transformers = []
    if len(cpg)>0:
        cpq_pipeline=Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("cpg",cpq_pipeline,cpg))

    if len(metadata)>0:
         metadata_pipeline=Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),
             ("encoder", OneHotEncoder(handle_unknown="ignore"))
         ])
         transformers.append(("metadata",metadata_pipeline,metadata))

    preprocessor=ColumnTransformer(transformers=transformers)
    return preprocessor
###############################
#1.3 Exploratory analysis
# Creating the dataset summary function
def data_summary(data,name):
    summary={"dataset":name,
             "samples": len(data),
             "age_mean":round(data["age"].mean(),2),
             "age_std":round(data["age"].std(),2),
             "age_min":data["age"].min(),
             "age_max":data["age"].max(),
             "sex_balance": data["sex"].value_counts().to_dict() if "sex" in data.columns else {},
             "ethnicity_balance":data["ethnicity"].value_counts().to_dict() if "ethnicity" in data.columns else {},
             "cpg_info": len([i for i in data.columns if i.startswith("cg")]) ,
             "total_missing_values": int(data.isna().sum().sum())
             }
    return summary

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

# Total statistics table
def stats_table(train, val, evaluation):
    rows = []
    for name, data in [("Train", train), ("Validation", val), ("Evaluation", evaluation)]:
        rows.append({
            "split": name,
            "n_samples": len(data),
            "age_mean ± std": f"{data['age'].mean():.1f} ± {data['age'].std():.1f}",
            "age_range": f"{data['age'].min():.0f} - {data['age'].max():.0f}",
            "Male": int((data["sex"] == "M").sum()) if "sex" in data.columns else 0,
            "Female": int((data["sex"] == "F").sum()) if "sex" in data.columns else 0
        })

    stats_tab = pd.DataFrame(rows)
    return stats_tab

#Checking distribution per split
def age_split_plot(train, val, evaluation):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, data) in zip(axes, [("Train", train), ("Validation", val), ("Evaluation", evaluation)]):
        ax.hist(data["age"], bins=20, color="blue", edgecolor="black")
        ax.set_title(name)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("../figures/age_histogram_split_data.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

###########TASK 2###########
# 2.1  OLS LinearRegression across feature sets

#OlS Linear Regression for the train_data

def ols_model(X_train, y_train,X_val):
    model=LinearRegression()
    model.fit(X_train, y_train)
    y_predict=model.predict(X_val)
    return model,y_predict

def evaluation_of_model (y_true,y_predict,n_bootstrap=100,seed=42):
    print("Evaluation of the model by using bootstrap")
    rng = np.random.RandomState(seed)
    y_true=np.array(y_true)
    y_predict=np.array(y_predict)

#RMSE, MAE, R², and Pearson r
    rmse_scores=[]
    mae_scores=[]
    r2_scores=[]
    pearson_corr_scores=[]
    indices = np.arange(len(y_true))

    for _ in range(n_bootstrap):
        sample_idx=rng.choice(indices, size=len(indices), replace=True)
        y_true_sample=y_true[sample_idx]
        y_pred_sample=y_predict[sample_idx]
        rmse=np.sqrt(mean_squared_error(y_true_sample, y_pred_sample))
        mae=mean_absolute_error(y_true_sample, y_pred_sample)
        r2=r2_score(y_true_sample, y_pred_sample)
        r, _=pearsonr(y_true_sample, y_pred_sample)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        pearson_corr_scores.append(r)

    rmse_scores=np.array(rmse_scores)
    mae_scores=np.array(mae_scores)
    r2_scores=np.array(r2_scores)
    pearson_corr_scores=np.array(pearson_corr_scores)

    full_rmse=np.sqrt(mean_squared_error(y_true, y_predict))
    full_mae=mean_absolute_error(y_true, y_predict)
    full_r2=r2_score(y_true, y_predict)
    full_r, _=pearsonr(y_true, y_predict)

#Addind and with 95% confidence intervals (CI).
    final_results= {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(ddof=1),
        "rmse_ci": np.percentile(rmse_scores, [2.5, 97.5]),
        "rmse_full": full_rmse,
        "rmse_scores": rmse_scores,

        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(ddof=1),
        "mae_ci": np.percentile(mae_scores, [2.5, 97.5]),
        "mae_full": full_mae,
        "mae_scores": mae_scores,

        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(ddof=1),
        "r2_ci": np.percentile(r2_scores, [2.5, 97.5]),
        "r2_full": full_r2,
        "r2_scores": r2_scores,

        "pearson_mean": pearson_corr_scores.mean(),
        "pearson_std": pearson_corr_scores.std(ddof=1),
        "pearson_ci": np.percentile(pearson_corr_scores, [2.5, 97.5]),
        "pearson_full": full_r,
        "pearson_scores":pearson_corr_scores
    }

    print("Validation metrics and 95% CI:")
    print(f"RMSE: {full_rmse:.4f} (95% CI: {final_results['rmse_ci'][0]:.4f} - {final_results['rmse_ci'][1]:.4f})")
    print(f"MAE: {full_mae:.4f} (95% CI: {final_results['mae_ci'][0]:.4f} - {final_results['mae_ci'][1]:.4f})")
    print(f"R^2: {full_r2:.4f} (95% CI: {final_results['r2_ci'][0]:.4f} - {final_results['r2_ci'][1]:.4f})")
    print(f"Pearson r: {full_r:.4f} (95% CI: {final_results['pearson_ci'][0]:.4f} - {final_results['pearson_ci'][1]:.4f})")
    return final_results

###########################################
#2.2  Three regression models at default hyperparameters

#MODELS with default parameters

#ElasticNet — L1+L2 regularised linear regression
def model_elasticnet(X_train,y_train,X_val):
    print("################Training of ElasticNet###############")
    model = ElasticNet()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_val)
    return model, y_predict

#SVR — Support Vector Regression with RBF kernel
def svr_model(X_train,y_train,X_val):
    print("###############Training of SVR###############")
    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)
    y_predict = model.predict(X_val)
    return model, y_predict

#BayesianRidge — Bayesian linear regression with automatic relevance determination
def bayesianridge_model(X_train, y_train, X_val):
    print("###############Training of BayesianRidge model###############")
    model = BayesianRidge()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_val)
    return model, y_predict

#Derivables
#report_table
def report_table(results):
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "RMSE (mean ± CI)": f"{res['rmse_mean']:.4f} ({res['rmse_ci'][0]:.4f} - {res['rmse_ci'][1]:.4f})",
            "MAE": f"{res['mae_mean']:.4f}",
            "R^2": f"{res['r2_mean']:.4f}",
            "Pearson r": f"{res['pearson_mean']:.4f}"
        })

    df = pd.DataFrame(rows)
    print("Report table of Model Performance:")
    print(df.to_string(index=False))
    return df


#Boxplots
def bootstrap_boxplots(results,path="../figures/bootstrap_boxplots_rmse_r2.png"):
    mname = list(results.keys())
    # Extract bootstrap distributions
    rmse_data = [results[m]["rmse_scores"] for m in mname]
    r2_data   = [results[m]["r2_scores"] for m in mname]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #For RMSE
    axes[0].boxplot(rmse_data, patch_artist=True)
    axes[0].set_title("Bootstrap RMSE Distribution", fontsize=12)
    axes[0].set_ylabel("RMSE in years")
    axes[0].set_xlabel("Model")
    axes[0].set_xticks(range(1, len(mname) + 1))
    axes[0].set_xticklabels(mname, rotation=20)

    # For R^2
    axes[1].boxplot(r2_data, patch_artist=True)
    axes[1].set_title("Bootstrap R^2 Distribution", fontsize=12)
    axes[1].set_ylabel("R^2")
    axes[1].set_xlabel("Model")
    axes[1].set_xticks(range(1, len(mname) + 1))
    axes[1].set_xticklabels(mname, rotation=20)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

#Checking for overfitting
def train_vs_val_all_models(models, X_train, y_train, X_val, y_val):
    results = []
    for name, model in models.items():
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        results.append({
            "Model": name,
            "Train RMSE": train_rmse,
            "Val RMSE": val_rmse,
            "Train R^2": train_r2,
            "Val R^2": val_r2
        })

    return pd.DataFrame(results)

###########TASK 3###########
#3.1  Stability Selection








