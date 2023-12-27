
# Basic Libraries
import pandas as pd
import numpy as np
import os

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm  # Progress bar
import warnings

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,  accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,  KFold, cross_validate, RandomizedSearchCV

# Machine Learning Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sklearn
from itertools import combinations  # For best subset selection
import math


sns.set_theme()  # Use seaborn's default theme
clrs = sns.color_palette()  # Get color palette

# Do not warn about chained assignment for Pandas
pd.options.mode.chained_assignment = None 

# Remove future warnings, caused by sklearn not using up to date pandas 
# functions for working with categorical data
warnings.simplefilter(action='ignore', category=FutureWarning)

# Have sklearn transformers output pandas dataframes instead of numpy arrays
sklearn.set_config(transform_output="pandas")

# Define a convenience function for cross validation.
def tune_param(model, X, y, display_name, folds=5, return_results=False, **kwargs):
    """
    Use GridSearchCV with k-fold cross validation to tune one parameter and 
    display a plot of the performance as a function of this parameter.
    
    Parameters
    ----------
    model : object
        The model to evaluate.
        
    X : DataFrame
        The training data to fit the model with.
        
    y : Series
        The responses to fit the model with.
        
    display_name : str
        The names of the parameters to use when displaying the plots.
        
    folds : int, Optional
        The number of folds to use. Defaults to 5.
        
    return_results : bool, Optional
        Whether to return the results of the GridSearchCV. Defaults to False.
        
    kwargs
        Each key is the parameter to tune, and the value is an iterable of the
        values to test. Up to one parameter can be specified this way.
    """
    
    param_grid = dict(kwargs.items())
    results = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=folds).fit(X, y)
    if len(param_grid) == 1:
        plot_cv_results(results, list(param_grid.keys())[0], display_name)
    else:
        print("Too many parameters to plot")
    if return_results:
        return results
    

def plot_cv_results(results, param_name, display_name):
    """
    Plot the results of a 1D GridSearchCV.
    
    Parameters
    ----------
    results : object
        The fit GridSearchCV object.
        
    param_name : str
        The full name of the parameter tuned.
        
    display_name : str
        The name of the parameter to display on the plot.
    """
    
    # Extract results
    param_name = "param_" + param_name
    params = results.cv_results_[param_name].data.astype(float)
    performance = -results.cv_results_[f"mean_test_score"]
    
    # Get best parameter
    best_idx = np.argmin(performance)
    best_param = params[best_idx]
    best_performance = performance[best_idx]
    
    plt.plot(params, performance)
    
    # Plot best model
    plt.axvline(x=best_param, color=clrs[3], linestyle="--")
    plt.axhline(y=best_performance, color=clrs[3], linestyle="--")
    
    # Label and adjust plot
    plt.xlabel(display_name)
    plt.ylabel(f"Cross-Validated RSS")
    if not isinstance(best_param, int):
        best_param = f"{best_param:.4f}"
    plt.title(f"RSS={best_performance:.4f} at {display_name}={best_param}")
    
def param_summary(model):
    """
    Display the nonzero coefficients of an estimator.
    
    Parameters
    ----------
    model : object
        The estimator which will be summarized.
    """
    
    out = pd.DataFrame(model.coef_, columns=["Coefficient"])
    out.index = model.feature_names_in_
    out = out[out["Coefficient"] != 0]
    display(out.T)

def best_subset(X, y):
    num_features_vals = list(range(1, len(X.columns)+1))  # Number of features being considered
    best_subsets = []  # List to store the best subset of features for each number of features
    best_models = []  # List to store the best models for each number of features
    best_r2_vals = []  # List to store the best r^2 values for each number of features
    best_train_rss_vals = []  # list o store the best training rss for each number of features

    for num_features in num_features_vals:  # Loop over number of features
        best_subset = None  # Variable to store the best subset of size num_features
        best_model = None  # Variable to store the corresponding best model
        best_r2 = -float("inf")  # Variable to store the corresponding best R^2
        best_rss = float("inf")  # Variable to store the corresponding best training RSS
        
        num_subsets = math.comb(len(X.columns), num_features)  # Calculate number of possible subsets
        pbar = tqdm(combinations(X.columns, num_features), total=num_subsets)  # Generate progress bar iterator
        for features in pbar:  # Loop over subsets of the features of size num_features
            X_subset = X[list(features)]  # Obtain the data corresponding to the features selected
            model = LinearRegression().fit(X_subset, y)  # Fit a linear model
            yhat = model.predict(X_subset)  # Get the predictions
            r2 = r2_score(y, yhat)  # Calculate the r2
            if r2 > best_r2:  # If this model is better than the best so far
                best_subset = features  # Update subset of features
                best_model = model  # Update best model
                best_r2 = r2  # Update best r2
                best_rss = mean_squared_error(y, yhat)  # Update best rss
                pbar.set_description(f"Best R2 with {num_features} Features: {best_r2:.4f}")  # Update progress bar
        best_subsets.append(best_subset)  # Store best subset for this number of features
        best_models.append(model)  # Store best model for this number of features
        best_r2_vals.append(best_r2)  # Store best r2 for this number of features
        best_train_rss_vals.append(best_rss)  # Store best rss for this number of features
    
    best_test_rss_vals = []  # Get test rss via cross-validation
    for model, subset in zip(best_models, best_subsets):
        cv = KFold(n_splits=10, shuffle=False)  # Define cross validation strategy
        # Perform cross validation
        cv_results = cross_validate(model, X[list(subset)], y, cv=cv, scoring="neg_mean_squared_error")
        best_test_rss_vals.append(-cv_results["test_score"].mean())  # Append mean RSS

    # Plot results
    fig, ax = plt.subplots()  # Generate figure and axis
    # Plot training and testing rss with respect to number of features
    sns.lineplot(x=num_features_vals, y=best_train_rss_vals, label="Train RSS", ax=ax)
    sns.lineplot(x=num_features_vals, y=best_test_rss_vals, label="Test RSS", ax=ax)

    # Calculate best number of features and best test RSS
    best_index = np.argmin(best_test_rss_vals)
    best_num_features = num_features_vals[best_index]
    best_rss = best_test_rss_vals[best_index]

    # Add lines for best number of features and best test RSS
    ax.axvline(x=best_num_features, color=clrs[3], linestyle="dashed")
    ax.axhline(y=best_rss, color=clrs[3], linestyle="dashed")
    ax.text(13, best_rss, f"{best_rss:.4f}")  # Add annotation for best test RSS

    # Label and display plot
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RSS")
    ax.set_title("Tuning Number of Features via Best Subset Selection")
    plt.show()

    return best_subset, best_model, best_r2, best_rss