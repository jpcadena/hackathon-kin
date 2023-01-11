"""
This is the function module.
It supports the main script for cleaning data y plotting.
"""
import itertools
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, \
    roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

# This constants could be imported from json file
RE_PATTERN: str = "([a-z])([A-Z])"
RE_REPL: str = r"\g<1> \g<2>"
PALETTE: str = 'pastel'
FIG_SIZE: tuple[int] = (15, 8)
colors: list[str] = ['lightskyblue', 'coral', 'palegreen']


def plot_count(df: pd.DataFrame, variables):
    """This method plots the counts of observations from the given variables"""
    plt.figure(figsize=FIG_SIZE)
    plt.suptitle('Countplot for Discrete variables')
    p: int = 1
    for i in variables:
        plt.subplot(1, 3, p)
        sns.countplot(x=df[i], hue=df.Exited, palette=PALETTE)
        label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=i)
        plt.xlabel(label, fontsize=15)
        plt.ylabel('Count', fontsize=15)
        p += 1
    plt.show()
    plt.savefig('discrete_variables.png')


def plot_distribution(df_column: pd.Series, color: str):
    """This method plots the distribution of the given
    quantitative continuous variable"""
    label = re.sub(pattern=RE_PATTERN, repl=RE_REPL,
                   string=str(df_column.name))
    sns.displot(x=df_column, kde=True, color=color, height=8, aspect=1.875)
    plt.title('Distribution Plot for ' + label)
    plt.xlabel(label, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.show()
    plt.savefig(str(df_column.name) + '.png')


def age_geo_scatter(df: pd.DataFrame, column: str):
    """This method plots the relationship between Age and the given column for
    the Geography subset"""
    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(x='Age', data=df, y=column, hue='Geography',
                    palette=PALETTE)
    label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=column)
    plt.title('Age Wise ' + label + ' Distribution')
    plt.show()
    print(df[['Age', column]].corr())
    plt.savefig('Age_' + column + '_Geography.png')


def boxplot_dist(df: pd.DataFrame, x: str, y: str):
    """This method plots the distribution of the x data with regards to the y
    data in a boxplot"""
    plt.figure(figsize=FIG_SIZE)
    x_label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=x)
    y_label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=y)
    sns.boxplot(x=x, y=y, data=df, palette=PALETTE)
    plt.title(x_label + ' with regards to ' + y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.show()
    plt.savefig(x + '_' + y + '.png')


def plot_confusion_matrix(cm, classes, name, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """This function plots the Confusion Matrix of the test and pred arrays"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(figsize=FIG_SIZE)
    plt.rcParams.update({'font.size': 16})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, color="blue")
    plt.yticks(tick_marks, classes, color="blue")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name + '_confusion_matrix.png')


def cleaning_df(df: pd.DataFrame, df_filter: pd.DataFrame) -> pd.DataFrame:
    """This function filters the given dataframes based on their CustomerId"""
    mask = df.CustomerId.isin(df_filter.CustomerId)
    df = df[mask]
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    return df


def lof_observation(df: pd.DataFrame) -> pd.DataFrame:
    """This function identifies outliers with LOF method"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num_cols = df.select_dtypes(include=numerics)
    df_outlier = df_num_cols.astype("float64")
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df: pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(stacked=True, xlim=[0, 20], color='r',
                   title='Visualization of outliers according to the LOF '
                         'method', style='.-')
    plt.show()
    plt.savefig('outliers.png')
    th_val = np.sort(df_scores)[2]
    outliers = df_scores > th_val
    df = df.drop(df_outlier[~outliers].index)
    print(df.shape)
    return df


def clear_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """This functions remove the outliers from Age and CreditScore variables"""
    q1: float = df["Age"].quantile(0.25)
    q3: float = df["Age"].quantile(0.75)
    iqr: float = q3 - q1
    lower: float = q1 - 1.5 * iqr
    upper: float = q3 + 1.5 * iqr
    print("Age - Lower score: ", lower, "and upper score: ", upper)
    df_outlier = df["Age"][(df["Age"] > upper)]
    df["Age"][df_outlier.index] = upper
    q1 = df["CreditScore"].quantile(0.25)
    q3 = df["CreditScore"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    print("CreditScore - Lower score: ", lower, "and upper score: ", upper)
    df_outlier = df["CreditScore"][(df["CreditScore"] < lower)]
    df["CreditScore"][df_outlier.index] = lower
    return df


def model_results(model, x, y, y2, x2):
    model.fit(x, y)
    model_name = type(model).__name__
    preds = model.predict(x2)
    print(model_name)
    print(classification_report(y2, preds))
    auc = roc_auc_score(y2, preds)
    print(auc)
    # CONFUSION MATRIX
    cfm = confusion_matrix(y2, y_pred=preds)
    confusion_matrix_title = str(model_name) + 'Churn Confusion matrix'
    name = str(model_name)
    plot_confusion_matrix(cfm, classes=['Non Churn', 'Churn'], name=name,
                          title=confusion_matrix_title, )
    tn, fp, fn, tp = cfm.ravel()
    print("True Negatives: ", tn)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("True Positives: ", tp)
    # ROC CURVE
    y_pred_proba = model.predict_proba(x2)
    skplt.metrics.plot_roc(y2, y_pred_proba, figsize=FIG_SIZE)
    plt.show()
    plt.savefig(str(model_name) + '_roc_curves.png')
