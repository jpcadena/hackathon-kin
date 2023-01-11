"""
This is the main script to run.
Finance and Risk Challenge for Kin Analytics.
"""
import datetime as dt
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from keras.layers import Dense
from keras.models import Sequential
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, \
    cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from functions import cleaning_df, plot_count, plot_distribution, \
    age_geo_scatter, boxplot_dist, lof_observation, clear_outliers, \
    model_results

start_time = time.time()
matplotlib.use('Qt5Agg')
plt.rcParams.update({'figure.max_open_warning': 0})
pd.set_option("display.max_columns", 20)
pd.set_option("max_colwidth", 1000)
pd.options.mode.chained_assignment = None

# This constants could be imported from json file
q: int = 1
MINIMUM_TENURE: int = 2
ddd: int = 365
SEED: int = 10101
MISSING_PERCENTAGE: float = 75.0
SCORING_METRIC: str = 'recall'
CURRENT_DATE: str = '2019-11-30'
MIN_DATE_CONTRACT: str = '2015-01-01'
CLIENTS_FILENAME: str = 'data/raw/clients_table.txt'
CREDITS_FILENAME: str = 'data/raw/credit_score_table.txt'
PRODUCTS_FILENAME: str = 'data/raw/products_table.txt'
TRANSACTIONS_FILENAME: str = 'data/raw/transactions_table.txt'
PALETTE: str = 'pastel'
FIG_SIZE: tuple[int] = (15, 8)
bins: list[int] = [18, 29, 40, 51, 62, 73, 84, 95]
colors: list[str] = ['lightskyblue', 'coral', 'palegreen']
bin_labels: list[str] = ['18-28', '29-39', '40-50', '51-61', '62-72', '73-83',
                         '84-94']

# reading databases
clients_dtypes: dict = {'CustomerId': int, 'Surname': str, 'Geography': str,
                        'Gender': str, 'HasCrCard': float,
                        'IsActiveMember': float,
                        'EstimatedSalary': float, 'application_date': str,
                        'exit_date': str, 'birth_date': str}
clients_parse_dates: list[str] = ['application_date', 'exit_date',
                                  'birth_date']
clients_df: pd.DataFrame = pd.read_csv(CLIENTS_FILENAME, sep=",", header=0,
                                       dtype=clients_dtypes,
                                       parse_dates=clients_parse_dates,
                                       infer_datetime_format=True)

# test for git
credit_dtypes: dict = {'CustomerId': int, 'Date': str, 'Score': int}
credit_score_df: pd.DataFrame = pd.read_csv(CREDITS_FILENAME, sep=",",
                                            header=0,
                                            dtype=credit_dtypes,
                                            parse_dates=['Date'],
                                            infer_datetime_format=True)
credit_score_df['Date'] = credit_score_df['Date'].dt.strftime("%Y-%m")

products_dtypes: dict = {'ContractId': str, 'CustomerId': int, 'Products': str}
products_df: pd.DataFrame = pd.read_csv(PRODUCTS_FILENAME, sep=",", header=0,
                                        dtype=products_dtypes)

transactions_dtypes: dict = {'CustomerId': int, 'Transaction': str,
                             'Value': float}
transactions_df: pd.DataFrame = pd.read_csv(TRANSACTIONS_FILENAME, sep=",",
                                            header=0,
                                            dtype=transactions_dtypes)

# initial number of records in the “Client” database
print(clients_df.head())
print(clients_df.shape)

# first filter
mask_contracts = (clients_df['application_date'] >= MIN_DATE_CONTRACT)
clients_df = clients_df.loc[mask_contracts]
print(clients_df.head())
print(clients_df.shape)  # number of records in the “Client” database

# second filter
clients_df = clients_df[clients_df["Geography"] != 'Italy']
print(clients_df.head())
print(clients_df.shape)  # number of records in the “Client” database

# third filter
min_count = int(((100 - MISSING_PERCENTAGE) / 100) * clients_df.shape[1] + 1)
clients_df.dropna(axis=0, thresh=min_count, inplace=True)
print(clients_df.head())
print(clients_df.shape)  # number of records in the “Client” database

# fourth filter
clients_df.drop_duplicates(subset=['CustomerId'], inplace=True)
print(clients_df.head())
print(clients_df.shape)  # number of records in the “Client” database

# fifth filter
clients_df['ActiveDays'] = (clients_df['exit_date'] -
                            clients_df['application_date']).dt.days
clients_df.ActiveDays.fillna(value=(
        dt.datetime.strptime(CURRENT_DATE, '%Y-%m-%d') -
        clients_df.application_date).dt.days, inplace=True)
clients_df['ActiveDays'] = clients_df['ActiveDays'].astype(int)
clients_df['Tenure'] = (clients_df['ActiveDays'] / ddd).astype(int)
mask_tenure = (clients_df['Tenure'] >= MINIMUM_TENURE)
clients_df = clients_df.loc[mask_tenure]
clients_df.reset_index(inplace=True)
clients_df.drop('index', axis=1, inplace=True)
clients_df['Exited'] = np.where(clients_df.exit_date.isna(), 0, 1)
clients_df['Exited'] = clients_df['Exited'].astype(int)
clients_df['IsActiveMember'] = clients_df['IsActiveMember'].astype(int)
clients_df['HasCrCard'] = clients_df['HasCrCard'].astype(int)
print(clients_df.head())
print(clients_df.shape)  # number of records in the “Client” database

# cleaning clients database
clients_df.reset_index(inplace=True)
clients_df.drop('index', axis=1, inplace=True)

# cleaning products database
products_df = cleaning_df(products_df, clients_df)

# cleaning transactions database
transactions_df = cleaning_df(transactions_df, clients_df)

# cleaning credit database
credit_score_df = cleaning_df(credit_score_df, clients_df)

# products data set
products_df_gb: pd.DataFrame = products_df.groupby('CustomerId')[
    ['CustomerId', 'Products']].size().reset_index(name='ProductsCounts')
mask_customer_products_gb = clients_df.CustomerId.isin(
    products_df_gb.CustomerId)
clients_df = clients_df[mask_customer_products_gb]
clients_df.set_index('CustomerId', inplace=True)
products_df_gb.set_index('CustomerId', inplace=True)
clients_df = clients_df.join(products_df_gb, how='inner')

# transactions data set
transactions_df_gb: pd.DataFrame = transactions_df.groupby(['CustomerId'])[
    ['Value']].sum()
clients_df = clients_df.join(transactions_df_gb, how='inner')
clients_df.rename(columns={'Value': 'Balance'}, inplace=True)

# credit bureau scores data set
credit_score_df.sort_values(by=['CustomerId', 'Date'], inplace=True)
credit_score_dict: dict = dict(tuple(credit_score_df.groupby('CustomerId')))
clients_df['application_date_monthly'] = clients_df[
    'application_date'].dt.strftime('%Y-%m')
clients_df['application_date_monthly'] = clients_df[
    'application_date_monthly'].astype(str)
clients_df.reset_index(inplace=True)
for key, value in credit_score_dict.items():
    value.reset_index(inplace=True)
    value.drop(columns={'CustomerId', 'index'}, axis=1, inplace=True)
    found_date = str(clients_df.loc[clients_df['CustomerId'] == key,
    'application_date_monthly'].iloc[0])
    found_score = int(value.loc[value['Date'] == found_date, 'Score'].iloc[0])
    found_index = int(clients_df.index[clients_df['CustomerId'] ==
                                       key].tolist()[0])
    clients_df.at[found_index, 'CreditScore'] = found_score

clients_df['Age'] = (clients_df['application_date'] -
                     clients_df['birth_date']).dt.days / ddd
clients_df.Age = clients_df.Age.astype(int)
clients_df.sort_values(by=['CustomerId'], inplace=True)
clients_df.drop(columns={'application_date', 'application_date_monthly',
                         'exit_date', 'ActiveDays', 'birth_date', 'Surname',
                         'CustomerId'},
                axis=1, inplace=True)
clients_df.sort_index(inplace=True)

# Exploratory Data Analysis
print(clients_df.head())
print(clients_df.shape)  # number of records in the “Client” database
print(clients_df.dtypes)
print(clients_df.info())
print(clients_df.describe(include='all', datetime_is_numeric=True))
missing_values = (clients_df.isnull().sum())
print(missing_values[missing_values > 0])
print(missing_values[missing_values > 0] / clients_df.shape[0] * 100)

# Identifying Class Imbalance in Exited (65-35%)
print(clients_df['Exited'].value_counts())
print(clients_df['Exited'].unique())
print(clients_df['Exited'].value_counts(normalize=True) * 100)

df: pd.DataFrame = clients_df.copy()
# Plots
labels = 'Churn', 'Not Churn'
fig, ax = plt.subplots(1, 2, figsize=FIG_SIZE)
plt.suptitle('Distribution of Customers Churn and Not Churn')
df['Exited'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%',
                                     labels=labels, ax=ax[0], shadow=True,
                                     colors=colors[:2])
ax[0].set_ylabel('')
sns.countplot(x='Exited', data=df, ax=ax[1], palette=PALETTE)
plt.show()
plt.savefig('customers_churn.png')

# Heatmap for df variables
fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
sns.heatmap(data=df.corr(), annot=True, cmap="RdYlGn")
plt.title('Heatmap showing correlations among columns', fontsize=15)
plt.show()
plt.savefig('correlations_heatmap.png')

# General distribution of  Discrete (Boolean) variables
discrete_variables = ['Tenure', 'IsActiveMember', 'HasCrCard']
plot_count(df, discrete_variables)

# Analyzing continuous quantitative behavior
plot_distribution(df.CreditScore, color=colors[0])
plot_distribution(df.EstimatedSalary, color=colors[1])
plot_distribution(df.Balance, color=colors[2])

# Analyzing relationship between variables
age_geo_scatter(df, 'CreditScore')
age_geo_scatter(df, 'Balance')
age_geo_scatter(df, 'EstimatedSalary')

boxplot_dist(df, 'Tenure', 'Geography')
boxplot_dist(df, 'Geography', 'ProductsCounts')

sns.catplot(x='Geography', hue='HasCrCard', data=df, kind="count",
            legend=False, palette=PALETTE, height=8, aspect=1.875)
plt.title('Customers frequency according to their filtered Geography',
          fontsize=15)
plt.legend(title='Has Credit Card', labels=['No', 'Yes'])
plt.show()
plt.savefig('geography_by_hascrcard.png')

plt.figure(figsize=FIG_SIZE)
sns.barplot(x='Age', y='EstimatedSalary', data=df, color=colors[0], ci=None)
plt.title('Estimated Salary with Age')
plt.show()
plt.savefig('salary_by_age.png')

countries: list = []
for n in df.Geography.unique():
    country_data = list(df[df.Geography == n]['Balance'])
    countries.append(country_data)
sns.set()
plt.figure(figsize=FIG_SIZE)
plt.hist(countries, bins=7, stacked=True, color=colors)
plt.xlabel('Balance')
plt.ylabel('Number of customers')
plt.legend(df.Geography.unique().tolist())
plt.title('Country Wise Balance Distribution')
plt.show()
plt.savefig('countries.png')

# Analyzing Age bins frequency distribution
dummy_age_labels = pd.cut(df['Age'], bins, labels=bin_labels, right=False)
df_copy: pd.DataFrame = df.copy()
df_copy['Age_labeled'] = dummy_age_labels
plt.figure(figsize=FIG_SIZE)
sns.barplot(x='Age_labeled', y='ProductsCounts', hue='Gender', data=df_copy,
            palette=PALETTE)
plt.title('Age bins distribution by Gender')
plt.show()
plt.savefig('age_by_gender.png')

# Plot a grid with count plots of all categorical variables
countplot_exited_data = ['Geography', 'Gender', 'Tenure', 'ProductsCounts',
                         'HasCrCard', 'IsActiveMember']
plt.figure(figsize=FIG_SIZE)
plt.suptitle('Discrete quantitative variables by Exited variable')
for j in countplot_exited_data:
    plt.subplot(2, 3, q)
    ax3 = sns.countplot(x=df[j], hue=df.Exited, palette=PALETTE)
    plt.xlabel(j)
    q += 1
plt.show()
plt.savefig('exited_discrete_variables.png')

# Plot a grid with count plots of all numeric variables
boxplot_exited_data = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
r: int = 1
plt.figure(figsize=FIG_SIZE)
plt.suptitle('Continuous quantitative variables by Exited variable')
for k in boxplot_exited_data:
    plt.subplot(2, 2, r)
    ax4 = sns.boxplot(y=df[k], x=df.Exited, hue=df.Exited, palette=PALETTE)
    plt.xlabel(k)
    r += 1
plt.show()
plt.savefig('exited_continuous_variables.png')

# Feature Engineering
df2 = pd.get_dummies(data=clients_df, columns=['Geography', 'Gender'])
cols_to_scale = ['Tenure', 'EstimatedSalary', 'Balance', 'CreditScore', 'Age',
                 'ProductsCounts']
scaler: MinMaxScaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
print(df2)

# Outliers Observe (LOF method and Suppress)
df2: pd.DataFrame = lof_observation(df2)
df2: pd.DataFrame = clear_outliers(df2)

# Data preprocessing
X = df2.drop('Exited', axis='columns')
y = df2['Exited']

sm = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
X, y = sm.fit_resample(X, y)  # resampled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=5, stratify=y)
print(X_train.shape, X_test.shape)
print(X_train[:10])
print(len(X_train.columns))

# ANN Model with TensorFlow
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=13, units=7,
                     kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=7, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1,
                     kernel_initializer="uniform"))
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=120)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print(classification_report(y_test, y_pred))
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
plt.figure(figsize=FIG_SIZE)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
print(y_test.shape)

# MODEL TRAINING
models = [('LOGR', LogisticRegression()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('RF', RandomForestClassifier()),
          ('SVC', SVC()), ('GBM', GradientBoostingClassifier()),
          ('XGBoost', XGBClassifier()), ('LightGBM', LGBMClassifier()),
          ('CatBoost', CatBoostClassifier()), ('ABoost', AdaBoostClassifier())]

df_result: pd.DataFrame = pd.DataFrame(
    columns=["model", "accuracy_score", "scale_method",
             "0_precision", "0_recall", "1_precision",
             "1_recall"])
index: int = 0
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, digits=2,
                                         output_dict=True)
    zero_report = class_report['0']
    one_report = class_report['1']
    df_result.at[index, ['model', 'accuracy_score', 'scale_method',
                         "0_precision", "0_recall", "1_precision",
                         "1_recall"]] = [name, score, "NA",
                                         zero_report['precision'],
                                         zero_report['recall'],
                                         one_report['precision'],
                                         one_report['recall']]
    index += 1
df_result = df_result.sort_values("accuracy_score", ascending=False)
print(df_result)

# Model TUNING
# LightGBM
lgbm_model = LGBMClassifier(silent=0, learning_rate=0.09, max_delta_step=2,
                            n_estimators=1000, boosting_type='gbdt',
                            max_depth=6, eval_metric="logloss", gamma=3,
                            base_score=0.5, num_leaves=100)
model_results(lgbm_model, X_train, y_train, y_test, X_test)

# XGBoost
xgb_model = XGBClassifier(silent=0, learning_rate=0.01, max_delta_step=5,
                          objective='reg:logistic', n_estimators=100,
                          max_depth=5, eval_metric="logloss", gamma=3,
                          base_score=0.5)
model_results(xgb_model, X_train, y_train, y_test, X_test)

# CatBoost
catboost_model = CatBoostClassifier(iterations=100, verbose=10)
model_results(catboost_model, X_train, y_train, y_test, X_test)

# Random Forest
param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2, 4, 6, 7, 8, 9],
              'n_estimators': [50, 100], 'min_samples_split': [3, 5, 6, 7]}
randFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5,
                            refit=True, verbose=0)
randFor_grid.fit(X_train, y_train)
print(randFor_grid.best_score_)
print(randFor_grid.best_params_)
print(randFor_grid.best_estimator_)
rnd_model = RandomForestClassifier(max_depth=8, max_features=8,
                                   min_samples_split=5, n_estimators=50)
model_results(rnd_model, X_train, y_train, y_test, X_test)

# Logistic Regression Model
logr_model = LogisticRegression()
model_results(logr_model, X_train, y_train, y_test, X_test)

# Decision Tree Classifier Model
dtc_model = DecisionTreeClassifier()
model_results(dtc_model, X_train, y_train, y_test, X_test)

# Ada Boost Classifier Model
abc_model = AdaBoostClassifier()
model_results(abc_model, X_train, y_train, y_test, X_test)

# Gradient Boosting Classifier Model
gbc_model = DecisionTreeClassifier()
model_results(gbc_model, X_train, y_train, y_test, X_test)

# SVC Model
svc_model = SVC(probability=True)
model_results(svc_model, X_train, y_train, y_test, X_test)

# K Neighbors Classifier Model
knc_model = KNeighborsClassifier()
model_results(knc_model, X_train, y_train, y_test, X_test)

# Cross validate model with Kfold stratified cross validation
kfold = StratifiedKFold(n_splits=5)
# Modeling step Test differents algorithms
random_state: int = 10101
classifiers = [LogisticRegression(), RandomForestClassifier(),
               KNeighborsClassifier(), SVC(), DecisionTreeClassifier(),
               GradientBoostingClassifier(), XGBClassifier(), LGBMClassifier(),
               CatBoostClassifier(), AdaBoostClassifier()]
cv_results: list = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y=y_train,
                                      scoring="accuracy", cv=kfold,
                                      n_jobs=4))
cv_means: list = []
cv_std: list = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,
                       "Algorithm": ["LogisticRegression", "RandomForest",
                                     "KNeighboors", "SVC", "DecisionTree",
                                     "GradientBoostingClassifier",
                                     "XGBClassifier", "LGBMClassifier",
                                     "CatBoostClassifier",
                                     "AdaBoostClassifier"]})
print(cv_res)
plt.figure(figsize=FIG_SIZE)
sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette=PALETTE,
            orient="h", **{'xerr': cv_std})
plt.xlabel("Mean Accuracy")
plt.title("Cross validation scores")
plt.show()
plt.savefig('cross_validation.png')
print(f"{time.time() - start_time} seconds")
