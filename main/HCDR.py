#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier # Import random Forest Classifier
from sklearn.linear_model import LogisticRegression # Import random Forest Classifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.datasets import fetch_datasets
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from imblearn.combine import SMOTEENN, SMOTETomek
from statsmodels.stats.outliers_influence import variance_inflation_factor # for VIF multicollinearity
from joblib import Parallel, delayed

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) # to run prediction score with 
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


# Loading helper classes

# In[2]:


# %load ../project_helper/DataProbe.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os, path
from statsmodels.stats.outliers_influence import variance_inflation_factor # for VIF multicollinearity
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder


class DataProbe:
    plot_dir = "../project_helper/data/plots"
    box_plot_dir = "../project_helper/data/plots/box"
    cat_plot_dir = "../project_helper/data/plots/cat"
    # RFC = RandomForestClassifier(n_estimators=100, warm_start=True, oob_score=True, max_features="sqrt")
    # LR = LogisticRegression(C=0.0001)

    @staticmethod
    def get_missing_data(df: pd.DataFrame):
        missing_values_total = df.isnull().sum()
        missing_values_perc = 100 * missing_values_total / len(df)

        missing_data = pd.concat([missing_values_total, missing_values_perc], axis=1)
        missing_data.sort_values([1], ascending=False, inplace=True)
        return missing_data

    @staticmethod
    def save_plot_for_columns(df: pd.DataFrame):
        sns.set(style="darkgrid")
        #my_path = os.path.abspath(DataProbe.plot_dir)
        # os.chdir(my_path)
        for col in df.columns:
            col_values = df[col].value_counts()
            sns.barplot(col_values.index, col_values.values, alpha=0.9)
            plt.figure()
            plt.title('Frequency Distribution of {}'.format(col))
            plt.ylabel('Number of Occurrences', fontsize=12)
            plt.xlabel("Field {}".format(col), fontsize=12)
            # plt.savefig('plot_{}.png'.format(col))
            plt.show()

    @staticmethod
    def cat_plot_for_columns(df: pd.DataFrame):
        sns.set(style="darkgrid")
        my_path = os.path.abspath(DataProbe.cat_plot_dir)
        os.chdir(my_path)
        for col in df.columns:
            sns.boxplot(data=df[col])
            plt.figure()
            plt.title('Frequency Distribution of {}'.format(col))
            # plt.savefig('plot_{}.png'.format(col))
            plt.show()

    @staticmethod
    def check_for_outliers(df: pd.DataFrame):
        os.mkdir(DataProbe.box_plot_dir)
        os.chdir(DataProbe.box_plot_dir)
        new_df = df.select_dtypes([np.number])
        for col in new_df.columns:
            plt.boxplot(df[col])
            plt.savefig('boxplot_{}.png'.format(col))
            plt.show()

    @staticmethod
    def remove_outliers(df: pd.DataFrame, sdFromPercentile: float):
        print("Size of the dataset before trimming outliers" + df.shape)
        q1, q3 = np.percentile(df, [25, 75])
        iqr = q3 - q1
        lb = q1 - (sdFromPercentile * iqr)
        ub = q3 + (sdFromPercentile * iqr)
        print(IQR)
        
        IQR_outliers = ((df['amount'] < (Q1 - 1.5 * IQR)) |(df['amount'] > (Q3 + 1.5 * IQR)))
        print(IQR_outliers)
        print("Size of the dataset before trimming outliers" + df.shape)

    @staticmethod
    def find_unique_values_columns(df: pd.DataFrame):
        for col in df.columns:
            print("unique value for {} is {}".format(col, list(df[col].unique())))

    @staticmethod
    def calculate_VIF(df: pd.DataFrame):
        threshold = 5.0
        columns = df.columns
        vif = pd.DataFrame()
        vif["VIF factor"] = [Parallel(n_jobs=-1, verbose=5)(
            delayed(variance_inflation_factor)(df[columns].values, df.columns.get_loc(i)) for i in df.columns)]
        vif["features"] = X.columns
        print(vif.round(1))

    @staticmethod
    def plot_target(df: pd.DataFrame):
        colors = ["#0101DF", "#DF0101"]
        sns.countplot('TARGET', data=df, palette=colors)
        plt.title('Class Distributions \n (0: No Default || 1: Default)', fontsize=14)

    # copied from https://www.kaggle.com/npramod/techniques-to-deal-with-imbalanced-data
    @staticmethod
    def transform(transformer, X, y):
        print("Transforming {}".format(transformer.__class__.__name__))
        X_resampled, y_resampled = transformer.fit_sample(X.values, y.values.ravel())
        return transformer.__class__.__name__, pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)

    # for label encoding
    @staticmethod
    def label_encode(training_data : pd.DataFrame, testing_data : pd.DataFrame):

        encoder = LabelEncoder()
        le_count = 0

        # Iterate through the columns
        for col in training_data:
            if training_data[col].dtype == 'object':
                # If 2 or fewer unique categories
                if len(list(training_data[col].unique())) <= 2:
                    # Train on the training data
                    encoder.fit(training_data[col])
                    # Transform both training and testing data
                    training_data[col] = encoder.transform(training_data[col])
                    testing_data[col] = encoder.transform(testing_data[col])

                    # Keep track of how many columns were label encoded
                    le_count += 1

        print('%d columns were label encoded.' % le_count)
        return training_data, testing_data

    


# In[3]:


def listdir(path):
    if os.path.exists(path):
        return os.listdir(path)
    else:
        return []
    
def get_files_with_path(path):
    file_list = listdir(path)
    file_ref = dict()
    for f in file_list:
        file_ref[f] = path + '/' + f
    return file_ref


# In[4]:


# regression helper functions
RFC = RandomForestClassifier(n_estimators=100, warm_start=True, oob_score=True, max_features="sqrt", random_state=0)
RFC_AFTER_FEATURE_REDUCTION = RandomForestClassifier(n_estimators=100, warm_start=True, oob_score=True, max_features="sqrt", random_state=0)
LR = LogisticRegression(C=0.0001)
CLF = DecisionTreeClassifier()
scores = []

def run_DT(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame):
    # Train Decision Tree Classifer
    clf = CLF.fit(x_train, y_train)

    # Predict the response for test datase
    y_pred = clf.predict(x_test)
    return clf, y_pred

def run_RFC(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame):
    # Train Random Forest Classifer
    rfc = RFC.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = rfc.predict(x_test)
    return rfc, y_pred

def run_RFC_with_important_features(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame):
    # Train Random Forest Classifer
    rfc = RFC_AFTER_FEATURE_REDUCTION.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = rfc.predict(x_test)
    return rfc, y_pred

def run_LR(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame):
    lr = LR.fit(x_train, y_train)
    y_logistic_pred = lr.predict(x_test)
    return lr, y_logistic_pred

def add_to_scores(regression_type: str, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                      y_test: pd.Series, y_pred: pd.Series):
    scores.append((regression_type,
                        metrics.f1_score(y_test, y_pred, average='macro'),
                        metrics.precision_score(y_test, y_pred, average='macro'),
                        metrics.recall_score(y_test, y_pred, average='macro'),
                        metrics.accuracy_score(y_test, y_pred),
                        metrics.confusion_matrix(y_test, y_pred)))

def print_scores():
    sampling_results = pd.DataFrame(scores, columns=['Sampling/Regression Type', 'f1', 'precision', 'recall', 'accuracy',
                                                         'confusion_matrix'])
    return sampling_results


# Phase 1 : checking and correcting data

# In[5]:


input_path = "../data/input"
fileList = listdir(input_path)
print(fileList)


# In[6]:


file_ref = get_files_with_path(input_path)
print(file_ref)


# In[7]:


training_data = pd.read_csv(file_ref.get("application_train.csv"))
testing_data = pd.read_csv(file_ref.get("application_test.csv"))


# verbose=True, null_counts=True to print all columns information, specifically interested in datatype

# In[8]:


print(training_data.info(verbose=True, null_counts=True))


# In[9]:


training_data.describe()


# In[10]:


print("training data before drop na : {}".format(training_data.shape))


# ## Phase 2 : Dropping missing data columns

# drop columns with more than 80% missing data

# In[11]:


training_data.dropna(thresh=len(training_data)*.2, axis=1, inplace=True)
print("training data after drop na : {}".format(training_data.shape))


# So practically, no effect

# Let's try to find out the missing data

# In[12]:


missing_data = DataProbe.get_missing_data(training_data)
print(missing_data)


# In[13]:


training_data_categorical = training_data.select_dtypes(include=['object']).copy()
training_data_categorical.head()


# In[14]:


training_data_categorical.info(verbose=True, null_counts=True)


# In[15]:


DataProbe.find_unique_values_columns(training_data_categorical)


# ## Phase 3 : Categorical data encoding

# In[16]:


# let's first encode categorical data
# first we will encode data taht has binary values( using values_count size <=2)
training_data, testing_data = DataProbe.label_encode(training_data, testing_data)
print(training_data.info(verbose=True, null_counts=True))
print(testing_data.info(verbose=True, null_counts=True))


# In[17]:


# one-hot encoding of categorical variables
training_data = pd.get_dummies(training_data)
testing_data = pd.get_dummies(testing_data)


# In[18]:


training_data.replace([np.inf, -np.inf], np.nan)
training_data.fillna(method='ffill', inplace=True)
testing_data.replace([np.inf, -np.inf], np.nan)
testing_data.fillna(method='ffill', inplace=True)


# # As this data was part of a kaggle competition, training data doesn't have target variable

# <i>The testing data doesn't have TARGET (y) column!
# As this data set is part of the kaggle competition that has already expired,
# the way to validate your dataset for target values is using as a separate submission function available on kaggle</i>

# In[19]:


print('target' in training_data.columns.str.lower())


# In[20]:


print('target' in testing_data.columns.str.lower())


# # The decision tree approach ( refered from https://www.datacamp.com/community/tutorials/decision-tree-classification-python ) to help me identify feature importance

# In[21]:


training_data.replace([np.inf, -np.inf], np.nan)
training_data.fillna(method='ffill', inplace=True)
training_data.dropna(inplace=True)
data_decision_tree = training_data.copy()
# data_decision_tree = training_data.copy()
data_decision_tree_Y = data_decision_tree['TARGET']
feature_cols = list(data_decision_tree.columns.difference(['TARGET']))

X = data_decision_tree[feature_cols]
Y = data_decision_tree_Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)  # 80% training and 20% test

# regressionHelper = RegressionHelper()
clf, y_pred = run_DT(X_train, y_train, X_test)
add_to_scores("Decision Tree Classifier", X_train, y_train, X_test, y_test, y_pred)


# ## Running with random forest classifier

# In[22]:


# running random forest
rfc, y_pred = run_RFC(X_train, y_train, X_test)
add_to_scores("Random Forest Classifier_without_sampling", X_train, y_train, X_test, y_test, y_pred)


# # Feature reduction

# In[23]:


# lets try to reduce the number of features
feature_importances = pd.DataFrame(rfc.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances = pd.Series(rfc.feature_importances_,index=X_train.columns)
feature_importances.nlargest(20).plot(kind='barh')
plt.show()


# ## Running with logistic regression

# In[24]:


# Running logistic regression
lr, y_logistic_pred = run_LR(X_train, y_train, X_test)
add_to_scores("Logistic Regression", X_train, y_train, X_test, y_test, y_pred)


# ## Feature importance
# <i> As a part of random forest classifier inbuilt feature importances, I will select the n=20 largest 'importances' features

# In[25]:


DataProbe.plot_target(training_data)


# In[26]:


print(training_data['TARGET'].value_counts())


# In[27]:


list_of_important_columns = feature_importances.nlargest(20).index.values
list_of_important_columns = np.append(list_of_important_columns, 'TARGET')
# print(list_of_important_columns)
training_data_important = training_data[list_of_important_columns]
DataProbe.plot_target(training_data_important)


# In[28]:


print(training_data_important['TARGET'].value_counts())


# ## Using the important features to run random forest

# In[29]:


model_data_train = training_data_important.copy()
model_data_train_Y = model_data_train['TARGET']
feature_cols = list(model_data_train.columns.difference(['TARGET']))
X = model_data_train[feature_cols]
Y = model_data_train_Y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
rfc, y_pred = run_RFC_with_important_features(X_train, y_train, X_test)
add_to_scores("Random Forest Classifier reduced features", X_train, y_train, X_test, y_test, y_pred)


# # Sampling
# <i> Though the result are pretty good after the last regression run, I am going to try some sampling 
# * SMOTE
# * RandomOverSampler
# * NearMiss
# * RandomUnderSampler
# * SMOTEENN
# * SMOTETomek
#     
# # Based on https://www.kaggle.com/npramod/techniques-to-deal-with-imbalanced-data

# In[30]:


# Running some sampling

datasets = []
datasets.append(("base",X_train,y_train))
datasets.append(DataProbe.transform(SMOTE(n_jobs=-1),X_train,y_train))
datasets.append(DataProbe.transform(RandomOverSampler(),X_train,y_train))
datasets.append(DataProbe.transform(NearMiss(n_jobs=-1),X_train,y_train))
datasets.append(DataProbe.transform(RandomUnderSampler(),X_train,y_train))
datasets.append(DataProbe.transform(SMOTEENN(),X_train,y_train))
datasets.append(DataProbe.transform(SMOTETomek(),X_train,y_train))


# In[38]:


model_data_train = training_data_important.copy()
model_data_train_Y = model_data_train['TARGET']
feature_cols = list(model_data_train.columns.difference(['TARGET']))
XX = model_data_train[feature_cols]
YY = model_data_train_Y
for transformer_type, X, Y in datasets:
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=0)
    rfc, rfc_pred = run_RFC_with_important_features(X_train, y_train, X_test)
    scores.append((transformer_type,
                    metrics.f1_score(y_test,rfc_pred, average='macro'),
                           metrics.precision_score(y_test,rfc_pred, average='macro'),
                           metrics.recall_score(y_test,rfc_pred, average='macro'),
                           metrics.accuracy_score(y_test,rfc_pred),
                           metrics.confusion_matrix(y_test,rfc_pred)))
    


# In[ ]:


correlations = X_train.corr()


# lets try to plot this using heatmap

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(correlations,xticklabels=correlations.columns,
                 yticklabels=correlations.columns, cbar=True, vmin=-0.5, vmax=0.5,
                 fmt='.2f', annot_kws={'size': 10}, annot=True, 
                 square=True, cmap=plt.cm.Blues)
ticks = np.arange(correlations.shape[0]) + 0.5
# ax.set_xticks(ticks)
# ax.set_xticklabels(correlations.columns, rotation=90, fontsize=20)
# ax.set_yticks(ticks)
# ax.set_yticklabels(correlations.columns, rotation=360, fontsize=20)

ax.set_title('correlation matrix')
plt.tight_layout()
plt.savefig("corr_matrix_incl_anno_double.png", dpi=300)


# In[43]:


training_data_copy = training_data_important.copy()
print("Size of the dataset before trimming outliers", training_data_copy.shape)
for col in training_data_copy:
    q1, q3 = np.percentile(training_data_copy[col], [5, 95])
    iqr = q3 - q1
    lb = q1 - (1.5 * iqr)
    ub = q3 + (1.5 * iqr)
    # print("Column {} iqr {}".format(col, iqr))
    # QR_outliers = ((training_data[col] < lb) |(training_data[col] > ub))
    outlier = [lb, ub]
    # print("Column {} iqr {}".format(col, outlier))
    training_data_copy = training_data_copy[~(((training_data[col] < lb)|(training_data[col] > ub)))]
    print("After deleting col {} , target value count {} ".format(col,training_data_copy['TARGET'].value_counts()))


# After deleting all outliers,  
# target value count 
# 0    278547
# 1     24542
# Name: TARGET, dtype: int64 

# In[45]:


training_data_copy.replace([np.inf, -np.inf], np.nan)
training_data_copy.fillna(method='ffill', inplace=True)
training_data_copy.dropna(inplace=True)
data_decision_tree = training_data_copy.copy()
data_decision_tree_Y = data_decision_tree['TARGET']
feature_cols = list(data_decision_tree.columns.difference(['TARGET']))

X = data_decision_tree[feature_cols]
Y = data_decision_tree_Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  # 80% training and 20% test
rfc, y_pred = run_RFC_with_important_features(X_train, y_train, X_test)
add_to_scores("Random Forest Classifier_outlier_removed", X_train, y_train, X_test, y_test, y_pred)


# In[46]:


sampling_results = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','confusion_matrix'])
sampling_results.head


# In[47]:


sampling_results


# In[ ]:




