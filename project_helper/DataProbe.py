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
        my_path = os.path.abspath(DataProbe.plot_dir)
        os.chdir(my_path)
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

        print("Size of the dataset before trimming outliers" + df.shape)

    def removeOutliers(x, outlierConstant):
        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        result = a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]
        return result

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

