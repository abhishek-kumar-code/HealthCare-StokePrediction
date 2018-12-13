import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn.feature_selection

df = pd.read_csv('train_2v.csv')
# change to htm
df.to_html('data.html')
# read the data and check it
print(df.head(5))
# take a look at the outcome variable stroke
print(df['stroke'].value_counts())

# print more columns to view data
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

# split into dataframe x and y
x = df.drop('stroke', 1)
print(x.head(5))
y = df['stroke']
print(y.head(5))

# dealing with data types/ data cleaning
# model can only handle numeric data types

print(x['gender'].head(5))

# use get_dummies in pandas
# or onehotencoder in scikit learn, any of them to convert to integer
print(pd.get_dummies(x['gender']).head(5))
print(pd.get_dummies(x['work_type']).head(5))
print(pd.get_dummies(x['ever_married']).head(5))

# not useful to dummy all the fields if they have high range categories
# deicde which categorical variables you want in model
# lets check those
for col_name in x.columns:
    if x[col_name].dtypes == 'object':
        unique_cat = len(x[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(
            col_name=col_name, unique_cat=unique_cat
        ))
# to check whether worktype has any useless category
print(x['work_type'].value_counts().sort_values(ascending=False).head(10))
# seems like it is better to dummy out all the features instead of bucket low frequency
# as all of them cover pretty much high values

# create a list of features to dummy
todummy_list = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Function to dummy all the categorical values used for modeling

# # Get one hot encoding for gender
# one_hot = pd.get_dummies(df['gender'])
# # Drop column gender as it is now encoded
# x = x.drop('gender',axis = 1)
# # Join the encoded df
# x = x.join(one_hot)
#
# # Get one hot encoding for ever_married
# one_hot = pd.get_dummies(df['ever_married'])
# # Drop column ever_married as it is now encoded
# x = x.drop('ever_married',axis = 1)
# # Join the encoded df
# x = x.join(one_hot)
#
# # Get one hot encoding for work_type
# one_hot = pd.get_dummies(df['work_type'])
# # Drop column work_type as it is now encoded
# x = x.drop('work_type',axis = 1)
# # Join the encoded df
# x = x.join(one_hot)
#
# # Get one hot encoding for Residence_type
# one_hot = pd.get_dummies(df['Residence_type'])
# # Drop column Residence_type as it is now encoded
# x = x.drop('Residence_type',axis = 1)
# # Join the encoded df
# x = x.join(one_hot)
#
#
# # Get one hot encoding for smoking_status
# one_hot = pd.get_dummies(df['smoking_status'])
# # Drop column smoking_status as it is now encoded
# x = x.drop('smoking_status',axis = 1)
# # Join the encoded df
# x = x.join(one_hot)
# print(x.head(5))


# check how much of the data is missing
print(x.isnull().sum().sort_values(ascending=False).head())


def dummy_df(df, todummy_list):
    for X in todummy_list:
        dummies = pd.get_dummies(df[X], prefix=X, dummy_na=False) #setting dummy_na to true means that
        # i am creating a column named smoking_status_nan where it would be 1
        df = df.drop(X, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


x = dummy_df(x, todummy_list)

print(x.head(5))

print(x.isnull().sum().sort_values(ascending=False).head())
# shape to find number of data and columns in the dataset
print(x.shape)

# now we try to drop a row which has all values as 0
# x.dropna(how='all').shape
# but did not drop any because we have at least a 0 in any of the row, so commented


# find the mean and median of the NaN columns to replace
print(x['bmi'].mean())
print(x['bmi'].median())
# mean and median seem to be almost same 28.6 and 27.7, so replacing with mean
x['bmi'].fillna(x['bmi'].mean(), inplace=True)
# print(x.isnull().sum().sort_values(ascending=False).head())  # to check top missing values columns
# print(x['bmi'].value_counts(dropna=False))


# print(x.head(5))


def find_outliers_tukey(X):
    q1 = np.percentile(X, 25)
    q3 = np.percentile(X, 75)
    iqr = q3-q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(X.index[(X < floor)|(X > ceiling)])
    outlier_values = list(X[outlier_indices])

    return outlier_indices, outlier_values


tukey_indices, tukey_values = find_outliers_tukey(x['bmi'])
print("show outliers", np.sort(tukey_values))


def plot_histogram(x):
    plt.hist(x, color='gray', alpha=0.5)
    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.show()


# plot_histogram(x['age'])


def plot_histogram_dv(x, y):
    plt.hist(list(x[y == 0]), alpha=0.5, label='DV=0')
    plt.hist(list(x[y == 1]), alpha=0.5, label='DV=1')
    plt.title("HIstogram of '{var_name}' by DV Category".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()


# plot_histogram_dv(x['age'], y)
# plot_histogram_dv(x['bmi'], y)
# plot_histogram_dv(x['hypertension'], y)
# plot_histogram_dv(x['avg_glucose_level'], y)

# use polynomialfeatures in sklearn.preprocessing to create two way intercations for all features
def add_intercations(df):
    #get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(X) for X in combos]

    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames

    # Remove interaction terms with all 0 values
    noint_interactions = [i for i, X in enumerate(list((df == 0).all())) if X]
    df = df.drop(df.columns[noint_interactions],axis=1)

    return df


x = add_intercations(x)
print(x.head(5))


# use pca from sklearn.decomposition to find principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
x_pca = pd.DataFrame(pca.fit_transform(x))

print(x_pca.head(5))


# Feature selection and model building
# build model with processed data

# use train_test_split in sklearn.model selection to split data


# x_train, x_test, y_test = train_test_split(x, y, train_size=0.70, random_state=1)
# create training and testing vars
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# check the total no. of features grown after dummying and adding interactions terms
print("initial size", df.shape)
print("after dumming and adding interactions", x.shape)

# such a large feature can cause overfitting and also slow computing
# so selecting features now

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(x_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [x.columns[i] for i in indices_selected]

x_train_selected = x_train[colnames_selected]
x_test_selected = x_test[colnames_selected]

print(colnames_selected)


def find_model_perf(x_train, y_train, x_test, y_test):
    model = LogisticRegression(solver='lbfgs')
    model.fit(x_train, y_train)
    y_hat = [X[1] for X in model.predict_proba(x_test)]
    auc = roc_auc_score(y_test, y_hat)

    return auc


auc_processed = find_model_perf(x_train_selected, y_train, x_test_selected, y_test)
print(auc_processed)
# gives 0.77

# build model with unprocessed data
# checking with the data without preprocessing
df_unprocessed = df
df_unprocessed = df_unprocessed.dropna(axis=0, how='any')
print(df.shape)
print(df_unprocessed.shape)

# remove non-numeric columns so model does not throw an error
for col_name in df_unprocessed.columns:
    if df_unprocessed[col_name].dtypes not in ['int32', 'int64', 'float32', 'float64']:
        df_unprocessed = df_unprocessed.drop(col_name, 1)


# split into features and outcomes
x_unprocessed = df_unprocessed.drop('stroke', 1)
y_unprocessed = df_unprocessed['stroke']

# how unfeature set looks like
print(x_unprocessed.head(5))

# split unprocessed data into train and test set
# build model and assess performance
x_train_unprocessed, x_test_unprocessed, y_train, y_test = train_test_split(
    x_unprocessed, y_unprocessed, test_size=0.2, random_state=1)

auc_unprocessed = find_model_perf(x_train_unprocessed, y_train, x_test_unprocessed, y_test)
print(auc_unprocessed)

# compare model performance

print('AUC of model with data preprocessing: {auc}'.format(auc=auc_processed))
print('AUC of model with data without preprocessing: {auc}'.format(auc=auc_unprocessed))
per_improve = ((auc_processed-auc_unprocessed)/auc_unprocessed)*100
print('Model improvement of preprocessing: {per_improve}%'.format(per_improve = per_improve))
