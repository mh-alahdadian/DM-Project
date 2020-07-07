import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

DRAW_DIST_PLOT = False  # Draw column histograms
LOG_STEPS = True  # Log program steps in console
BINNING = False  # Apply binning to selected columns

USE_KNN = False  # Use KNN model
USE_RF = True  # Use Random Forest model
USE_BAGGING = False  # Use Bagging model


# Create dataframes from json files
df_test = pd.read_json(join('.', 'Test_Data.json'), lines=True)
df_train = pd.read_json(join('.', 'Train_Data.json'), lines=True)

baseTimeSpan = pd.Timestamp("2012-01-01 00:00:00+00:00")


def jsonToDataFrame(json, include_outlier):
    res_x = pd.DataFrame()
    res_y = pd.DataFrame()
    res_x["len"] = json.text.agg(len)
    res_x["truncated"] = json.truncated
    res_x["hashtags"] = json.entities.agg(lambda x: len(x["hashtags"]))
    res_x["user_mentions"] = json.entities.agg(lambda x: len(x["user_mentions"]))
    # Removed because nearly ALL the values are 0
    # res_x["favorite_count"] = json.favorite_count
    res_x["is_quote_status"] = json.is_quote_status
    res_x["lang"] = json.lang

    # Will be 1 if sensitive, else 0. Removed because all values are 0.
    # res_x["possibly_sensitive"] = json.possibly_sensitive.agg(lambda x: 0 if math.isnan(x) or x == 0 else 1)

    def filledReader(name, defaultValue=None):
        return lambda x: defaultValue if x is None else x[name]

    def booleanToBinary(column):
        return column.apply(lambda x: 0 if not x else 1)

    # Removes outlier data
    def remove_outlier(df_in, column):
        # IQR method
        # q1 = df_in[column].quantile(0.25)
        # q3 = df_in[column].quantile(0.75)
        # iqr = q3 - q1  # Inter-quartile range
        # fence_low = q1 - 1.5 * iqr
        # fence_high = q3 + 1.5 * iqr

        # 6-sigma method
        fence_low = df_in[column].mean() - df_in[column].std() * 3
        fence_high = df_in[column].mean() + df_in[column].std() * 3

        return df_in[(df_in[column] < fence_high) & (df_in[column] > fence_low)]

    # Applies binnings to data
    def apply_binning(df_in, column):
        bins = [df_in[column].quantile(0.00),
                df_in[column].quantile(0.25),
                df_in[column].quantile(0.50),
                df_in[column].quantile(0.75),
                df_in[column].quantile(1.00)]
        return pd.cut(df_in[column], bins, labels=False, duplicates='drop')

    outlier_columns = []
    def userInfoToRes(str, user):
        # Account age in years
        res_x[str + "__created_at"] = user.agg(filledReader("created_at")) \
                                          .agg(pd.Timestamp).astype(int) / (10 ** 9 * 1440 * 60 * 365) - 36
        # if null, __created_at will be -300. This line will fix the error.
        res_x[str + "__created_at"] = res_x[str + "__created_at"].agg(lambda x: -1 if x < 0 else x)
        res_x[str + "__default_profile_image"] = user.agg(filledReader("default_profile_image", False))
        res_x[str + "__favourites_count"] = user.agg(filledReader("favourites_count"))
        res_x[str + "__followers_count"] = user.agg(filledReader("followers_count"))
        # All the values are 0
        # res_x[str + "__following"] = user.agg(filledReader("following", False))
        # friends_count == followers_count
        # res_x[str + "__friends_count"] = user.agg(filledReader("friends_count"))
        # res_x[str + "__has_extended_profile"] = user.agg(filledReader("has_extended_profile", False))
        res_x[str + "__lang"] = user.agg(filledReader("lang", ""))
        res_x[str + "__listed_count"] = user.agg(filledReader("listed_count"))
        # All the values are 0
        # res_x[str + "__protected"] = user.agg(filledReader("protected", False))
        res_x[str + "__statuses_count"] = user.agg(filledReader("statuses_count"))
        res_x[str + "__verified"] = user.agg(filledReader("verified", False))

        outlier_columns.extend([str + "__favourites_count", str + "__followers_count",
                                str + "__listed_count", str + "__statuses_count"])

    userInfoToRes("retweeter-user",
                  json.agg(lambda x: None if type(x.retweeted_status) is not dict else x.user, axis="columns"))
    userInfoToRes("original-user",
                  json.agg(lambda x: x.user if type(x.retweeted_status) is not dict else x.retweeted_status["user"],
                           axis="columns"))

    # 1 if the tweet is viral (retweets are above median), 0 if not
    res_x['viral'] = json.retweeted_status.agg(lambda x: None if type(x) is not dict else x["retweet_count"])
    res_x['viral'] = res_x['viral'].agg(lambda x: 1 if x >= res_x['viral'].median() else 0)

    # Draws histograms for each column. used for categorizing
    if DRAW_DIST_PLOT:
        for (columnName, columnData) in res_x.iteritems():
            if not res_x.dtypes[columnName] == 'object':
                print(res_x[columnName].describe())
                sns.distplot(columnData, kde=False)
                plt.title(columnName)
                plt.show()

    for (columnName, columnData) in res_x.iteritems():
        columnType = res_x.dtypes[columnName]
        # Change float into int
        if columnType == 'float64':
            res_x[columnName] = res_x[columnName].fillna(0.0).astype(int)

        # Change True/False into 0/1
        if columnType == 'bool':
            res_x[columnName] = booleanToBinary(res_x[columnName])

    # Remove all outlier data
    if not include_outlier:
        # Remove tweets that are longer than 140 char.
        indexNamesTop = res_x[res_x['len'] > 140].index
        res_x.drop(indexNamesTop, inplace=True)

        # Remove all rows with negative int values. Currently Disabled
        # for (columnName, columnData) in res_x.iteritems():
        #     columnType = res_x.dtypes[columnName]
        #     if columnType == 'int64' or columnType == 'int32':
        #         indexNames = res_x[res_x[columnName] < 0].index
        #         res_x.drop(indexNames, inplace=True)

        # Remove outlier data and apply binning.
        for (columnName) in outlier_columns:
            res_x = remove_outlier(res_x, columnName)
            # Apply binning
            if BINNING:
                res_x[columnName] = apply_binning(res_x, columnName)
                if res_x.dtypes[columnName] == 'float64':
                    res_x[columnName] = res_x[columnName].fillna(0.0).astype(int)

    # 1->Short Tweet(1-50) / 2->Medium Tweet(50-100) / 3->Long Tweet(100-140)
    res_x['len'] = res_x['len'].agg(lambda x: 1 if x < 51 else (2 if x < 101 else 3))

    res_y['viral'] = res_x['viral']
    res_x = res_x.drop('viral', axis=1)

    res_x = pd.get_dummies(res_x)
    return res_x, res_y


Train_X, Train_Y = jsonToDataFrame(df_train, False)
if LOG_STEPS:
    print("Train Data Loaded!")

X_train, X_test, Y_train, Y_test = train_test_split(Train_X, Train_Y, test_size=0.25)

# Changing from columns to values
# https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
Y_train = Y_train.values.ravel()
Y_test = Y_test.values.ravel()

# Models start from here

# KNN Model
if USE_KNN:
    score_knn = []
    weights = ['uniform', 'distance']
    # weights = ['uniform']
    for i in np.arange(1, 5, 1):
        for j in np.arange(1, 5, 1):
            for k in weights:
                if LOG_STEPS:
                    print("Calculating KNN Score with n =", i, ", p =", j, ", weights =", k)
                knn = KNeighborsClassifier(n_neighbors=i, p=j, weights=k)
                knn.fit(X_train, Y_train)
                score_knn.append([i, j, k, np.mean(cross_val_score(knn, X_train, Y_train, scoring='recall', cv=5))])

    # Calculating the best parameters
    score_knn = pd.DataFrame(score_knn)
    score_knn = score_knn.sort_values(by=3, ascending=False)

    print("The best n_neigbors:", score_knn.iat[0, 0])
    print("The best p:", score_knn.iat[0, 1])
    print("The best weights:", score_knn.iat[0, 2], '\n')

    knn = KNeighborsClassifier(n_neighbors=score_knn.iat[0, 0], p=score_knn.iat[0, 1], weights=score_knn.iat[0, 2])

    knn.fit(X_train, Y_train)

    print("KNN Score:", knn.score(X_test, Y_test))
    print("Recall Score:", recall_score(Y_test, knn.predict(X_test)))
    print("Precision Score:", precision_score(Y_test, knn.predict(X_test)))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, knn.predict(X_test)), "\n\n")


# Random Forest
if USE_RF:
    score_rf = []
    # max_features = ['auto', 'sqrt', 'log2']
    max_features = ['auto']
    for i in np.arange(140, 155, 1):
        for j in max_features:
            if LOG_STEPS:
                print("Calculating RF Score with n_estimators =", i, ", max_features =", j)
            rf = RandomForestClassifier(n_estimators=i, max_features=j)
            rf.fit(X_train, Y_train)
            score_rf.append([i, j, np.mean(cross_val_score(rf, X_train, Y_train, scoring='recall', cv=5))])

    score_rf = pd.DataFrame(score_rf)
    score_rf = score_rf.sort_values(by=2, ascending=False)

    print("The best n_estimator:", score_rf.iat[0, 0])
    print("The best max_features:", score_rf.iat[0, 1], "\n")

    rf = RandomForestClassifier(n_estimators=score_rf.iat[0, 0], max_features=score_rf.iat[0, 1])
    rf.fit(X_train, Y_train)

    print("Random Forest Score:", rf.score(X_test, Y_test))
    print("Recall Score:", recall_score(Y_test, rf.predict(X_test)))
    print("Precision Score:", precision_score(Y_test, rf.predict(X_test)))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, rf.predict(X_test)))

    # get importance
    importance = rf.feature_importances_
    df_importance = pd.DataFrame(
        {
            'Name': X_train.columns,
            'Importance': importance
        }
    )
    # Sort and save importance
    df_importance = df_importance.sort_values(by=['Importance'], ascending=False)
    df_importance.to_excel(join('.', 'Importances.xlsx'))
    if LOG_STEPS:
        print("Importance Data Saved!")


# Bagging
if USE_BAGGING:
    score_bag = []
    for i in np.arange(1, 20, 1):
        for j in np.arange(1, 20, 1):
            if LOG_STEPS:
                print("Calculating Bagging Score with n_estimators =", i, ", max_samples =", j)
            bag = BaggingClassifier(n_estimators=i, max_samples=j)
            bag.fit(X_train, Y_train)
            score_bag.append([i, j, np.mean(cross_val_score(bag, X_train, Y_train, scoring='recall', cv=5))])

    score_bag = pd.DataFrame(score_bag)
    score_bag = score_bag.sort_values(by=2, ascending=False)

    print("The best n_estimator:", score_bag.iat[0, 0])
    print("The best max_samples:", score_bag.iat[0, 1], "\n")

    bag = BaggingClassifier(n_estimators=score_bag.iat[0, 0], max_samples=score_bag.iat[0, 1])
    bag.fit(X_train, Y_train)

    print("Bagging Score:", bag.score(X_test, Y_test))
    print("Recall Score:", recall_score(Y_test, bag.predict(X_test)))
    print("Precision Score:", precision_score(Y_test, bag.predict(X_test)))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, bag.predict(X_test)), "\n")

X_test, Y_test = jsonToDataFrame(df_test, True)
if LOG_STEPS:
    print("Test Data Loaded!")

# Add the missing columns between test and train data (all of then are language columns)
for (columnName, columnData) in X_train.iteritems():
    if not (columnName in X_test.columns):
        X_test[columnName] = 0

for (columnName, columnData) in X_test.iteritems():
    if not (columnName in X_train.columns):
        X_train[columnName] = 0

# Predict and write to file
Y_test = rf.predict(X_test)
df_test['Viral_prediction'] = Y_test
df_test.to_json(join('.', 'Test_Final.json'))
if LOG_STEPS:
    print("New Test Data Saved!")
