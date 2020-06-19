import pandas as pd
import math
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import max_error
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import max_error
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.experimental import enable_iterative_imputer
# import sklearn.impute as impute
# import pandas_profiling as pp
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import DBSCAN
# from scipy import stats
# from sklearn.preprocessing import MinMaxScaler
# import scipy as sp
# from sklearn.preprocessing import RobustScaler
# from mlxtend.frequent_patterns import apriori, association_rules
# from scipy import sparse
import datetime
import os

DRAW_DIST_PLOT = False

# Create dataframes from json files
# Linux
# df_test = pd.read_json('./Test_Data.json', lines=True)
# df_train = pd.read_json('./Train Data.json', lines=True)
# Windows
fileDir = os.path.dirname(os.path.abspath(__file__))
testAbsPath = os.path.join(fileDir, 'Test_Data.json')
trainAbsPath = os.path.join(fileDir, 'Train_Data.json')

df_test = pd.read_json(testAbsPath, lines=True)
df_train = pd.read_json(trainAbsPath, lines=True)

baseTimeSpan = pd.Timestamp("2012-01-01 00:00:00+00:00")


def jsonToDataFrame(json):
    res = pd.DataFrame()
    res["len"] = json.text.agg(len)
    # removed because it can't be used
    # res["age"] = json["created_at"].agg(lambda x: (pd.Timestamp(x) - baseTimeSpan))
    res["truncated"] = json.truncated
    res["hashtags"] = json.entities.agg(lambda x: len(x["hashtags"]))
    res["user_mentions"] = json.entities.agg(lambda x: len(x["user_mentions"]))
    # removed because all values are 'None"
    # res["geo"] = json.geo
    res["favorite_count"] = json.favorite_count
    res["is_quote_status"] = json.is_quote_status
    res["lang"] = json.lang
    # Will be 1 if sensitive, else 0
    res["possibly_sensitive"] = json.possibly_sensitive.agg(lambda x: 0 if math.isnan(x) or x == 0 else 1)

    def filledReader(name):
        return lambda x: None if x is None else x[name]

    def booleanToBinary(column):
        return column.apply(lambda x: 0 if not x else 1)

    def userInfoToRes(str, user):
        res[str + "__created_at"] = user.agg(filledReader("created_at")).agg(pd.Timestamp).astype(int)\
            / (10**9 * 1440 * 60 * 365)  # Account age in years
        res[str + "__default_profile_image"] = user.agg(filledReader("default_profile_image"))
        # remove because nemifahmam chie
        # res[str + "__entities"] = user.agg(filledReader("entities"))
        res[str + "__favourites_count"] = user.agg(filledReader("favourites_count"))
        res[str + "__followers_count"] = user.agg(filledReader("followers_count"))
        res[str + "__following"] = user.agg(filledReader("following"))
        res[str + "__friends_count"] = user.agg(filledReader("friends_count"))
        res[str + "__has_extended_profile"] = user.agg(filledReader("has_extended_profile"))
        res[str + "__is_translator"] = user.agg(filledReader("is_translator"))
        res[str + "__lang"] = user.agg(filledReader("lang"))
        res[str + "__listed_count"] = user.agg(filledReader("listed_count"))
        # res[str + "__geo"] = user.agg(filledReader("geo"))
        res[str + "__protected"] = user.agg(filledReader("protected"))
        res[str + "__statuses_count"] = user.agg(filledReader("statuses_count"))
        res[str + "__verified"] = user.agg(filledReader("verified"))

    userInfoToRes("current-user", json.user)
    userInfoToRes("original-user", json.retweeted_status.agg(lambda x: None if type(x) is float else x["user"]))

    # Change True/False into 0/1
    for (columnName, columnData) in res.iteritems():
        print(columnName, res.dtypes[columnName])
        if res.dtypes[columnName] == 'bool':
            res[columnName] = booleanToBinary(res[columnName])

    # 1->Short Twee(1-50) / 2->Medium Tweet(50-100) / 3->Long Tweet(100-140)
    res['len'] = res['len'].agg(lambda x: 1 if x < 51 else (2 if x < 101 else 3))

    # Draws histograms for each column. used for categorizing
    if DRAW_DIST_PLOT:
        for (columnName, columnData) in res.iteritems():
            sns.distplot(columnName, kde=False)
            plt.title(columnName)
            plt.show()

    return res


X_train = jsonToDataFrame(df_train)
# X_test = jsonToDataFrame(df_test)
