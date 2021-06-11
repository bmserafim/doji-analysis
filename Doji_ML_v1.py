import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
sns.set()
pd.options.display.width = 0

#importing warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#getting files from Github repository and setting up Dataframes for training and testing
url_data_rawM15 = "https://github.com/bmserafim/doji-analysis/blob/main/Dados%20importados%20(Candles%2015min)%20limpos%20(200k).csv?raw=true"
url_data_rawM15_test = "https://github.com/bmserafim/doji-analysis/raw/main/GBPUSD_M15_10k_2021.csv"
data_raw = pd.read_csv(url_data_rawM15)
data_raw_test = pd.read_csv(url_data_rawM15_test)

###---TRAINING---###

#creating a new Dataframe (for training) filtering original data
data = data_raw[["Date", "Time", "Open", "High", "Low", "Close"]]

#defining candle body size
data["body_size"] = (data["Close"] - data["Open"]) * 100000

#defining top and bottom wick sizes
data.loc[data["body_size"] >= 0, "top_wick"] = (data["High"] - data["Close"])* 100000
data.loc[data["body_size"] < 0, "top_wick"] = (data["High"] - data["Open"])* 100000

data.loc[data["body_size"] >= 0, "bottom_wick"] = (data["Open"] - data["Low"])* 100000
data.loc[data["body_size"] < 0, "bottom_wick"] = (data["Close"] - data["Low"])* 100000

#defining top/bottom wick ratio
data.loc[(data["bottom_wick"] != 0), "wick_ratio"] = (data["top_wick"]/data["bottom_wick"])

#defining if the candle is a doji
data.loc[((data["body_size"] <= 50) & (data["body_size"] >= -50) & (data["wick_ratio"] <= 1.2) & (data["wick_ratio"] >= 0.8)), "its a doji?"] = "Yes"

#creating a new Dataframe with dojis only
data_doji_columns = {"3th_previous":[], "2th_previous":[], "previous":[], "doji":[], "following":[], "following_high":[], "following_low":[], "SL hit?":[], "target hit?": []}
data_doji_clean = pd.DataFrame(data_doji_columns)

#filling data_doji_clean only with dojis
i = 0

for row in data["its a doji?"]:
    if data["its a doji?"][i] == "Yes":
        data_doji_clean = data_doji_clean.append({"doji":data["body_size"][i], "doji_high":data["High"][i], "doji_low":data["Low"][i], "following": data["body_size"][i+1],
                          "following_high": data["High"][i+1], "following_low": data["Low"][i+1], "previous": data["body_size"][i-1], "2th_previous": data["body_size"][i - 2],
                          "3th_previous": data["body_size"][i - 3]}, ignore_index=True)
    i += 1

#defining if Stop Loss (SL) was hit (doji's high or low price)
data_doji_clean["SL hit?"] = "No"
data_doji_clean.loc[(data_doji_clean["previous"] < 0) & (data_doji_clean["following_low"] <= data_doji_clean["doji_low"]), "SL hit?"] = "Yes"
data_doji_clean.loc[(data_doji_clean["previous"] > 0) & (data_doji_clean["following_high"] >= data_doji_clean["doji_high"]), "SL hit?"] = "Yes"

#creating a new DataFrame
data_doji_filtered_columns = {"3th_previous":[], "2th_previous":[], "previous":[], "doji":[], "following":[], "SL hit?":[], "target hit?":[]}
data_doji_filtered = pd.DataFrame(data_doji_filtered_columns)

#filling Dataframe with dojis that satisfy a specific condition
j = 0

for row in data_doji_clean["following"]:
    if (data_doji_clean["3th_previous"][j] > 0 and data_doji_clean["2th_previous"][j] > 0 and data_doji_clean["previous"][j] > 0) or (data_doji_clean["3th_previous"][j] < 0 and data_doji_clean["2th_previous"][j] < 0 and data_doji_clean["previous"][j] < 0):
        data_doji_filtered = data_doji_filtered.append({"doji": data_doji_clean["doji"][j], "following": data_doji_clean["following"][j], "SL hit?": data_doji_clean["SL hit?"][j],
                    "previous": data_doji_clean["previous"][j], "2th_previous": data_doji_clean["2th_previous"][j], "3th_previous": data_doji_clean["3th_previous"][j]}, ignore_index=True)
    j += 1

#defining if target (80 points) was hit
data_doji_filtered["target hit?"] = "No"
data_doji_filtered.loc[(data_doji_filtered["previous"] < 0) & (data_doji_filtered["following"] >= 40), "target hit?"] = "Yes"
data_doji_filtered.loc[(data_doji_filtered["previous"] > 0) & (data_doji_filtered["following"] <= 40), "target hit?"] = "Yes"

#defining if trade was positive (target hit without hitting stop loss)
data_doji_filtered["positive trade?"] = "No"
data_doji_filtered.loc[(data_doji_filtered["SL hit?"] == "No") & (data_doji_filtered["target hit?"] == "Yes"), "positive trade?"] = "Yes"

#removing data from following candle, to prevent data leakage
data_doji_filtered.drop(columns=["following"], inplace = True)

#defining data to be used in a logistic regression model, in order to predict if a specific trade would hit the target
x_train = data_doji_filtered.select_dtypes("float64")
y_train = data_doji_filtered["target hit?"]

logRegModel = LogisticRegression()
logRegModel.fit(x_train, y_train)

#using a dummy model to evaluate if logistic regression model was actually effective
dummy_model = DummyClassifier('most_frequent')
dummy_model.fit(x_train, y_train)

#using the same data, but now in a tree model
tree_model = DecisionTreeClassifier(max_depth = 3)
tree_model.fit(x_train, y_train)

##repeating the previous steps, but now to predict with a specific trade would be positive (hitting target but not stop loss)
x2_train = data_doji_filtered.select_dtypes("float64")
y2_train = data_doji_filtered["positive trade?"]

logRegModel2 = LogisticRegression()
logRegModel2.fit(x2_train, y2_train)

#using a dummy model to evaluate if logistic regression model was actually effective
dummy_model2 = DummyClassifier('most_frequent')
dummy_model2.fit(x2_train, y2_train)

#using the same data, but now in a tree model
tree_model2 = DecisionTreeClassifier(max_depth = 3)
tree_model2.fit(x2_train, y2_train)

###---TESTING---###

#creating a new Dataframe (for testing) filtering original data
data_test = data_raw_test[["Date", "Time", "Open", "High", "Low", "Close"]]

#defining candle body size
data_test["body_size"] = (data_test["Close"] - data_test["Open"]) * 100000

#defining top and bottom wick sizes
data_test.loc[data_test["body_size"] >= 0, "top_wick"] = (data_test["High"] - data_test["Close"])* 100000
data_test.loc[data["body_size"] < 0, "top_wick"] = (data_test["High"] - data_test["Open"])* 100000

data_test.loc[data["body_size"] >= 0, "bottom_wick"] = (data_test["Open"] - data_test["Low"])* 100000
data_test.loc[data["body_size"] < 0, "bottom_wick"] = (data_test["Close"] - data_test["Low"])* 100000

#defining top/bottom wick ratio
data_test.loc[(data["bottom_wick"] != 0), "wick_ratio"] = (data_test["top_wick"]/data_test["bottom_wick"])

#defining if the candle is a doji
data_test.loc[((data_test["body_size"] <= 50) & (data_test["body_size"] >= -50) & (data_test["wick_ratio"] <= 1.2) & (data_test["wick_ratio"] >= 0.8)), "its a doji?"] = "Yes"

#creating a new Dataframe with dojis only
data_test_doji_columns = {"3th_previous":[], "2th_previous":[], "previous":[], "doji":[], "following":[], "following_high":[], "following_low":[], "SL hit?":[], "target hit?": []}
data_test_doji_clean = pd.DataFrame(data_test_doji_columns)

#filling data_doji_clean only with dojis
h = 3

for row in data_test["its a doji?"]:
    if (data_test["its a doji?"][h] == "Yes"):
        data_test_doji_clean = data_test_doji_clean.append({"doji":data_test["body_size"][h], "doji_high":data_test["High"][h], "doji_low":data_test["Low"][h], "following": data_test["body_size"][h+1],
                          "following_high": data_test["High"][h+1], "following_low": data_test["Low"][h+1], "previous": data_test["body_size"][h-1], "2th_previous": data_test["body_size"][h - 2],
                          "3th_previous": data_test["body_size"][h - 3]}, ignore_index=True)
    h += 1
    if h >= 10000: break

#defining if Stop Loss (SL) was hit (doji's high or low price)
data_test_doji_clean["SL hit?"] = "No"
data_test_doji_clean.loc[(data_test_doji_clean["previous"] < 0) & (data_test_doji_clean["following_low"] <= data_test_doji_clean["doji_low"]), "SL hit?"] = "Yes"
data_test_doji_clean.loc[(data_test_doji_clean["previous"] > 0) & (data_test_doji_clean["following_high"] >= data_test_doji_clean["doji_high"]), "SL hit?"] = "Yes"

#creating a new DataFrame
data_test_doji_filtered_columns = {"3th_previous":[], "2th_previous":[], "previous":[], "doji":[], "following":[], "SL hit?":[], "target hit?":[]}
data_test_doji_filtered = pd.DataFrame(data_test_doji_filtered_columns)

#filling Dataframe with dojis that satisfy a specific condition
j = 0

for row in data_test_doji_clean["following"]:
    if (data_test_doji_clean["3th_previous"][j] > 0 and data_test_doji_clean["2th_previous"][j] > 0 and data_test_doji_clean["previous"][j] > 0) or (data_test_doji_clean["3th_previous"][j] < 0 and data_test_doji_clean["2th_previous"][j] < 0 and data_test_doji_clean["previous"][j] < 0):
        data_test_doji_filtered = data_test_doji_filtered.append({"doji": data_test_doji_clean["doji"][j], "following": data_test_doji_clean["following"][j], "SL hit?": data_test_doji_clean["SL hit?"][j],
                    "previous": data_test_doji_clean["previous"][j], "2th_previous": data_test_doji_clean["2th_previous"][j], "3th_previous": data_test_doji_clean["3th_previous"][j]}, ignore_index=True)
    j += 1

#defining if target (80 points) was hit
data_test_doji_filtered["target hit?"] = "No"
data_test_doji_filtered.loc[(data_test_doji_filtered["previous"] < 0) & (data_test_doji_filtered["following"] >= 40), "target hit?"] = "Yes"
data_test_doji_filtered.loc[(data_test_doji_filtered["previous"] > 0) & (data_test_doji_filtered["following"] <= 40), "target hit?"] = "Yes"

#defining if trade was positive (target hit without hitting stop loss)
data_test_doji_filtered["positive trade?"] = "No"
data_test_doji_filtered.loc[(data_test_doji_filtered["SL hit?"] == "No") & (data_test_doji_filtered["target hit?"] == "Yes"), "positive trade?"] = "Yes"

#removing data from following candle, to prevent data leakage
data_test_doji_filtered.drop(columns=["following"], inplace = True)

#defining data to be used in testing
x_test = data_test_doji_filtered.select_dtypes("float64")
y_test = data_test_doji_filtered["target hit?"]

#evaluating Logistic Regression model score
print("Logistic Regression Score ('target hit?'): %.2f" % (logRegModel.score(x_test, y_test)))

#evaluating Dummy model score, as a comparison
dummy_prediction = dummy_model.predict(x_test)
print("Dummy Prediction Score ('target hit?'): %.2f" % (accuracy_score(y_test, dummy_prediction)))

#evaluating Tree model score
print("Tree Model Score ('target hit?'): %.2f" % (tree_model.score(x_test, y_test)))

##repeating process, but now to evaluate whether the models were satisfactory to predict if a trade would be positive (hitting only target, not stop loss also)

x2_test = data_test_doji_filtered.select_dtypes("float64")
y2_test = data_test_doji_filtered["positive trade?"]

print("Logistic Regression Score ('positive trade?'): %.2f" % (logRegModel2.score(x2_test, y2_test)))

dummy_prediction2 = dummy_model2.predict(x2_test)
print("Dummy Prediction Score ('positive trade?'): %.2f" % (accuracy_score(y2_test, dummy_prediction2)))

print("Tree Model Score ('positive trade?'): %.2f" % (tree_model2.score(x2_test, y2_test)))