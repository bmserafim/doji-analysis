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


#importing warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#getting file from Github repository and setting up a Dataframe
url_data_rawM15 = "https://github.com/bmserafim/doji-analysis/blob/main/Dados%20importados%20(Candles%2015min)%20limpos%20(200k).csv?raw=true"
data_raw = pd.read_csv(url_data_rawM15)

#creating a new Dataframe filtering original data
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

#m = 0

#for row in data_doji_clean["following"]:
#    if (data_doji_clean["previous"][m] < 0) and (data_doji_clean["following_low"][m] <= data_doji_clean["doji_low"][m]):
#        data_doji_clean["SL hit?"][m] = "Yes"
#    elif (data_doji_clean["previous"][m] > 0) and (data_doji_clean["following_high"][m] >= data_doji_clean["doji_high"][m]):
#        data_doji_clean["SL hit?"][m] = "Yes"
#    else:
#        data_doji_clean["SL hit?"][m] = "No"
#    m += 1

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
data_doji_filtered.loc[(data_doji_filtered["previous"] < 0) & (data_doji_filtered["following"] >= 80), "target hit?"] = "Yes"
data_doji_filtered.loc[(data_doji_filtered["previous"] > 0) & (data_doji_filtered["following"] <= 80), "target hit?"] = "Yes"

#defining if trade was positive (target hit without hitting stop loss)
data_doji_filtered["positive trade?"] = "No"
data_doji_filtered.loc[(data_doji_filtered["SL hit?"] == "No") & (data_doji_filtered["target hit?"] == "Yes"), "positive trade?"] = "Yes"

#k = 0

#for row in data_doji_filtered["following"]:
#    if (data_doji_filtered["previous"][k] < 0 and data_doji_filtered["SL hit?"][k] == "No" and data_doji_filtered["following"][k] >= 40) or (data_doji_filtered["previous"][k] > 0 and data_doji_filtered["SL hit?"][k] == "No" and data_doji_filtered["following"][k] <= 40):
#        data_doji_filtered["target hit?"][k] = "Yes"
#        #data_doji_filtered["SL hit?"][k] = data_doji_clean["SL hit?"][k]

#    else:
#       data_doji_filtered["target hit?"][k] = "No"
#        #data_doji_filtered["SL hit?"][k] = data_doji_clean["SL hit?"][k]
#    k += 1

#removing data from following candle, to prevent a data leak
data_doji_filtered.drop(columns=["following"], inplace = True)

#defining data to be used in a logistic regression model, in order to predict if a specific trade would hit the target
x = data_doji_filtered.select_dtypes("float64")
y = data_doji_filtered["target hit?"]

logRegModel = LogisticRegression()
x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size = 0.1, stratify=y, random_state=800)
logRegModel.fit(x_training, y_training)
logRegModel.score(x_testing, y_testing)
print("Logistic Regression Score ('target hit?'): ", round(logRegModel.score(x_testing, y_testing), 2))

#using a dummy model to evaluate if logistic regression model was actually effective
dummy_model = DummyClassifier('most_frequent')
dummy_model.fit(x_training, y_training)
dummy_prediction = dummy_model.predict(x_testing)
accuracy_score(y_testing, dummy_prediction)
print("Dummy Prediction Score ('target hit?'): ", round(accuracy_score(y_testing, dummy_prediction), 2))

#using the same data, but now in a tree model
tree_model = DecisionTreeClassifier(max_depth = 3)
tree_model.fit(x_training, y_training)
print("Tree Model Score ('target hit?'): ", round(tree_model.score(x_testing, y_testing), 2))

##repeating the previous steps, but now to predict with a specific trade would be positive (hitting target but not stop loss)
x = data_doji_filtered.select_dtypes("float64")
y = data_doji_filtered["positive trade?"]

logRegModel = LogisticRegression()
x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size = 0.1, stratify=y, random_state=800)
logRegModel.fit(x_training, y_training)
logRegModel.score(x_testing, y_testing)
print("Logistic Regression Score ('positive trade?'): ", round(logRegModel.score(x_testing, y_testing), 2))

#using a dummy model to evaluate if logistic regression model was actually effective
dummy_model = DummyClassifier('most_frequent')
dummy_model.fit(x_training, y_training)
dummy_prediction = dummy_model.predict(x_testing)
accuracy_score(y_testing, dummy_prediction)
print("Dummy Prediction Score ('positive trade?'): ", round(accuracy_score(y_testing, dummy_prediction), 2))

#using the same data, but now in a tree model
tree_model = DecisionTreeClassifier(max_depth = 3)
tree_model.fit(x_training, y_training)
print("Tree Model Score ('positive trade?'): ", round(tree_model.score(x_testing, y_testing), 2))