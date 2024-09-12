import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



penguin_df = pd.read_csv("penguins.csv")
# print(penguin_df.head())

penguin_df.dropna(inplace=True)
output = penguin_df["species"]

features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

#one-hot encoding
features = pd.get_dummies(features)
print(features) #see all the resulting numericized variables (lots of bools for categories)

# output = 0/1/2/1 or similar
# uniques = "a", "b", "c" - no duplicates
output, uniques = pd.factorize(output)

#reserve part of the data
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size = .8)

rfc = RandomForestClassifier(random_state = 15) #this is a seed to maintain same set, geeks use 42
rfc.fit (x_train.values, y_train) #build the forest from x/y

y_pred = rfc.predict(x_test.values) #use the test values to generate an accuracy score
score = accuracy_score(y_pred, y_test)

print("Our accuracy score for this model is {}".format(score))


rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

# print("Here are our output variables")
# print(output.head())
# print("Here are our feature variables")
# print(features.head())