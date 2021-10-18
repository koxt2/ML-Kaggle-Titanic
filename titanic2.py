import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

start_time = time.time()

########## SETUP ##########
###########################
# Make the data run the same every time it's run 
np.random.seed(42)

imputer = SimpleImputer(strategy="most_frequent")
encoder = OneHotEncoder(sparse=False)
scaler = StandardScaler()
forest_clf = RandomForestClassifier(random_state=42)

########## Load Data ##########
###############################
def load_train_data():
    return pd.read_csv('/Users/richard/Documents/Python/Machine-Learning/titanic/train.csv')
train_data = load_train_data()

def load_test_data():
    return pd.read_csv('/Users/richard/Documents/Python/Machine-Learning/titanic/test.csv')
test_data = load_test_data()

########## Calibrate Train Data ##########
##########################################

# Separate predictor and labels from prepared_train_data
train_data_predictors = train_data.drop("Survived", axis=1)
train_data_labels = train_data["Survived"].copy()

# Create new categories
#train_data_predictors["family_size"] = train_data_predictors["SibSp"] + train_data_predictors["Parch"]

# Drop irrelevant categories
train_data_predictors = train_data_predictors.drop(["Name", "Ticket", "Cabin"], axis=1)

# Use SimpleImputer to fill missing data
imputer.fit(train_data_predictors)
x = imputer.transform(train_data_predictors)
train_data_predictors_transformed = pd.DataFrame(x, columns=train_data_predictors.columns, index=train_data_predictors.index)

# Use OneHotEncoder to encode alpha data, and numerical with more than three possibilities
train_data_predictors_encoded = pd.DataFrame(encoder.fit_transform(train_data_predictors_transformed[["Pclass", "Sex", "Embarked"]]))
train_data_predictors_transformed_encoded = train_data_predictors_transformed.join(train_data_predictors_encoded)
train_data_predictors_transformed_encoded = train_data_predictors_transformed_encoded.drop(["Pclass", "Sex", "Embarked"], axis=1)

# Create new categories
train_data_predictors_transformed_encoded["family_size"] = train_data_predictors_transformed_encoded["SibSp"] + train_data_predictors_transformed_encoded["Parch"]
train_data_predictors_transformed_encoded = train_data_predictors_transformed_encoded.drop(["SibSp", "Parch"], axis=1)

# Scale the data
prepared_train_data_predictors = scaler.fit_transform(train_data_predictors_transformed_encoded.astype(np.float64))

########## Calibrate Test Data ##########
##########################################

# Create new categories
#test_data["family_size"] = test_data["SibSp"] + test_data["Parch"]

# Drop irrelevant categories
test_data = test_data.drop(["Name", "Ticket", "Cabin"], axis=1)

# Use SimpleImputer to fill missing data
x = imputer.transform(test_data)
test_data_transformed = pd.DataFrame(x, columns=test_data.columns, index=test_data.index)

# Use OneHotEncoder to encode alpha data, and numerical with more than three possibilities
test_data_encoded = pd.DataFrame(encoder.fit_transform(test_data_transformed[["Pclass", "Sex", "Embarked"]]))
test_data_transformed_encoded = test_data_transformed.join(test_data_encoded)
test_data_transformed_encoded = test_data_transformed_encoded.drop(["Pclass", "Sex", "Embarked"], axis=1)

# Create new categories
test_data_transformed_encoded["family_size"] = test_data_transformed_encoded["SibSp"] + test_data_transformed_encoded["Parch"]
test_data_transformed_encoded = test_data_transformed_encoded.drop(["SibSp", "Parch"], axis=1)

# Scale the data
prepared_test_data = scaler.fit_transform(test_data_transformed_encoded.astype(np.float64))

########## FINAL MODEL ##########
#################################
# Fine Tune using GridSearch 
param_grid = [{'n_estimators': [10, 30, 60], 'max_features': [1, 2, 4, 8, 14]},]
grid_search = GridSearchCV(forest_clf, param_grid, cv=3)
grid_search.fit(prepared_train_data_predictors, train_data_labels)

final_model = grid_search.best_estimator_

#############################################################
########## Use the final model to predict Survived ##########
#############################################################

survived_prediction = final_model.predict(prepared_test_data) # Perform the prediction


print("""
Grid Search - best parameters
""", grid_search.best_params_)

print("""
Cross Validation Score - Train Data
""", cross_val_score(forest_clf, prepared_train_data_predictors, train_data_labels, cv=3, scoring="accuracy"))

print("""
Cross Validation Score - Test Data
""", cross_val_score(forest_clf, prepared_test_data, survived_prediction, cv=3, scoring="accuracy"))

survived_prediction = pd.DataFrame(survived_prediction)
survived_prediction.to_csv('file.csv') 

print ("""
My program took""", time.time() - start_time, "to run")

