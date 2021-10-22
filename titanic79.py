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
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

########## Calibrate Train Data ##########
##########################################

# Separate predictor and labels from prepared_train_data
train_data_predictors = train_data.drop("Survived", axis=1)
train_data_labels = train_data["Survived"].copy()

# Create new categories
#train_data_predictors["family_size"] = train_data_predictors["SibSp"] + train_data_predictors["Parch"]

train_data_predictors['Title'] = train_data_predictors['Name'] \
        .str.extract(', ([A-Za-z]+)\.', expand=False)

# Convert to categorical values Title 
train_data_predictors["Title"] = train_data_predictors["Title"].replace(['Lady', 'the Countess',
                                             'Capt', 'Col','Don', 'Dr', 
                                             'Major', 'Rev', 'Sir', 'Jonkheer',
                                             'Dona'], 'Rare')

train_data_predictors["Title"] = train_data_predictors["Title"].map({"Master":0, "Miss":1, "Ms" : 1 ,
                                         "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, 
                                         "Rare":3})

#print(train_data_predictors)

# Use SimpleImputer to fill missing data
imputer.fit(train_data_predictors)
x = imputer.transform(train_data_predictors)
train_data_predictors_transformed = pd.DataFrame(x, columns=train_data_predictors.columns, index=train_data_predictors.index)

# Use OneHotEncoder to encode alpha data, and numerical with more than three possibilities
train_data_predictors_encoded = pd.DataFrame(encoder.fit_transform(train_data_predictors_transformed[["Pclass", "Sex", "Embarked", "Title"]]))
train_data_predictors_transformed_encoded = train_data_predictors_transformed.join(train_data_predictors_encoded)

# Create new categories
train_data_predictors_transformed_encoded["family_size"] = train_data_predictors_transformed_encoded["SibSp"] + train_data_predictors_transformed_encoded["Parch"]

# Drop irrelevant categories
train_data_predictors_transformed_encoded = train_data_predictors_transformed_encoded.drop(["Pclass", "Sex", "Embarked", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Title"], axis=1)

# Scale the data
prepared_train_data_predictors = scaler.fit_transform(train_data_predictors_transformed_encoded.astype(np.float64))

########## Calibrate Test Data ##########
##########################################

# Create new categories
#test_data["family_size"] = test_data["SibSp"] + test_data["Parch"]

# Convert to categorical values Title 
test_data['Title'] = test_data['Name'] \
        .str.extract(', ([A-Za-z]+)\.', expand=False)

test_data["Title"] = test_data["Title"].replace(['Lady', 'the Countess',
                                             'Capt', 'Col','Don', 'Dr', 
                                             'Major', 'Rev', 'Sir', 'Jonkheer',
                                             'Dona'], 'Rare')

test_data["Title"] = test_data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 ,
                                         "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, 
                                         "Rare":3})

#print(test_data)

# Use SimpleImputer to fill missing data
x = imputer.transform(test_data)
test_data_transformed = pd.DataFrame(x, columns=test_data.columns, index=test_data.index)

# Use OneHotEncoder to encode alpha data, and numerical with more than three possibilities
test_data_encoded = pd.DataFrame(encoder.fit_transform(test_data_transformed[["Pclass", "Sex", "Embarked", "Title"]]))
test_data_transformed_encoded = test_data_transformed.join(test_data_encoded)

# Create new categories
test_data_transformed_encoded["family_size"] = test_data_transformed_encoded["SibSp"] + test_data_transformed_encoded["Parch"]

# Drop irrelevant categories
test_data_transformed_encoded = test_data_transformed_encoded.drop(["Pclass", "Sex", "Embarked", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Title"], axis=1)

# Scale the data
prepared_test_data = scaler.fit_transform(test_data_transformed_encoded.astype(np.float64))

########## FINAL MODEL ##########
#################################
# Fine Tune using GridSearch 
param_grid = [{'n_estimators': [10, 30, 60, 100, 200], 'max_features': [1, 2, 4, 8, 10, 14]},]
grid_search = GridSearchCV(forest_clf, param_grid, cv=3)
grid_search.fit(prepared_train_data_predictors, train_data_labels)

final_model = grid_search.best_estimator_

########## Display accuracy scores ##########
#############################################
print("""
Grid Search - best parameters
""", grid_search.best_params_)

forest_scores = cross_val_score(forest_clf, prepared_train_data_predictors, train_data_labels, cv=10, scoring="accuracy")
print("""
Mean Cross Validation Score - Train Data
""", forest_scores.mean())

########## Use the final model to predict Survived ##########
#############################################################

survived_prediction = final_model.predict(prepared_test_data) # Perform the prediction

########## Save results ##########
##################################
survived_prediction = pd.DataFrame(survived_prediction)
survived_prediction.to_csv('file.csv') 

print ("""
My program took""", time.time() - start_time, "to run")

