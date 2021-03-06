import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# import data
dataset = pd.read_excel('ENB2012_data.xlsx', sheet_name='Sheet1')
X = dataset.iloc[:, :-2].values
Y1 = dataset.iloc[:, 8]
Y2 = dataset.iloc[:, 9]
# split data into train and test set

X_intermediate, X_test, Y1_intermediate, Y1_test, Y2_intermediate, Y2_test = train_test_split(X, Y1, Y2, test_size=0.2,
                                                                                              random_state=0)
X_train, X_validation, Y1_train, Y1_validation, Y2_train, Y2_validation = train_test_split(X_intermediate,
                                                                                           Y1_intermediate,
                                                                                           Y2_intermediate,
                                                                                           test_size=0.25,
                                                                                           random_state=0)
del X_intermediate, Y1_intermediate, Y2_intermediate

# print proportions
print('Y1: train: {}% | validation: {}% | test {}%'.format(round(len(Y1_train)/len(Y1), 2),
                                                       round(len(Y1_validation)/len(Y1), 2),
                                                       round(len(Y1_test)/len(Y1), 2)))
print('Y2: train: {}% | validation: {}% | test {}%'.format(round(len(Y1_train)/len(Y2), 2),
                                                       round(len(Y1_validation)/len(Y2), 2),
                                                       round(len(Y1_test)/len(Y2), 2)))


print(X_train.shape, Y1_train.shape)
print(X_test.shape, Y1_test.shape)
print(X_train.shape, Y2_train.shape)
print(X_test.shape, Y2_test.shape)

# Standard Scaling
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting to training set
fitForY1 = LinearRegression()
fitForY2 = LinearRegression()

fitForY1.fit(X_train, Y1_train)
print('The linear cofficients for Y1', fitForY1.coef_)

fitForY2.fit(X_train, Y2_train)
print('The linear cofficients', fitForY2.coef_)

# test predictions
y1_pred = fitForY1.predict(X_test)
y2_pred = fitForY2.predict(X_test)

# plot of Y1 vs Y1 prediction
plt.scatter(Y1_test, y1_pred, color='blue')
plt.title('Y1 test vs Y1 Predictions')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.grid(True)
plt.show()

# plot of Y2 vs Y2 prediction
plt.scatter(Y2_test, y2_pred, color='red')
plt.title('Y2 test vs Y2 Predictions')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.grid(True)
plt.show()

# Accuracy for Y1
confidenceY1 = fitForY1.score(X_test, Y1_test)
print('This is the prediction accuracy for Y1', confidenceY1)
plt.title('Actual Y1 vs. Y1_Predict', size=10)
plt.scatter(Y1_test, y1_pred, color='c', marker='.')
plt.xlabel('Actual Y1', size=10)
plt.ylabel('Y1 Predict', size=10)
plt.grid(True)
plt.show()

# Accuracy for Y2
confidenceY2 = fitForY2.score(X_test, Y1_test)
print('This is the prediction accuracy for Y2', confidenceY2)
plt.title('Actual Y2 vs. Y2_Predict', size=10)
plt.scatter(Y2_test, y2_pred, color='c', marker='.')
plt.xlabel('Actual Y2', size=10)
plt.ylabel('Y2 Predict', size=10)
plt.grid(True)
plt.show()

# Cross-Validation for Y1
scoresY1 = cross_val_score(fitForY1, X, Y1, cv=10)
print('Cross Validated Scores', scoresY1)
# kfold
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, validation_index in kf.split(X, Y1):

    X_train, X_validation = X[train_index], X[validation_index]
    Y1_train, Y1_validation = Y1[train_index], Y1[validation_index]

predictions2 = cross_val_predict(fitForY1, X, Y1, cv=10)
accuracy = metrics.r2_score(Y1, predictions2)
print('This is R2 for Y1', accuracy)
# Graph for Cross Validation
plt.scatter(Y1, predictions2, color='orange', marker='.')
plt.xlabel('Actual Y1', size=10)
plt.ylabel('Y1_Predict', size=10)
plt.title('Actual and Predicted Y1 Values using 10 Fold Cross Validation', size=10)
plt.show()

# Cross-Validation for Y2
scoresY2 = cross_val_score(fitForY2, X, Y2, cv=10)
print('Cross Validated Scores', scoresY2)
# k fold
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, validation_index in kf.split(X, Y2):

    X_train, X_validation = X[train_index], X[validation_index]
    Y2_train, Y2_validation = Y2[train_index], Y2[validation_index]

predictions2 = cross_val_predict(fitForY2, X, Y2, cv=10)
accuracy = metrics.r2_score(Y2, predictions2)

# Graph for Cross Validation 
print('This is R2 for Y2', accuracy)
plt.scatter(Y2, predictions2, color='green', marker='.')
plt.xlabel('Actual Y2', size=10)
plt.ylabel('Y2_Predict', size=10)
plt.title('Actual and Predicted Y2 Values using 10 Fold Cross Validation', size=10)
plt.show()
