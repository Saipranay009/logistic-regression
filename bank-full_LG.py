# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:55:57 2022

@author: Sai pranay
"""
#-----------------------IMPORTING_THE_DATA_SET---------------------------------
import pandas as pd
bank=pd.read_csv("E:\DATA_SCIENCE_ASS\LOGISTIC_REGRESSION\\bank-full.csv",sep=';')
print(bank)
list(bank)
bank.shape
bank.head(10)
bank.describe()
bank.value_counts()
bank.info()
bank.dtypes


#-----------------working_on_label_encoding------------------------------------

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
bank["marital_"] = LE.fit_transform(bank["marital"])
bank[["marital", "marital_"]].head(11)
bank.shape
pd.crosstab(bank.marital,bank.marital_)


bank["job_cat"] = LE.fit_transform(bank["job"])
bank[["job","job_cat"]].head(11)
pd.crosstab(bank.job,bank.job_cat)


bank["education_"] = LE.fit_transform(bank["education"])
bank[["education","education_"]].head(11)
pd.crosstab(bank.education,bank.education_)


bank["default_"] = LE.fit_transform(bank["default"])
bank[["default","default_"]].head(11)
pd.crosstab(bank.default,bank.default_)

bank["housing_"] = LE.fit_transform(bank["housing"])
bank[["housing","housing_"]].head(11)
pd.crosstab(bank.housing,bank.housing_)

bank["loan_"] = LE.fit_transform(bank["loan"])
bank[["loan","loan_"]].head(11)
pd.crosstab(bank.loan,bank.loan_)

bank["contact_"] = LE.fit_transform(bank["contact"])
bank[["contact","contact_"]].head(11)
pd.crosstab(bank.contact,bank.contact_)


bank["month_"] = LE.fit_transform(bank["month"])
bank[["month","month_"]].head(11)
pd.crosstab(bank.month,bank.month_)


bank["poutcome_"] = LE.fit_transform(bank["poutcome"])
bank[["poutcome","poutcome_"]].head(11)
pd.crosstab(bank.poutcome,bank.poutcome_)


bank["y_"] = LE.fit_transform(bank["y"])
bank[["y","y_"]].head(11)
pd.crosstab(bank.y,bank.y_)

bank.head(10)
bank.dtypes
list(bank)
bank.shape

#-----------------------------SPLITTING_THE_DATA_SET---------------------------

x = bank.iloc[:,0:26]
x

#--DROPPING_THE_VALUE_WHICH_HAS_DONE_ORIGINAL_COLUMNS_VALUE_OF_LABEL_ENCODING--

x1 = bank.drop(['job','marital', 'education', 'default', 'housing', 'loan', 'contact','month','poutcome','y','y_'],axis = 1)
x1.shape
x1.dtypes

y = bank['y_']
print(y)
y.shape

#--------------------------Model_Fitting---------------------------------------

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x1,y)
LR.intercept_
LR.coef_

#-------------------------------prediction-------------------------------------

Y_Pred = LR.predict(x1)

#-------------by_using_confusion_matrix_finding_accuracy_score-----------------

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

CM = confusion_matrix(y, Y_Pred)
CM

TN = CM[0,0]
FN = CM[1,0]
FP = CM[0,1]
TP = CM[1,1]

print("Accuracy score:",(accuracy_score(y,Y_Pred)*100).round(3))
print("Recall/Senstivity score:",(recall_score(y,Y_Pred)*100).round(3))
print("Precision score:",(precision_score(y,Y_Pred)*100).round(3))

Specificity = TN / (TN + FP)
print("Specificity score:",(Specificity*100).round(3))
print("F1 score:",(f1_score(y,Y_Pred)*100).round(3))

#------------------------------visualization-----------------------------------

import matplotlib.pyplot as plt
plt.matshow(CM)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()




