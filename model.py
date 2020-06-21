# Himanshu Tripathi

# import necessary libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

dataset=pd.read_csv("diabetes.csv")

print(dataset.columns)

X=dataset[['Age','BMI','BloodPressure','Glucose']]
Y=dataset[['Outcome']]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=30)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model=lr.fit(xtrain,ytrain)

pickle.dump(model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

pred=model.predict(xtest)