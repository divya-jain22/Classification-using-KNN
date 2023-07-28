# Importing all the required libraries
import pandas as pd
dataset=pd.read_csv('BGdataset.csv')
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
dataset.head(5)
# Making an object of LabelEncoder 
ok=LabelEncoder()
dataset["gender"].unique()
dataset["gender"]=ok.fit_transform(dataset["gender"])
dataset["hair"]=ok.fit_transform(dataset["hair"])
dataset["fair"]=ok.fit_transform(dataset["fair"])
dataset
# Splitting the Training Data in the Features and the Target data
x=dataset[["hair"]+["fair"]]
y=dataset["gender"]
# Making an object of knn classifier 
model=KNeighborsClassifier()
# Model fitting onto the given training dataset
model.fit(x,y)
# Taking random test data to test the model
test=[[0,1]]
# Predicting the target value for the given random test data
model.predict(test)
