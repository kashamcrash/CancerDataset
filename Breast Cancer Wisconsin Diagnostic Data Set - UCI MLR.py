#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create a model to predict the probability of a tumor being benign/malignant based on the historical medical data 
# Credentials - kasham1991@gmail.com / karan sharma

# The original dataset is on UCI Machine Learning Repository and Kaggle
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

# Attribute Information:

# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)

# Ten real-valued features are computed for each cell nucleus:

# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)

# The mean, standard error and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.

# All feature values are recoded with four significant digits.

# Missing attribute values: none

# Class distribution: 357 benign, 212 malignant


# In[2]:


# Importing the required libraries
# Loading the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("C://Datasets//cancertype.csv")
dataset.head()


# In[3]:


# dataset.info()
# Diagnosis is categorical
# dataset['diagnosis'].value_counts().sum()
# 212 - Malignant, and 357 - Benign; class imbalance is suitable
print(dataset['diagnosis'].value_counts())
print(dataset['diagnosis'].value_counts(normalize = True))


# In[4]:


# Looking at the basic statistics
# What is this Unnameed: 32 column with NaN values?
# Features must be standardized into a common scale
dataset.describe()


# In[5]:


# Removing the Id and Unnamed: 32
dataset.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)


# In[6]:


# Looking for null values
# No null/missing values!
dataset.isnull().sum()


# In[7]:


# Lets do some visualization
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# Diagnosis
plt.figure(figsize = (5, 5))
dataset['diagnosis'].value_counts().plot(kind = 'bar', color = 'skyblue', label ='0: Benign 1 :Malignant')
plt.title("Countplot: Benign vs Malignant")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.legend()
plt.show()


# In[8]:


# Since there are 33 features, let check for multicollinearity
# Let us use heatmap function
plt.figure(figsize = (25, 25))
sns.heatmap(dataset.corr(), annot = True, linewidths = 0.20)


# In[9]:


# The highly corelated pairs are:  
# texture_mean & texture_worst
# area_mean & radius_worst
# perimeter_mean and radius_worst

plt.figure(figsize = (5,5))
sns.scatterplot(x = 'texture_mean', y ='texture_worst', hue = 'diagnosis', data = dataset)
plt.xlabel('texture_mean', fontsize = 10)
plt.ylabel('texture_worst', fontsize = 10)
plt.title('texture_mean vs texture_worst', fontsize = 10)
plt.show()


plt.figure(figsize = (5,5))
sns.scatterplot(x = 'area_mean', y = 'radius_worst', hue = 'diagnosis', data = dataset)
plt.xlabel('area_mean', fontsize = 10)
plt.ylabel('radius_worst', fontsize = 10)
plt.title('area_mean vs radius_worst', fontsize = 10)
plt.show()

plt.figure(figsize = (5,5))
sns.scatterplot(x = 'perimeter_mean', y = 'radius_worst', hue = 'diagnosis', data = dataset)
plt.xlabel('perimeter_mean', fontsize = 10)
plt.ylabel('radius_worst', fontsize = 10)
plt.title('preimeter_mean vs radius_worst', fontsize = 10)
plt.show()


# In[10]:


# Encoding categorical data column diagnosis
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
dataset['diagnosis'] = encode.fit_transform(dataset['diagnosis'])
dataset.head()


# In[11]:


# Model Building
# But first, lets standardize the dataset and import the libraries

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


# In[12]:


# Creating and splitting the dataset
x = dataset.drop('diagnosis', axis = 1)
y = dataset['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[13]:


# Standardizing the data involes moving each datapoint to a distribution of mean = 0 and SD = 1
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)


# In[14]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
y_log_predict = logmodel.predict(x_test)
# y_log_predict
print('Accuracy of Logistic Regression: {}'.format(accuracy_score(y_test, y_log_predict)))
print("Confusion Matrix for Logistic regression\n\n", confusion_matrix(y_test, y_log_predict, labels=[0, 1]))
print(classification_report(y_test, y_log_predict)) 


# In[15]:


# KNN Classifier
knnmodel = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knnmodel.fit(x_train, y_train)
y_knn_predict = knnmodel.predict(x_test)
print('Accuracy of Logistic KNN: {}'.format(accuracy_score(y_test, y_knn_predict)))
print("Confusion Matrix for KNN\n\n", confusion_matrix(y_test, y_knn_predict, labels=[0, 1]))
print(classification_report(y_test, y_knn_predict)) 


# In[16]:


# Xtreme gradient boosting
XGBmodel = XGBClassifier() 
XGBmodel.fit(x_train, y_train)
y_XGB_predict = XGBmodel.predict(x_test)
print('Accuracy of XGB Model: {}'.format(accuracy_score(y_test, y_XGB_predict)))
print("Confusion Matrix for XGB Model\n\n", confusion_matrix(y_test, y_XGB_predict, labels=[0, 1]))
print(classification_report(y_test, y_XGB_predict)) 


# In[17]:


# Gradient Boosting
GBmodel = GradientBoostingClassifier()
GBmodel.fit(x_train, y_train)
y_GB_predict = GBmodel.predict(x_test)
print('Accuracy of Gradient Boosting: {}'.format(accuracy_score(y_test, y_GB_predict)))
print("Confusion Matrix for Gradient Boosting\n\n", confusion_matrix(y_test, y_GB_predict, labels=[0, 1]))
print(classification_report(y_test, y_GB_predict)) 


# In[18]:


# Randomn Forest
rfmodel = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfmodel.fit(x_train, y_train)
y_rfmodel_predict = rfmodel.predict(x_test)
print('Accuracy of Randomn Forest: {}'.format(accuracy_score(y_test, y_rfmodel_predict)))
print("Confusion Matrix for Random Forest\n\n", confusion_matrix(y_test, y_rfmodel_predict, labels=[0, 1]))
print(classification_report(y_test, y_rfmodel_predict)) 


# In[19]:


# Support Vector Machine
supmodel = SVC(probability = True)
supmodel.fit(x_train, y_train)
y_supmodel_predict = supmodel.predict(x_test)
print('Accuracy of SVM: {}'.format(accuracy_score(y_test, y_supmodel_predict)))
print("Confusion Matrix for SVC\n\n", confusion_matrix(y_test, y_supmodel_predict, labels=[0, 1]))
print(classification_report(y_test, y_supmodel_predict)) 


# In[20]:


# Thank You :) 

