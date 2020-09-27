import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)




data = pd.read_csv('E:projectKNN\heart.csv')
data.head()



#count how many have disease or not 1=yes 0=no
data.target.value_counts()



#visualize result
sns.countplot(x="target", data=data, palette="bwr")
plt.show()


#male female visualize 
sns.countplot(x='sex', data=data, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


#relation between “Maximum Heart Rate” and “Age”
plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="green")
plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)], c = 'red')
plt.legend(["Not Disease", "Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()



#label dataset with X(matrix of independent variables) and y(vector of the dependent variable)
X = data.iloc[:,:-1].values
y = data.iloc[:,13].values



# here i have used  min Max scaler rather than standardScaler
from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
x=minmax.fit(X).transform(X)
x
# ok

#split data into 75% for train 25% for test
X_train, X_test, y_train, y_test =  train_test_split(x,y,test_size = 0.25, random_state=0)
#print(y_test)   testing for showing reson behind using random state



# # normalize or feature scaling
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# X_train



# finding maximum and minimum value of each feature
# i just find the 'chol' feature max and min for understanding how can we find out the max and min for each features .
maximum=(data.loc[data['chol'].idxmax()])['chol']
minimum=(data.loc[data['chol'].idxmin()])['chol']
print(maximum,minimum)




# scaling range is 0 to 1
# low=0
# high=1
# scaled_x = (x-min(x)/max(x)-min(x))*((high-low)+low)


max_x={}
min_x={}
features=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
for i in range(len(features)):
    max_x[features[i]]=data.loc[data[features[i]].idxmax()][features[i]]
    min_x[features[i]]=data.loc[data[features[i]].idxmin()][features[i]]
print(max_x)
print(min_x)


# ok


features=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

def scaled_input(raw_input):
    return [((raw_input[i]-min_x[features[i]])/(max_x[features[i]]-min_x[features[i]]))*((1-0)+0) for i in range(len(raw_input))]

# here you have to plug your input_features , and after running this cell it will be scaled and stored in final_input variable.
raw_input=[35,1,0,120,198,0,1,130,1,1.6,1,0,3]

final_input=scaled_input(raw_input)
print(final_input)


#train data with KNN model
#for k=10 accuracy will be
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train,y_train)
#prediction
y_pred = classifier.predict(X_test)
#check accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))



classifier.predict([final_input])        #output prediction of KNN


# ok



#so here we can see for k=10 we find the most accuracy of 88% 
#now use confusion matrix for testing how many records tested corecctly
#confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix for KNN:\n%s" % confusion_matrix)
print(classification_report(y_pred,y_test))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

clf.predict([[35,1,0,120,198,0,1,130,1,1.6,1,0,3]])



# Model Accuracy, how often is the classifier correct?
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))



from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n%s" % confusion_matrix)
print(classification_report(y_pred,y_test))



data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']
from sklearn.model_selection import train_test_split
clfr=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfr.fit(X_train,y_train)

# prediction on test set
y_pred=clfr.predict(X_test)

clfr.predict([[35,1,0,120,198,0,1,130,1,1.6,1,0,3]])





#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:{:.2f}".format(accuracy))





from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix of random forest:\n%s" % confusion_matrix)
print(classification_report(y_pred,y_test))



#adaboost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # 70% training and 30% test

# ok 218 adboost


from sklearn.ensemble import AdaBoostClassifier
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=20, learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
print('prediction result of ADAboost')
model.predict([[35,1,0,120,198,0,1,130,1,1.6,1,0,3]])




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
y_pred = model.predict(X_test)
print("Accuracy of adaboost:",metrics.accuracy_score(y_test, y_pred))
print()
print('Accuracy for training set for Random Forest = {}'.format((cm[0][0] + cm[1][1])/len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm[0][0] + cm[1][1])/len(y_test)))





from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix ada boost:\n%s" % confusion_matrix)
print(classification_report(y_pred,y_test))


from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)


y_pred=lr.predict(X_test)
lr.predict([[35,1,0,126,282,0,0,156,1,0,2,0,2]])


from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix logistic regresion:\n%s" % confusion_matrix)
print(classification_report(y_pred,y_test))



print("Accuracy of LR:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)



from sklearn import svm
#Create a svm Classifier
clfs = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clfs.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clfs.predict(X_test)
clfs.predict([[35,1,0,126,282,0,0,156,1,0,2,0,2]])



#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))




from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix SVM:\n%s" % confusion_matrix)
print(classification_report(y_pred,y_test))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)


print()
print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


classifier.predict([[35,1,0,120,198,0,1,130,1,1.6,1,0,3]])



from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)



from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)


y_pred_train = xg.predict(X_train)



for i in range(0, len(y_pred_train)):
    if y_pred_train[i]>= 0.5:       # setting threshold to .5
       y_pred_train[i]=1
    else:  
       y_pred_train[i]=0



cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

xg.predict([[35,1,0,120,198,0,1,130,1,1.6,1,0,3]])

print("Confusion matrix XGboost:\n%s" % confusion_matrix)