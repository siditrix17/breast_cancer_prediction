import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection	import	train_test_split
import time
from sklearn.naive_bayes import	GaussianNB
from sklearn.metrics	import	accuracy_score





data=load_breast_cancer()
label_names=data['target_names']
labels=data['target']
feature_names=data['feature_names']
features=data['data']


resp=input('Do you want to view the header of attributes: y/n  ')
if (resp=='Y' or resp=='y'):
 print('label_names:',label_names)
 print('labels:',labels[0])
 print('feature_names:',feature_names[0])
 print('features:',features[0])

print('spiltting the data into training and test sets')
time.sleep(1)
train,test,train_labels,test_labels= train_test_split(features,labels,test_size=0.33,random_state=42)


print('----------initailizing model:')
##initializaing model
time.sleep(1)
gb=GaussianNB()

#training
print('----------training:')
time.sleep(1)
model=gb.fit(train,train_labels)

##prediction
print('-----------predicting:')
time.sleep(1)
predic= gb.predict(test)
print('predicted output:',predic)

##evaluating accuracy
time.sleep(1)
print('-----------evaluating accuracy:')
print('Models accuracy:',accuracy_score(test_labels,predic))

