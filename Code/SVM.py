"""
Created on Fri Mar 17 12:54:12 2023
@author: Jamila Aferhane
"""
#This code trains different classification models on a banking transactions data. 
#Here are the used technique: SVM


# Import necessary libraries

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.metrics import recall_score, roc_curve, roc_auc_score


# Import Data

#Our Data is about banking transactions collected using credit cards.
data = pd.read_csv('creditcard.csv')
print(data.head())

#Count the number of data points for each class

class_counts = data['Class'].value_counts()
print("Class 0 count", class_counts[0])
print("Class 1 count", class_counts[1])

# Organize Data

#Store the input in x
x = data.drop(data.columns[-1], axis=1)
#Store the ouput in y (the 'Class' column: 1 for fraudulent transaction, 0 for non fraudulent transction)
y = data['Class']
#Split data
#the random_state argument is for controlling the randomization
#we give 30% of data to the testing set
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#SVM (Support Vector Machines)

    #1 - Select and train model
#Here we keep the linear kernel since it gives a good performance
#We suppose that the data is strictely collected,
#for this we consider a high value of C which means the model will try to classify every point correctely
SVM_model = svm.SVC(C=100, kernel='linear')
SVM_model.fit(x_train, y_train)

    # 2 - Prediction
test_pred_svm = SVM_model.predict(x_test)

    # 3 - Evaluation
        #Accuracy score
accuracy_score_svm = accuracy_score(y_test,test_pred_svm)
print('The accuracy score of SVM is:')
print(accuracy_score_svm)

        #Precision
SVM_precision = precision_score(y_test, test_pred_svm)
print('\nThe precision of SVM is:')
print(SVM_precision)

        #Recall
SVM_recall = recall_score(y_test, test_pred_svm)
print('\nThe recall of SVM is:')
print(SVM_recall)

         #F1-score
SVM_f1 = f1_score(y_test, test_pred_svm)
print('\nThe F1_score of SVM is:')
print(SVM_f1)
        #Confusion matrix
svm_cm = confusion_matrix(y_test, test_pred_svm)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(svm_cm)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted negative', 'Predicted positive'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual negative', 'Actual positive'))
for i in range(2):
   for j in range(2):
       ax.text(j, i, svm_cm[i, j], ha='center', va='center', color='white', fontsize=20)
ax.set_title("SVM Confusion matrix", fontsize=20)
plt.show()
        #ROC curve
fpr,tpr,thresholds=roc_curve(y_test,test_pred_svm)
plt.plot(fpr,tpr,label='SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC curve')
plt.legend()
plt.show()
         #ROC AUC score
SVM_auc = roc_auc_score(y_test, test_pred_svm)
print("\nThe AUC of KNN is:")
print(SVM_auc)

