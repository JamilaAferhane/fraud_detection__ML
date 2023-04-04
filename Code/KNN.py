"""
Created on Fri Mar 17 12:54:12 2023
@author: Jamila Aferhane
"""
#This code trains different classification models on a banking transactions data. 
#Here are the used technique: KNN


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

#KNN (K Nearest Neighbours) 

    # 1 - Select and train the model
KNN_model = KNeighborsClassifier(n_neighbors= 4)
KNN_model.fit(x_train, y_train)

    # 2 - Prediction
test_pred_knn = KNN_model.predict(x_test)

    # 3 - Evaluation
    
        #Accuracy score
KNN_accuracy_score = accuracy_score(y_test, test_pred_knn)
print('The accuracy score of KNN for 1 neighbors is:')
print(0.9980805917395222)
print('The accuracy score of KNN for 3 neighbors is :')
print(0.9983263696265346)
print('The accuracy score of KNN for 4 neighbors is :')
print(KNN_accuracy_score)
print('The accuracy score of KNN for 5 neighbors is :')
print(0.99833807333544)
print('The accuracy score of KNN for 6 neighbors is :')
print(0.9983263696265346)
print('The accuracy score of KNN for 10 neighbors is :')
print(0.9983029622087239)

        #Precision
KNN_precision = precision_score(y_test, test_pred_knn)
print('\nThe precision of KNN is:')
print(KNN_precision)

        #Recall
KNN_recall = recall_score(y_test, test_pred_knn)
print('\nThe recall of KNN is:')
print(KNN_recall)

         #F1-score
KNN_f1 = f1_score(y_test, test_pred_knn)
print('\nThe F1_score of KNN is:')
print(KNN_f1)

         #Confusion matrix
KNN_cm = confusion_matrix(y_test, test_pred_knn)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(KNN_cm)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted negative', 'Predicted positive'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual negative', 'Actual positive'))
for i in range(2):
   for j in range(2):
       ax.text(j, i, KNN_cm[i, j], ha='center', va='center', color='white', fontsize=20)
ax.set_title("KNN Confusion matrix", fontsize=20)
plt.show()
         #ROC curve
fpr,tpr,thresholds=roc_curve(y_test,test_pred_knn)
plt.plot(fpr,tpr,label='KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC curve')
plt.legend()
plt.show()
         #ROC AUC score
KNN_auc = roc_auc_score(y_test, test_pred_knn)
print("\nThe AUC of KNN is:")
print(KNN_auc)

