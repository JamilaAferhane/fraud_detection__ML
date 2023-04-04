"""
Created on Fri Mar 17 12:54:12 2023
@author: Jamila Aferhane
"""
#This code trains different classification models on a banking transactions data. 
#Here are the used techniques: DNN classifier


# Import necessary libraries

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.metrics import recall_score, roc_curve, roc_auc_score
import tensorflow as tf


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




#Deep Learning - DNN

    # 1- Feature columns
feat_cols = []
for col in data.columns[:-1]:
    feat_cols.append(tf.feature_column.numeric_column(col))

    # 2- Input function for training
train_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=x_train,
    y=y_train,
    batch_size=30,
    num_epochs=2000,
    shuffle=True
)

    # 3 - Input function for evaluation
eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=x_test,
    y=y_test,
    batch_size=20,
    num_epochs=1,
    shuffle=False
)

    # 4 -  DNNClassifier model
model = tf.estimator.DNNClassifier(
    hidden_units=[512, 256, 128],
    feature_columns=feat_cols,
    n_classes=2
)

    # 5 - Train the model
model.train(input_fn=train_input_func, steps=5000)

    # 7 - Evaluate the model
results = model.evaluate(input_fn=eval_input_func)
print(results)

    # 8 - Predictions
    
        #Input function for prediction
pred_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=x_test,
    batch_size=20,
    num_epochs=1,
    shuffle=False
)
        #Prediction
predictions = list(model.predict(input_fn=pred_input_func))
prediction = [p['class_ids'][0] for p in predictions]

    # 9 - Evaluation
        #Accuracy score
DNN_accuracy_score = accuracy_score(y_test, prediction)
print('\nThe accuracy score of DNN is:')
print(DNN_accuracy_score)
        #Precision
DNN_precision = precision_score(y_test, prediction)
print('\nThe precision of DNN is:')
print(DNN_precision)
        #Recall
DNN_recall = recall_score(y_test, prediction)
print('\nThe recall of DNN is:')
print(DNN_recall)
        #F1-score
DNN_f1 = f1_score(y_test, prediction)
print('\nThe F1_score of DNN is:')
print(DNN_f1)
        #Confusion matrix
DNN_cm = confusion_matrix(y_test, prediction)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(DNN_cm)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted negative', 'Predicted positive'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual negative', 'Actual positive'))
for i in range(2):
   for j in range(2):
       ax.text(j, i, DNN_cm[i, j], ha='center', va='center', color='white', fontsize=20)
ax.set_title("DNN Confusion matrix", fontsize=20)
plt.show()
        #ROC curve
fpr,tpr,thresholds=roc_curve(y_test,prediction)
plt.plot(fpr,tpr,label='DNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DNN ROC curve')
plt.legend()
plt.show()
        #ROC AUC score
DNN_auc = roc_auc_score(y_test, prediction)
print("\nThe AUC of DNN is:")
print(DNN_auc)
