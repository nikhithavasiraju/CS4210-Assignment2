#-------------------------------------------------------------------------
# AUTHOR: Nikhitha Vasiraju
# FILENAME: knn.py
# SPECIFICATION: LOO-CV using 1NN for email classification
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

wrong = 0
total = len(db)

#Loop your data to allow each instance to be your test set

for test_idx, i in enumerate(db):
 
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    X = []
    for train_idx, row in enumerate(db):
        if train_idx == test_idx:
            continue
        X.append([float(v) for v in row[0:20]])

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    Y = []
    for train_idx, row in enumerate(db):
        if train_idx == test_idx:
            continue
        Y.append(1 if row[20] == 'ham' else 2)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    testSample = [float(v) for v in i[0:20]]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here

    clf = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here

    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    true_label = 1 if i[20] == 'ham' else 2
    if class_predicted != true_label:
        wrong += 1

#Print the error rate
#--> add your Python code here

error_rate = wrong / total
print("LOO-CV error rate (1NN):", round(error_rate, 4))





