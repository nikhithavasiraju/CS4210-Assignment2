#-------------------------------------------------------------------------
# AUTHOR: Nikhitha Vasiraju
# FILENAME: decision_tree_2.py
# SPECIFICATION: This file creates trees with a maximum depth of 5 and trains, tests, and outputs the performance
#                of the three models created using the three training datasets on the test set. Then, this process
#                is repeated 10 times and the average accuracy as the final classification performance for each 
#                model.
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 60 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    df_training = pd.read_csv(ds)
    for _, row in df_training.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    for data in dbTraining:

        # Age
        if data[0] == 'Young':
            age = 1
        elif data[0] == 'Prepresbyopic':
            age = 2
        else:
            age = 3

        # Spectacle
        if data[1] == 'Myope':
            spectacle = 1
        else:
            spectacle = 2

        # Astigmatism
        if data[2] == 'No':
            astig = 1
        else:
            astig = 2

        # Tear
        if data[3] == 'Reduced':
            tear = 1
        else:
            tear = 2

        X.append([age, spectacle, astig, tear])


    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

        # Class label
        if data[4] == 'Yes':
            Y.append(1)
        else:
            Y.append(2)

    accuracy_sum = 0

    #Loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       correct = 0

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            
            if data[0] == 'Young':
                age = 1
            elif data[0] == 'Prepresbyopic':
                age = 2
            else:
                age = 3

            spectacle = 1 if data[1] == 'Myope' else 2
            astig = 1 if data[2] == 'No' else 2
            tear = 1 if data[3] == 'Reduced' else 2

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here

            class_predicted = clf.predict([[age, spectacle, astig, tear]])[0]
            true_label = 1 if data[4] == 'Yes' else 2

            if class_predicted == true_label:
                correct += 1

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here

       accuracy = correct / len(dbTest)
       accuracy_sum += accuracy

    final_accuracy = accuracy_sum / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    
    print('final accuracy when training on', ds, ':', round(final_accuracy, 3))




