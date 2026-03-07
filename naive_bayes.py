#-------------------------------------------------------------------------
# AUTHOR: Nikhitha Vasiraju
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes classifier for PlayTennis using weather_training.csv
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 65 minutes
# --------------------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

def encode_outlook(v):
    if v == 'Sunny':
        return 1
    elif v == 'Overcast':
        return 2
    else:  
        return 3

def encode_temperature(v):
    if v == 'Hot':
        return 1
    elif v == 'Mild':
        return 2
    else:  
        return 3

def encode_humidity(v):
    if v == 'High':
        return 1
    else: 
        return 2

def encode_wind(v):
    if v == 'Weak':
        return 1
    else:  
        return 2

X = []
for row in dbTraining:
    outlook = encode_outlook(row[1])
    temp = encode_temperature(row[2])
    humidity = encode_humidity(row[3])
    wind = encode_wind(row[4])
    X.append([outlook, temp, humidity, wind])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

Y = []
for row in dbTraining:
    label = 1 if row[5] == 'Yes' else 2
    Y.append(label)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here

clf = GaussianNB(var_smoothing=1e-9)
clf = clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here

print("Day, prob(Yes), prob(No), predicted")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

for row in dbTest:
    day = row[0]

    outlook = encode_outlook(row[1])
    temp = encode_temperature(row[2])
    humidity = encode_humidity(row[3])
    wind = encode_wind(row[4])

    probs = clf.predict_proba([[outlook, temp, humidity, wind]])[0]
    pred = clf.predict([[outlook, temp, humidity, wind]])[0]

    prob_yes = probs[0]
    prob_no = probs[1]
    pred_label = "Yes" if pred == 1 else "No"

    print(f"{day}, {prob_yes:.4f}, {prob_no:.4f}, {pred_label}")


