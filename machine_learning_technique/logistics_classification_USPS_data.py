
#    logistics   ######################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

myFile = np.genfromtxt('alp.csv', delimiter=',')

mydata = myFile[:,0:256]
mylabel = myFile[:,256]
train_prop = [0.1, 0.2, 0.5, 0.8, 0.9]

logisticRegr = LogisticRegression()
accuracies = []
for j in range(len(train_prop)):
    X_train, X_test, y_train, y_test = train_test_split(mydata, mylabel, train_size=train_prop[j])

    # train the k-Nearest Neighbor classifier with the current value of `k`
    #repeat 100 times
    n=3
    first_score = []
    first_training_error = []
    while n>0:
        logisticRegr.fit(X_train, y_train)
        first_training_error.append(logisticRegr.fit(X_train, y_train).score(X_train, y_train))
        prediction =logisticRegr.predict(X_test)

        # Use score method to get accuracy of model
        first_score.append(logisticRegr.score(X_test, y_test))
        score = sum(first_score) / len(first_score)
        train_score = sum(first_training_error) / len(first_training_error)
        n -= 1

    print("training_error=%.3f, testing_error=%.3f, train_test_prop= %.2f" % (1-train_score, 1 - score, train_prop[j]))
    accuracies.append(score)


# find the value of k that has the largest accuracy
i = int(np.argmin(accuracies))
#print("highest testing accuracy is " + accuracies[i])