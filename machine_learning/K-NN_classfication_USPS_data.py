from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

myFile = np.genfromtxt('alp.csv', delimiter=',')

mydata = myFile[:,0:256]
mylabel = myFile[:,256]
train_prop = [0.1, 0.2, 0.5, 0.8, 0.9]
kVals = [5,10,15,30]
accuracies = []
for j in range(len(train_prop)):
    X_train, X_test, y_train, y_test = train_test_split(mydata, mylabel, train_size=train_prop[j])

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(len(kVals)):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        #repeat 100 times
        n=100
        first_score = []
        first_training_error = []
        while n>0:
            model = KNeighborsClassifier(n_neighbors=kVals[k])
            first_training_error.append(model.fit(X_train, y_train).score(X_train, y_train))
            model.fit(X_train, y_train)

            # evaluate the model and update the accuracies list
            first_score.append(model.score(X_test, y_test))
            score = sum(first_score)/len(first_score)
            train_score = sum(first_training_error) / len(first_training_error)
            n -= 1

        print("k=%d, training_error=%.3f, testing_error=%.3f, train_test_prop= %.2f" % (kVals[k], 1-train_score, 1 - score, train_prop[j]))
        accuracies.append(score)


# find the value of k that has the largest accuracy
i = int(np.argmin(accuracies))
#print("highest testing accuracy is " + accuracies[i])