import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


X = np.array([[0.5, 0.6], [0.7, 0.5], [0.5, 0.5], [0.2, 0.2]])
y = np.array([0, 0, 1, 1])

clf1 = LogisticRegression(solver='lbfgs', max_iter=100, random_state=123)
clf1.fit(X, y)
probas = clf1.predict_proba(X)
y_pred = [0 if pr[0] > pr [1] else 1 for pr in probas]
print("probas", probas)  
accuracy = accuracy_score(y, y_pred)
print("Accuracy (train): %0.1f%% " % (accuracy * 100))