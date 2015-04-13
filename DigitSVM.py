import numpy as np
from sklearn import svm, metrics
from scipy import stats

train = np.genfromtxt('train.csv', skip_header=1, dtype = int, delimiter=',')
test = np.genfromtxt('test.csv', skip_header=1, dtype = int, delimiter=',')

x = train[:,1:]
Y = train[:,0]

Z = stats.zscore(test, axis=1, ddof=1)

X = stats.zscore(x, axis=1, ddof=1)

n_samples = len(X[1])

clf = svm.SVC(kernel = 'poly', degree = 4, gamma=0.001, coef0 = 0.1)
clf.fit(X,Y)

predicted = clf.predict(Z)

out = open("output.txt", "w")
print >> out, "\n".join(str(i) for i in predicted)
out.close()
