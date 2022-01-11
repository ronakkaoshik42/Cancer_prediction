import pandas as pd
import gzip
import numpy as np
# a = pd.read_csv('k_100_compress_for_classifier.tsv', header=0, sep='\t')
# print(a.head())

with open('k_100_compress_for_classifier.tsv', 'rb') as fd:
    gzip_fd = gzip.GzipFile(fileobj=fd)
    a = pd.read_csv(gzip_fd, header=0, sep='\t')

X = a.iloc[:,1:101]
y = a.iloc[:,101]

d={}
c=0
count = {}
ly = []
for i in range(y.shape[0]):
    if not y[i] in d:
        d[y[i]] = c
        count[y[i]]=0
        c+=1
    else:
        count[y[i]]+=1
    ly.append(d[y[i]])


msk = np.random.rand(len(X)) < 0.9

X_train = X[msk]
X_test = X[~msk]

y_train = y[msk]
y_test = y[~msk]

l = []
for i in count:
    if count[i]>0:
        l.append(i)

rm_ind = []
y_new = []

for i in range(len(X)):
    if not y[i] in l:
        rm_ind.append(i)
    else:
        y_new.append(l.index(y[i]))


X_new = X.drop(rm_ind)



# d={}
# c=0
# count = {}
# ly_train = []
# for i in range(y_train.shape[0]):
#     if not y_train[i] in d:
#         d[y_train[i]] = c
#         count[y_train[i]]=0
#         c+=1
#     else:
#         count[y_train[i]]+=1
#     ly_train.append(d[y_train[i]])
# ly_test = []

# # for i in range(1106):
# #     ly_test.append(d[y_test[i]])



# print(len(X))

