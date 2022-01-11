import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import time
from sklearn.metrics import accuracy_score,  f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
import random
import time
import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import collections
import warnings
import pickle
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

######## PCA #############

# with open('xtrain.pickle', 'rb') as handle:
#     X_train = pickle.load(handle)
# with open('xtest.pickle', 'rb') as handle:
#     X_test = pickle.load(handle)
# with open('ytrain.pickle', 'rb') as handle:
#     y_train = pickle.load(handle)
# with open('ytest.pickle', 'rb') as handle:
#     y_test = pickle.load(handle)

# pca = PCA(n_components=100)
# Xsp_train = pca.fit_transform(X_train)
# Xsp_test = pca.fit_transform(X_test)

# X = torch.from_numpy(Xsp_train).cuda()
# y = torch.from_numpy(np.array(y_train)).cuda()
# X_val = torch.from_numpy(Xsp_test.astype('float32')).cuda()
# y_val = torch.from_numpy(np.array(y_test).astype('float32')).cuda()
# print(X.shape)
######## NO PCA ##########

# with open('xtrain.pickle', 'rb') as handle:
#     X_train = pickle.load(handle)
# with open('xtest.pickle', 'rb') as handle:
#     X_test = pickle.load(handle)
# with open('ytrain.pickle', 'rb') as handle:
#     y_train = pickle.load(handle)
# with open('ytest.pickle', 'rb') as handle:
#     y_test = pickle.load(handle)

# X = torch.from_numpy(X_train).cuda()
# y = torch.from_numpy(np.array(y_train)).cuda()
# X_val = torch.from_numpy(X_test.astype('float32')).cuda()
# y_val = torch.from_numpy(np.array(y_test).astype('float32')).cuda()


########### VAE ###############

with open('xtrainv.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('xtestv.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open('ytrainv.pickle', 'rb') as handle:
    y_train = pickle.load(handle)
with open('ytestv.pickle', 'rb') as handle:
    y_test = pickle.load(handle)

X = torch.from_numpy(X_train.to_numpy()).cuda()
y = torch.from_numpy(np.array(y_train)).cuda()
X_val = torch.from_numpy(X_test.to_numpy().astype('float32')).cuda()
y_val = torch.from_numpy(np.array(y_test).astype('float32')).cuda()



###############################

class Data(Dataset):
    def __init__(self):
        self.x=X
        self.y=y
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

data_set=Data()
trainloader=DataLoader(dataset=data_set,batch_size=64)

class Net(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.linear2=nn.Linear(H,D_out)

        
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))  
        x=self.linear2(x)
        return x

input_dim=100    # how many Variables are in the dataset
hidden_dim = 50 # hidden layers
output_dim=12    # number of classes

model=Net(input_dim,hidden_dim,output_dim).cuda()

criterion=nn.CrossEntropyLoss()
criterion = criterion.cuda()


learning_rate=0.1
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs=10
loss_list=[]

#n_epochs
for epoch in tqdm(range(n_epochs)):
    for x, y in trainloader:
        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z=model(x.float())
        loss=criterion(z,y.type(torch.LongTensor).cuda())
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()
        
        loss_list.append(loss.data)
        
        yhat = model(X_val)
        print('epoch {}, loss {}, Train Accuracy {}, Test Accuracy {}'.format(epoch, loss.item(),torch.sum(torch.argmax(z,dim=1)==y)/x.shape[0],torch.sum(torch.argmax(yhat,dim=1)==y_val)/X_test.shape[0]))

from sklearn.metrics import f1_score

y_true = y_val.cpu().tolist()
y_pred = torch.argmax(yhat,dim=1).cpu().tolist()
print(f1_score(y_true, y_pred, average='macro'))
print(f1_score(y_true, y_pred, average='micro'))
cf_matrix = confusion_matrix(y_true, y_pred)
print(cf_matrix)

import seaborn as sns
fig, ax = plt.subplots(figsize=(10,10))
sns_plot = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
sns_plot.figure.savefig("output.png")
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
# sns.heatmap(cf_matrix, annot=True)