#Self Organising Map

#Importing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting dataset
dataset=pd.read_csv('Credit_Card_Applications.csv')
X=dataset.iloc[:,:15].values
Y=dataset.iloc[:,15].values    #this is only used for checking afterwards 
# only X will be used , as its unsupervised DL

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

#Training the SOM
from minisom import MiniSom
SOM=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
SOM.random_weights_init(X)
SOM.train_random(X,num_iteration=100)

#Visualisation
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(SOM.distance_map().T)
colorbar()
markers=['o','v']  #a approcved , o not approved
colors=['r','g']
for i,x in enumerate(X):
    win_node=SOM.winner(x)
    plot(win_node[0]+0.5,win_node[1]+0.5,markers[Y[i]],markeredgecolor=colors[Y[i]],
         markerfacecolor='None',markersize=10,markeredgewidth=2)
show()

#Finding the frauds
mappings=SOM.win_map(X)
list_frauds=mappings[(3,1)]
list_frauds=sc.inverse_transform(list_frauds)