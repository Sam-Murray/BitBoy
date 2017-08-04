from mygrad.nnet.layers import dense, RecurrentUnit 
from mygrad.nnet.activations import relu
from mygrad import Tensor
import numpy as np
import csv

def unzip(l):
    return zip(*l)
#Gets data from bitfinex exchange
def loss(x,y_true):
    return (((x-y_true)**2).sum())/(len(x))
def grad_descent(t_params,rate):
    for w in t_params:
        w.data-=(w.grad*rate)
def get_reg(t_params,reg):
    tot=0
    for W in t_params:
        tot=tot + reg * (W**2).sum()
    return tot

def get_clump(xtrain,clumpSize,sentenceSize):
    clump=[]
    for i in range(clumpSize):
        r=np.random.randint(0,len(xtrain)-sentenceSize)
        clump.append(xtrain[r:r+sentenceSize])
    clump=np.array(clump)
    clump=np.transpose(clump)
    clump=clump[...,np.newaxis]
    return clump

x_train=np.load("xtrain.npy")
x_val=np.load("xval.npy")
x_test=np.load("xtest.npy")

#initailizes hyperparameters
#where d determines the number of trainable parameters for 1st layer
D=100
#where c is the context (1)
C=1 
#where P is the size of the prediction(output)
P=1
#where q determines the number of trainable parameters for 2nd layer
Q=10000
#where O is output size(1)
O=1
#Learning rate
rate=1e-7
#Regulation strength
reg=1e5
#Back Propigation Level
bp_lim=5
#sentence size
S=100
#clump size
N=200

#intitailizing trainable parameters

#DxD, where d determines the number of trainable parameters for 1st layer
W_0=Tensor(0.001 * np.random.randn(*(D,D)))

#CxD where c is the context (1)
U_0=Tensor(0.001 * np.random.randn(*(C,D)))

#DxP where P is the size of the prediction(output)
V_0=Tensor(0.001 * np.random.randn(*(D,P)))

#QxQ, where q determines the number of trainable parameters for 2nd layer
# W_1=Tensor(0.001 * np.random.randn(*(Q,Q)))

# #PxQ
# U_1=Tensor(0.001 * np.random.randn(*(P,Q)))

# #QxO where O is output size(1)
# V_1=Tensor(0.001 * np.random.randn(*(Q,O)))

#,W_1,U_1,V_1
paramList=[W_0,U_0,V_0]

#initailizes reccurent networks
recurrent0=RecurrentUnit(U_0,W_0,bp_lim)
#recurrent1=RecurrentUnit(U_1,W_1,bp_lim)

Loss=[]

for i in range(5000):
    clump=get_clump(x_train,N,S)
    HiddenDescriptor0=recurrent0(clump)
    layer1=dense(HiddenDescriptor0,V_0)
    L=(loss(layer1,np.transpose(clump))+get_reg(paramList,reg))
    print("Hoowoo")
    L.backward()
    print("WooHoo")
    grad_descent(paramList,rate)
    
    L.no_recursion_null_grad
    W_0.null_gradients()
    U_0.null_gradients()
    np.save("W",W_0)
    np.save("U",U_0)
    np.save("V",V_0)
    Loss.append(L.data)
    np.save("Loss",np.array(Loss))
    print(i)
    print(L.data)
    