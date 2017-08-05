from mygrad.nnet.layers import dense, recurrent
from mygrad.nnet.activations import relu
from mygrad import Tensor
from os import mkdir
import numpy as np
from itertools import product
import csv

def unzip(l):#I love you unzip
    return zip(*l)
class bitboy():
    """Wrapper class for RNN trained on bitcoin price data.
        Feilds
        ------
        W: Tensor shape(DxD). Trainable parameter.
        U: Tensor shape(CxD). Trainable parameter.
        V: Tensor shape(DxP). Trainable parameter.
        N: Int, Hyperparameter. Size of Clump.
        S: Int, Hyperparameter. Sequence length.
        x_train: 1d Numpy array. Contains training data.
        x_test: 1d Numpy array. Contains testing data.
        x_val: 1d Numpy array. Contains testing data.
        bp_lim: Int Hyperparameter. Back propigation Limit(May be None).
        reg: Int, Hyperparameter. Regulation rate.
        rate:Int, Hyperparameter. Learning rate.
        Losses:The loss after each training iteration.
        
        Methods
        -------
        init: Initializes learning parameters and hyperparameters
        load_data: Will load data from their respective .npy files, or load and process cvs file containing
                    (time,price,volume) of bitcoin if given a target directory to load from. Will overide
                    current .npy file with new information.
        load_params: Loads Trainable parameter from their .npy files
        save_params: Saves Trainable parameter to their .npy files
        train: Trains trainable parameters with gradeint descent
        grad_descent: Uses gradeint descent on each of the trainable params.
        loss: Computes loss for 1 clump that has gone through a forward pass.
        get_reg: Computes regulation portion of loss.
        get_clump:Gets a SxN clump of training data, where get_clump()[:][0] is a consecutive sequence.
        
        """
    
    def __init__(self,D,C,P,S,N,rate,reg,bp_lim=5):
        """Initalizes Trainable Params:
            Params:
            ------
            D:(int)Size of hidden layer (S+1,N,D).
            C:(int)Size of input context, usually 1.
            P:(int)Size of output data, usually also 1.
            S:(int)Size of sentences in clump.
            N:(int)Size of clump.
            rate:(float)Learning rate.
            reg:(float)regulation rate."""
        #DxD, where d determines the number of trainable parameters for 1st layer
        self.W=Tensor(0.001 * np.random.randn(*(D,D)))
        #CxD where c is the context (1)
        self.U=Tensor(0.001 * np.random.randn(*(C,D)))
        #DxP where P is the size of the prediction(output)
        self.V=Tensor(0.001 * np.random.randn(*(D,P)))
        
        self.N=N
        self.S=S
        self.bp_lim=bp_lim
        self.rate=rate
        self.reg=reg
    def get_average_price(self,time,price,volume,part_leng=5):
        """Gets the average price of bitcoin over every part_leng transactions.
            Params
            ------
            time:(np.array(int)), The Unix time of each transaction.
            price:(np.array(float)),Price that bitcoin was sold during transaction.
            volume:(np.array(float)), the volume of each transition.
            part_leng:(int,(Optional:Default=5)), The partition length.
            Return
            ------
            Tuple(np.array(float),np.array(int)), The average weighted price over each partition, the time interval of each
            partition.
            
            
            """
        slicelength=len(price)//part_leng
        slicestart=len(price)%part_leng
        tprice=np.reshape(price[slicestart:],(slicelength,5))
        tvol=np.reshape(volume[slicestart:],(slicelength,5))
        voltot=np.sum(tvol,1)
        tTime=time[slicestart::part_leng]
        weightedPrice=np.einsum('ij,ij->i', tprice, tvol)/voltot
        return (weightedPrice, tTime)

    def load_data(self,fileString=None,xtrain=195000,xtest=50000,xval=5000):
        """Loads training data/test data from file string, or from respective .npy files if fileString=None
           Params
           ------
           fileString(string,(Optional, Default=None)): The csv file from which to load the (time,price,volume) tuples. If none,
           will try to load the .npy for x_train, x_val, and x_test files from local directory.
           xtrain:(int)length of x_train
           xtest:(int)length of x_test
           xval:(int)length of x_val
           Return
           ------
           None
           """
        if fileString == None:
            self.x_train=np.load("xtrain.npy")
            self.x_val=np.load("xval.npy")
            self.x_test=np.load("xtest.npy")
        else:
            print("opening file")
            with open(fileString, mode='r') as q:
                print("in file")
                data=[]
                File=csv.reader(q, delimiter=',')
                for line in File:
                    data.append(line)
            timePriceVolume=list(unzip(data))
            print("done reading in")
            time=np.array(timePriceVolume[0],dtype=int)
            price=np.array(timePriceVolume[1],dtype=float)
            volume=np.array(timePriceVolume[2],dtype=float)
            aPrice,Time=self.get_average_price(time,price,volume)
            print("done getting average price")
            aPriceData=aPrice[-(xtrain+xtest,xval):]
            self.x_train=aPriceData[:xtrain]
            self.x_val=aPriceData[xtrain:xtrain+xval]
            self.x_test=aPriceData[xtrain+xval:]
    def get_clump(self,x):
        """Clumps x data into a (S,N,1) array of random sequences in the x data; takes N random S length sequence from x data
            and transposes them
            Params
            ------
            x(np.array(floats)): The data to be clumped.
            Return
            ------
            np.array(floats):shape = (S,N,1), clumped data"""
        clump=[]
        for i in range(self.N):
            r=np.random.randint(0,len(x)-self.S)
            clump.append(x[r:r+self.S])
        clump=np.array(clump)
        clump=np.transpose(clump)
        clump=clump[...,np.newaxis]
        return clump
    def get_loss(self,x,y_true):
        """Takes the truth and predicted data, returns loss
            Params
            ------
            x(Tensor):Tensor of predictions. Must be same shape as y_true.
            y_true(np.array(floats)):Array of true data. Must be same shape as x.
            Return
            ------
            (Tensor)The mean squared error of x and y_true."""
        
        return (((x-y_true)**2).sum())/(len(x))
    
    def grad_descent(self):
        """Performs gradent descent on all parameters"""
        self.W.data-=(self.W.grad*self.rate)
        self.U.data-=(self.U.grad*self.rate)
        self.V.data-=(self.V.grad*self.rate)
    def get_reg(self):
        """Gets regulation component of error
           Return
           ------
           The regulation component of error"""
        tot=0
        tot=tot + self.reg * (self.W**2).sum()
        tot=tot + self.reg * (self.U**2).sum()
        tot=tot + self.reg * (self.V**2).sum()
        return tot
    def save_params(self,Wfile="W.npy",Ufile="U.npy",Vfile="V.npy"):
        """Saves parameters to respective .npy files"""
        np.save(Wfile,self.W)
        np.save(Ufile,self.U)
        np.save(Vfile,self.V)
    def load_params(self,Wfile="W.npy",Ufile="U.npy",Vfile="V.npy"):
        """loads parameters from respective .npy files"""
        self.W=np.load(Wfile)
        self.U=np.load(Ufile)
        self.V=np.load(Vfile)
    def save_loss(self,lossFile="Loss.npy"):
        np.save(lossFile,self.loss.data)
    def return_loss(self):
        return self.loss
    def train(self,iterations=5000):
        """Trains all parameters on training data.
            Params
            ------
            iterations(int):The number of training iterations."""
        self.loss=[]
        for i in range(iterations):
            clump=self.get_clump(self.x_train)
            
            HiddenDescriptor=recurrent.simple_RNN(clump,self.U,self.W,bp_lim=self.bp_lim)
            layer1=dense(HiddenDescriptor,self.V)
            L=self.get_loss(layer1,np.insert(clump, 0, 0, axis=0))+self.get_reg()
            print("Hoowoo")
            L.backward()
            print("WooHoo")
            self.grad_descent()
            L.null_gradients()
            self.loss.append(L.data)
            print(i)
            print(L.data)
        self.save_params()

def test_hyper(rates,regs):
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
    i=0
    losses=[]
    ratereg=[]
    for rate, reg in product(rates,regs):
        
        bb=bitboy(D,C,P,S,N,rate,reg)
        bb.load_data()
        bb.train(1000)
        losses.append(bb.return_loss())
        ratereg.append([rate,reg])
    np.save("ratereg.npy",np.array(ratereg))
    np.save("Losses.npy",np.array(losses))
            
if __name__ == "__main__":
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
    rate=1e-9
    #Regulation strength
    reg=1e3
    #Back Propigation Level
    bp_lim=5
    #sentence size
    S=100
    #clump size
    N=200
    bb=bitboy(D,C,P,S,N,rate,reg)
    bb.load_data()
    bb.train(50000)
    
    
    
    
    
    
    
    
    
    
    