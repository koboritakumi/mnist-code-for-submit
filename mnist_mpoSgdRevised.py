import numpy as np
#import cupy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensornetwork as tn
from opt_einsum import contract
import time
import random

"""
The code is the fc2 neural network for mnist whose linear operations are replaced by mpo.
(28,28)->(4,7,7,4)->(4,4,4,4)->(1,1,10,1)->(10)
The optimizer uses minibatch SGD.
"""

def X_tensor(X_data):
    now_data = [[[[[X_data[i][7*j+k][7*m+l]/255.0 for m in range(4)]for l in range(7)]for k in range(7)]for j in range(4)] for i in range(len(X_data))]
    return np.array(now_data)

def y_initialize(y_data):
        y=[]  
        for _ in range(len(y_data)):
            now=np.zeros(10)
            now[y_data[_]]=1.0
            y.append(now)

        return np.array(y)

def set_w1(d):
    W=[]
    for i in range(4):
        if i==0 or i==3:
            now_w = np.random.random((4,4,d))-0.5
        else :
            now_w = np.random.random((7,4,d,d))-0.5
        W.append(now_w)
    return W

def set_w2(d):
    W=[]
    for i in range(4):
        if i==0 or i==3:
            now_w = np.random.random((4,1,d))-0.5
        elif i==1:
            now_w = np.random.random((4,1,d,d))-0.5
        else :
            now_w = np.random.random((4,10,d,d))-0.5
        W.append(now_w)
    return W


class mpo_fc2:

    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    tX_train = X_train.reshape((60000,4,7,7,4))/255.0
    tX_test = X_test.reshape((10000,4,7,7,4))/255.0
    d = 4
    w1 = set_w1(d)
    w2 = set_w2(d)
    yr_train = y_initialize(y_train)
    yr_test = y_initialize(y_test)
    
    def __init__(self,eta=0.01,n_iter=1000,batch_size=100):
        self.eta=eta
        self.n_iter=n_iter
        self.batch_size = batch_size
        self.train_loss = []
        self.test_loss = []
        self.outputs = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.time = time.time()

    def relu(self,input_tensor):
        shape = input_tensor.shape
        shape_pro = 1
        for i in shape : 
            shape_pro = shape_pro*i
        output_tensor = input_tensor.copy().reshape(shape_pro)
        for i in range(len(output_tensor)):
            output_tensor[i] = output_tensor[i] if output_tensor[i]>0 else 0
        output_tensor = output_tensor.reshape(shape)

        return output_tensor 

    def soft_max(self,input_tensor,n_sample):
        return [np.exp(input_tensor[i])/(np.exp(input_tensor[i]).sum()) for i in range(n_sample)]

    def delta_relu(self,input_tensor,ref_tensor):
        shape = input_tensor.shape
        shape_pro = 1
        for i in shape : 
            shape_pro = shape_pro*i
        finput_tensor = input_tensor.copy().reshape(shape_pro)
        fref_tensor = ref_tensor.copy().reshape(shape_pro)
        for i in range(len(finput_tensor)):
            finput_tensor[i] = finput_tensor[i] if fref_tensor[i]>0 else 0
        finput_tensor = finput_tensor.reshape(shape)

        return finput_tensor 

    def forward(self,X_data,n_sample):
        self.outputs.clear()
        now = X_data.copy()
        
        self.outputs.append(now)
        
        now = contract('ija,klab,mnbc,opc,uikmo->ujlnp',self.w1[0],self.w1[1],self.w1[2],self.w1[3],now)
        
        self.outputs.append(now)
        now = self.relu(now)
        
        self.outputs.append(now)
        
        now = contract('ija,klab,mnbc,opc,uikmo->ujlnp',self.w2[0],self.w2[1],self.w2[2],self.w2[3],now)
        now = now.reshape(n_sample,10)
        
        self.outputs.append(now)
        
        now = self.soft_max(now,n_sample)
        
        self.outputs.append(now)

        return now
        """"
        outputs=[入力(4,7,7,4),w1入力(4,4,4,4),relu後(4,4,4,4),w2後(10),soft_max(10)]
        """
    def backward(self,y,n_sample):
        delta0 = np.array(self.outputs[4])-np.array(y)
        
        delta0 = delta0.reshape(n_sample,1,1,10,1)
        deltaw20 = contract('uxbcd,bqzj,cwjk,dek,uyqwe->uxyz',self.outputs[2],self.w2[1],self.w2[2],self.w2[3],delta0)
        deltaw21 = contract('uaxcd,aqz,cwpk,dek,uqywe->uxyzp',self.outputs[2],self.w2[0],self.w2[2],self.w2[3],delta0)
        deltaw22 = contract('uabxd,aqi,bwiz,drp,uqwyr->uxyzp',self.outputs[2],self.w2[0],self.w2[1],self.w2[3],delta0)
        deltaw23 = contract('uabcx,aqi,bwij,crjz,uqwry->uxyz',self.outputs[2],self.w2[0],self.w2[1],self.w2[2],delta0)

        delta1 = contract('aqi,bwij,crjz,dyz,uqwry->uabcd',self.w2[0],self.w2[1],self.w2[2],self.w2[3],delta0)
        
        delta2 = self.delta_relu(delta1,self.outputs[1])
        
        deltaw10 = contract('uxbcd,bqzj,cwjk,dek,uyqwe->uxyz',self.outputs[0],self.w1[1],self.w1[2],self.w1[3],delta2)
        deltaw11 = contract('uaxcd,aqz,cwpk,dek,uqywe->uxyzp',self.outputs[0],self.w1[0],self.w1[2],self.w1[3],delta2)
        deltaw12 = contract('uabxd,aqi,bwiz,drp,uqwyr->uxyzp',self.outputs[0],self.w1[0],self.w1[1],self.w1[3],delta2)
        deltaw13 = contract('uabcx,aqi,bwij,crjz,uqwry->uxyz',self.outputs[0],self.w1[0],self.w1[1],self.w1[2],delta2)
        

        self.w2[0]-=self.eta*np.einsum('uxyz->xyz',deltaw20)/n_sample
        self.w2[1]-=self.eta*np.einsum('uxyzp->xyzp',deltaw21)/n_sample
        self.w2[2]-=self.eta*np.einsum('uxyzp->xyzp',deltaw22)/n_sample
        self.w2[3]-=self.eta*np.einsum('uxyz->xyz',deltaw23)/n_sample
        self.w1[0]-=self.eta*np.einsum('uxyz->xyz',deltaw10)/n_sample
        self.w1[1]-=self.eta*np.einsum('uxyzp->xyzp',deltaw11)/n_sample
        self.w1[2]-=self.eta*np.einsum('uxyzp->xyzp',deltaw12)/n_sample
        self.w1[3]-=self.eta*np.einsum('uxyz->xyz',deltaw13)/n_sample
    
    def loss_function(self,x_data,y_data):
        return np.array([-(y_data[i].T@np.log(x_data[i]))for i in range(len(y_data))]).sum()

    def fit(self):
        n_sample = int(np.ceil(len(self.tX_train)/self.batch_size))
        randomArray = list(range(len(self.X_train)))
        random.shuffle(randomArray)
        batches_x = [[self.tX_train[randomArray[i*n_sample+j]]for j in range(n_sample)]for i in range(self.batch_size)]
        batches_y = [[self.yr_train[randomArray[i*n_sample+j]]for j in range(n_sample)]for i in range(self.batch_size)]
        for _ in range(self.n_iter):
            index = np.random.randint(0,self.batch_size)
            batch_x = batches_x[index]
            batch_y = batches_y[index]

            output = self.forward(batch_x,n_sample)

            self.backward(batch_y,n_sample)
            

            if _%100==0:
                output = self.forward(self.tX_test,len(self.tX_test))
                now_acc = np.array([(np.argmax(output[i])==self.y_test[i]) for i in range(len(self.tX_test))]).sum() / (len(self.X_test))
                now_loss = self.loss_function(output,self.yr_test) / (len(self.X_test))

                self.test_accuracy.append(now_acc)
                self.train_loss.append(now_loss)
                print('iter:',_,'accuracy:',now_acc,'loss_function:',now_loss,'time:',time.time()-self.time)
                self.time = time.time()

        plt.plot(self.test_accuracy)
        plt.show()

        output = self.forward(self.tX_test,len(self.tX_test))
        now_acc = np.array([np.argmax(output[i])==self.y_test[i] for i in range(len(self.tX_test))]).sum() / (len(self.X_test))
        now_loss = self.loss_function(output,self.yr_test) / (len(self.X_test))
        print('iter:',_,'accuracy:',now_acc,'loss_function:',now_loss,'time:',time.time()-self.time)

    #def test(self):
    #    now_acc =0
    #    now_loss = 0.
    #    for i in range(len(self.tX_test)):
    #        output = self.forward(self.tX_test[i])
    #        if np.argmax(output)==self.y_test[i]:
    #            now_acc+=1 
    #        now_loss += self.loss_function(output,self.yr_test[i])
#
    #    now_acc/=(len(self.tX_test))
    #    now_loss/=(len(self.tX_test))
    #    self.test_accuracy.append(now_acc)
    #    self.test_loss.append(now_loss)
    #    print('test_accuracy:',now_acc,'test_loss_function:',now_loss)



if __name__ == '__main__':
    cnn = mpo_fc2(eta=0.8,n_iter=25000,batch_size=100)
    cnn.fit()
    #cnn.test()
