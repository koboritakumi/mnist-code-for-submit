import numpy as np
import matplotlib.pyplot as plt
import random
import pandas
import tensorflow as tf
import tensornetwork as tn
from opt_einsum import contract
import time

"""
The code is the normal fc2 neural network for mnist.
The optimizer uses minibatch SGD.
"""

def y_initialize(y_data):
        y=[]
        for _ in range(len(y_data)):
            now=np.zeros(10)
            now[y_data[_]]=1.0
            y.append(now)

        return np.array(y)

class fc2:

    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((60000,28*28))/255.0
    X_test = X_test.reshape((10000,28*28))/255.0
    w1 = np.random.random((16*16,28*28))-0.5
    w2 = np.random.random((10,16*16))-0.5
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
        self.time =time.time()

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

    def soft_max(self,input_vector):
        return np.exp(input_vector)/(np.exp(input_vector).sum())

    def delta_relu(self,input_vector,ref_vector):
        output_vector = [input_vector[i] if ref_vector[i]>0 else 0 for i in range(len(input_vector))]
        return output_vector

    def forward(self,X_data):
        self.outputs.clear()
        now = X_data.copy()
        
        self.outputs.append(now)
        now = self.w1@now
        
        self.outputs.append(now)
        now = self.relu(now)
        
        self.outputs.append(now)
        now = self.w2@now
        
        self.outputs.append(now)
        now = self.soft_max(now)
        
        self.outputs.append(now)

        return now

    def backward(self,y):
        now_num=len(self.outputs)-1
        delta0 = self.outputs[now_num]-y
        now_num-=1
        delta1 = np.outer(delta0,self.outputs[now_num-1])
        delta2 = self.w2.T@delta0
        now_num-=1
        now_num-=1
        delta3 = self.delta_relu(delta2,self.outputs[now_num])
        now_num-=1
        delta4 = np.outer(delta3,self.outputs[now_num])

        self.w2-=self.eta*delta1
        self.w1-=self.eta*delta4

    
    def loss_function(self,x_data,y_data):
        return -(y_data.T@np.log(x_data))

    def fit(self):
        n_sample = int(np.ceil(len(self.X_train)/self.batch_size))
        randomArray = random.shuffle(list(range(len(self.X_train))))
        batches_x = [[self.X_train[randomArray[i+n_sample*_]]for i in range(n_sample)] for _ in range(self.batch_size)]
        batches_y = [[self.yr_train[randomArray[i+n_sample*_]]for i in range(n_sample)] for _ in range(self.batch_size)]
        for _ in range(self.n_iter):
            index = np.random.randint(0,len(self.n_iter))
            batch_x = batches_x[index]
            batch_y = batches_y[index]
            output = self.forward(batch_x)
            self.backward(batch_y)
            

            
            now_acc =0
            now_loss = 0.
            #for i in range(len(self.X_train)):
            #    output = self.forward(self.X_train[i])
            #    if np.argmax(output)==self.y_train[i]:
            #        now_acc+=1 
#
            #    now_loss += self.loss_function(output,self.yr_train[i])
            for i in range(len(self.X_test)):
                output = self.forward(self.X_test[i])
                if np.argmax(output)==self.y_test[i]:
                    now_acc+=1 
                now_loss += self.loss_function(output,self.yr_test[i])

            #now_acc/=(len(self.X_train))
            #now_loss/=(len(self.X_train))
            now_acc/=(len(self.X_test))
            now_loss/=(len(self.X_test))
            #self.train_accuracy.append(now_acc)
            self.test_accuracy.append(now_acc)
            self.train_loss.append(now_loss)
            #print('iter:',_,'accuracy:',now_acc,'loss_function:',now_loss,'time:',time.time()-self.time)
            print('iter:',_,'accuracy:',now_acc,'loss_function:',now_loss,'time:',time.time()-self.time)
            self.time = time.time()

        plt.plot(self.test_accuracy)
        plt.show()

    def test(self):
        now_acc =0
        now_loss = 0.
        for i in range(len(self.X_test)):
            output = self.forward(self.X_test[i])
            if np.argmax(output)==self.y_test[i]:
                now_acc+=1 
            now_loss += self.loss_function(output,self.yr_test[i])

        now_acc/=(len(self.X_test))
        now_loss/=(len(self.X_test))
        self.test_accuracy.append(now_acc)
        self.test_loss.append(now_loss)
        print('test_accuracy:',now_acc,'test_loss_function:',now_loss)



if __name__ == '__main__':
    cnn = fc2(eta=0.01,n_iter=100,batch_size=100)
    cnn.fit()
    cnn.test()