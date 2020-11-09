import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):

    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
   
    fx=sigmoid(x)
    return fx * (1-fx)

def mse_loss(y_true, y_pred):
  
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork():
 
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
    
    def feedforward(self, x):
    #x is a numpy array with 2 elements, for example [input1, input2]
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # samples in the dataset.
        - all_y_trues is a numpy with n elements.
        Elements in all_y_trues correspond to those indata.
        """
        learn_rate = 0.1 
        epochs = 1000 
        loss_f =  np.zeros((epochs, 1))
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
            #---Do a feedforward (we'll need values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 =  sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 =  sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

            #--- Calculate partial derivatives.
            #--- Naming: d_L_d_w1 represents "partial L/partial w1"
                d_L_d_ypred = - (y_true - y_pred)

            #Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

            #Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

            #Neuron h2
                d_h1_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h1_d_w4 = x[0] * deriv_sigmoid(sum_h2)
                d_h1_d_b2 = deriv_sigmoid(sum_h2)

            #--- update weights and biases
            #Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            #Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

            #Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h1_d_b2

    #--- Calculate total loss at the end of each epoch
    
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            loss_f[epoch] = loss
            if epoch % 10 == 0:
                print ("Epoch %d loss: %.3f", (epoch, loss))

    #fig,ax = plt.subplots()
        plt.plt(loss_f)
        plt.title('loss_function')
        plt.show()

if __name__ == '__main__':

    #Define dataset
    data = np.array([
    [0.511, 1],
    [1.962, 1.625],
    [-0.820, -0.875],
    [-1.182, -1.5],
    [-0.094, -0.75],
    [0.873, 1.125],
    [0.148, 0.75],
    [-1.424, -0.75],
    [0.027, -0.625],
    ])

    all_y_trues = np.array([
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
    ])

    #Train our neural network
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)


