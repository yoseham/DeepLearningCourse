import numpy as np
import torch

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11


class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            num_input: size of each input sample
            num_output: size of each output sample
            trainable: whether if this layer is trainable
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

    def forward(self, Input, labels):
        """
          Inputs: (minibatch)
          - Input: (batch_size, 784)
          - labels: the ground truth label, shape (batch_size, )
        """

        ############################################################################
        # TODO: Put your code here
        # Apply linear transformation (WX+b) to Input, and then
        # calculate the average accuracy and loss over the minibatch
        # Return the loss and acc, which will be used in solver.py
        # Hint: Maybe you need to save some arrays for gradient computing.

        ############################################################################
        input_size = len(Input)

        score = np.dot(Input, self.W) + self.b   # 计算wx+b

        shift_score = score - np.max(score, axis=1, keepdims=True)   # 减去最大值，避免上溢或下溢

        softmax_score = np.exp(shift_score) / np.sum(np.exp(shift_score), axis=1, keepdims=True)    # 计算softmax

        onehot_label = np.zeros_like(softmax_score)
        onehot_label[range(input_size), labels] = 1        # 将label 转为 onehot编码

        loss = np.sum(-onehot_label * np.log(softmax_score)) / input_size    # 计算损失

        acc = np.sum(labels == np.argmax(softmax_score, axis=1)) / input_size   # 计算正确率

        # 保存用以计算dW， dB
        self.X = Input
        self.labels = onehot_label
        self.softmax_score = softmax_score

        return loss, acc

    def gradient_computing(self):
        # 根据公式，计算dW， dB
        self.grad_W = -np.dot(self.X.T, self.labels - self.softmax_score) / len(self.X)
        self.grad_b = -np.sum(self.labels - self.softmax_score) / len(self.X)


        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient of W and b.

        # self.grad_W = 
        # self.grad_b =
        ############################################################################


    def XavierInit(self):
        """
        Initialize the weigths
        """
        raw_std = (2 / (self.num_input + self.num_output)) ** 0.5
        init_std = raw_std * (2 ** 0.5)
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
