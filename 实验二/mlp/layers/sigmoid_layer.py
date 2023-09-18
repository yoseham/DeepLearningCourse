""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Sigmoid激活函数: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入应用Sigmoid激活函数并返回结果
	    ############################################################################

		sigmoid = lambda x: 1 / (1 + np.exp(-x))
		y = sigmoid(Input)
		self.df = y * (1 - y)
		return y

	def backward(self, delta):

		############################################################################
	    # TODO: 
		# 根据delta计算梯度
	    ############################################################################
		return delta * self.df