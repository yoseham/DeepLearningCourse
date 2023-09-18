""" ReLU激活层 """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		ReLU激活函数: relu(x) = max(x, 0)
		"""
		self.trainable = False # 没有可训练的参数

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入应用ReLU激活函数并返回结果
	    ############################################################################
		self.df = np.where(Input < 0, 0, 1)
		return np.where(Input < 0, 0, Input)

	def backward(self, delta):

		############################################################################
	    # TODO: 
		# 根据delta计算梯度
	    ############################################################################
		return delta * self.df