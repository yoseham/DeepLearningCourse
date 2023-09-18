""" Softmax交叉熵损失层 """

import numpy as np

# 为了防止分母为零，必要时可在分母加上一个极小项EPS
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      输入: (minibatch)
	      - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
	      - gt: 真实标签, 尺寸(batch_size, 10)
	    """

		############################################################################
	    # TODO: 
		# 在minibatch内计算平均准确率和损失，分别保存在self.accu和self.loss里(将在solver.py里自动使用)
		# 只需要返回self.loss
	    ############################################################################

		input_size = len(logit)
		shift = logit - np.max(logit, axis=1, keepdims=True)  # 减去最大值，避免上溢或下溢
		self.softmax_score = np.exp(shift) / np.sum(np.exp(shift), axis=1, keepdims=True)  # 计算softmax
		self.loss = np.sum(-gt * np.log(self.softmax_score)) / input_size  # 计算损失
		self.acc = np.sum(np.argmax(gt, axis=1) == np.argmax(self.softmax_score, axis=1)) / input_size  # 计算正确率
		self.labels = gt

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)
	    ############################################################################

		return self.softmax_score - self.labels