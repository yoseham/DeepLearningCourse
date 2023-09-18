""" 欧式距离损失层 """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.

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
		self.loss = np.sqrt(np.sum((logit - gt)**2)) / 2     # 计算欧式距离
		self.acc = np.sum(np.argmax(gt, axis=1) == np.argmax(logit, axis=1)) / input_size
		self.gt = gt
		self.logit = logit

		return self.loss

	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)
	    ############################################################################
		return self.logit - self.gt