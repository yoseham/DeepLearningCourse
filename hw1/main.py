from solver import Solver
from visualize import plot_loss_and_acc

cfg = {
    'data_root': 'data',
    'max_epoch': 10,
    'batch_size': 100,
    'learning_rate': 0.01,
    'momentum': 0.5,
    'display_freq': 50,
}

runner = Solver(cfg)
loss1, acc1 = runner.train()
plot_loss_and_acc({
    "momentum=0": [loss1, acc1],
})