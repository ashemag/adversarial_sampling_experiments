# mlp

## Baseline
`data_experiments.py` allows us to iteratively induce a minority class within CIFAR-10 dataset. 
`scripts.py` demonstrates how we can set num_epochs, seed, target percentage, and target label to
compare a representative dataset with an imbalanced dataset that reduced the target label to target percentage.

The objeective of this baseline approach is to illustrate how the presence of a minority class
affects the standard accuracy of a CNN classifier.

`ModelBuilder` includes implementations of common neural network architectures.

Questions: Contact ashe.magalhaes@gmail.com