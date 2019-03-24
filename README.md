### Minority Class Experiments

`minority_class_experiments.py` experimenst with induced minority classes within CIFAR-10 dataset. 
Running this file will download CIFAR-10 to a data directory if it is not present. </br>

<strong> Flags </strong>
<ul>
 <li>label: 1 of 10 CIFAR-10 classes 
 <li> seed: to recreate experiments
 <li> num_epochs: Number of epochs model trains for 
 <li> target_percentage: In %, Int 100-1
 <li> full_flag: For baseline experiments, True for balanced dataset (i.e. no minority class). Default=False
</ul>


The objective of this baseline approach is to illustrate how the presence of a minority class
affects the standard accuracy of a machine learning classifier.

`models` folder includes our DenseNet implementations and a base class for training/evaluation. </br>

### Adversarial Attack Experiments
Coming soon. </br>

Questions? Contact ashe.magalhaes@gmail.com.