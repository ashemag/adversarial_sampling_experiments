## Adversarial Sampling As A Defense Against Discrimination 
The potential of imbalanced datasets to create bias/discriminations in a machine learning classifier is a widely studied problem.
However, certain classes within a dataset are more likely to be discriminated against than others when made the minority class.
The phenomena of interclass bias describes the ability of a classifier to generalize from the features of the other classes present in the data. 
That is to say, not all classes are discriminated against equally when they are the minority class in an imbalanced dataset. <br/> 

This can be explored through a series of experiments that test DenseNet model performance on varying sizes of minority
class for each of the 10 classes in the CIFAR-10 dataset. <br/> 

### Minority Class Experiments

`minority_class_experiments.py` experiments with induced minority classes within CIFAR-10 dataset. 
Running this file will download CIFAR-10 to a data directory if it is not present. </br>

<strong> Flags </strong>
<ul>
 <li>label: 1 of 10 CIFAR-10 classes 
 <li> seed: to recreate experiments
 <li> num_epochs: Number of epochs model trains for 
 <li> target_percentage: In %, integer value between 1 and 100
 <li> full_flag: For baseline experiments, True for balanced dataset (i.e. no minority class). Default=False
</ul>


This approach illustrates how the presence of a minority class of varying sizes 
affects the accuracy (includes f-score) of a machine learning classifier.

`models` folder includes our DenseNet implementations and a base class for training/evaluation. </br>

To run locally: <br/> 
`python minority_class_experiments.py --cpu True`

To run on slurm: <br/> 
Delete `minority_class_experiments*` files in `data/` for new round of experiments <br/>
`cd active_scripts`<br/> 
If experiment .sh files are not created, create by modifying `run_jobs.py` and uncommenting `create_files` function <br/>
`rm sl*` to remove old slurm files </br> 
`run_jobs.py` to deploy on slurm cluster </br>

### Adversarial Attack Experiments
Coming soon. </br>

Questions? Contact ashe.magalhaes@gmail.com.

> Copyright 2019 Ashe Magalhaes <br/> <br/>Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: <br/> <br/>The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. <br/><br/>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
