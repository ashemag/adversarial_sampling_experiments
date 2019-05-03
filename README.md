### Minority Class Experiments

`minority_class_experiments.py` experiments with induced minority classes within CIFAR-10 dataset. 
Running this file will download CIFAR-10 to a data directory if it is not present. </br>

<strong> Flags </strong>
<ul>
 <li>label: 1 of 10 CIFAR-10 classes 
 <li> seed: to recreate experiments
 <li> num_epochs: Number of epochs model trains for 
 <li> target_percentage: In %, Int 100-1
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