# Auto-encodeurs atomiques parcimonie-max et application aux problèmes inverses

Code for the paper "Auto-encodeurs atomiques parcimonie-max et application aux problèmes inverses" (GRETSI 2025)


[Ali Joundi](ali.joundi@u-bordeaux.fr), [Yann Traonmilin](https://yanntraonmilin.perso.math.cnrs.fr/), [Alasdair Newson](https://sites.google.com/site/alasdairnewson/home)



## Prerequisites
The numerical experiments in the paper were computed with:__
python 3.9.6, __
torch 2.0.1,__
Numpy 1.24.4,__
Matplotlib 3.7.4,__
Pillow 8.0.1__




## The code
Several scripts are provided. __
Hyperparameters.py : Allows to initialize variables, to choose the dataset and the desired atomic autoencoder __
trainer.py : Trains from scratch an atomic autoencoder__
decomposition.py : Displays the decomposition a given atomic autoencoder achieves__
Inverse_problem: Solve the super resolution inverse problem. __
Inverse_problem_hyperparameter.py: Hyperparameters of the inverse problem (step_size, iterations...)__


## Reproductible figures and results
Figure 2 - 3 - 4 - 5__
Recovery PSNRs can be computed via the scripts but the tables 1 and 2 are not displayed.__

## Instructions to obtain figures/results

### Simple training
choose the desired options in Hyperparameters.py ie datasets_NAME (for the dataset MNIST, FASHION MNIST or CIFAR), dataset_size and operation (Simple, sum or max) and run:__

python3 trainer.py__
Return: folder (in Model_n_Res) with the model, the losses, the architecture and reconstruction figure.__

### Figure 2/3/4
To decompose an image via a given atomic autoencoder and obtain the histograms, choose the desired options in Hyperparameters.py 
ie datasets_NAME (for the dataset MNIST, FASHION MNIST or CIFAR), subset_size and operation (max or sum) and run:__

python3 decomposition.py__
Return: A decomposition image, and the histogram of activated atoms for the chosen dataset (of size subset_size)__


### Figure 5 Table 2
TO solve the super resolution inverse problem with a desired autoencoder, choose the dataset and the autoencoder type (simple, sum or min) in Hyperparameters.py. Inverse_problem_hyperparameter.py contains the parameters of the inverse problem (number of iterations, step size etc...). Run:__

python3 inverse_problem.py__
Return: Solutions of the super resolution inverse problem for (n="number_of_images_to_recover") images__

If you want to compute the average PSNR of the recovery of a whole testset uncomment the last line of inverse_problem.py (and rerun)__




