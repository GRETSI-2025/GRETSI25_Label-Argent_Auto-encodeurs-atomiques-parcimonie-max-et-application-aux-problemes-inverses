# Auto-encodeurs atomiques parcimonie-max et application aux problèmes inverses

Code for the paper "Auto-encodeurs atomiques parcimonie-max et application aux problèmes inverses" (GRETSI 2025)


[Ali Joundi](ali.joundi@u-bordeaux.fr), [Yann Traonmilin](https://yanntraonmilin.perso.math.cnrs.fr/), [Alasdair Newson](https://sites.google.com/site/alasdairnewson/home)



## Prerequisites
The numerical experiments in the paper were computed with python 3.8.5. To install the same config as in the paper:
```bash
pip install -r requirements.txt
```



## Overview of the code
Several scripts are provided:
- Hyperparameters.py : Allows to initialize variables, to choose the dataset and the desired atomic autoencoder
- trainer.py : Trains from scratch an atomic autoencoder
- decomposition.py : Displays the decomposition a given atomic autoencoder achieves
- Inverse_problem: Solve the super resolution inverse problem.
- Inverse_problem_hyperparameter.py: Hyperparameters of the inverse problem (step_size, iterations...)


## Reproductible figures and results
- Figure 2 - 3 - 4 - 5
- Recovery PSNRs can be computed via the scripts but the tables 1 and 2 are not displayed.

## Instructions to obtain figures/results

### Simple training
choose the desired options in Hyperparameters.py ie datasets_NAME (for the dataset MNIST, FASHION MNIST or CIFAR), dataset_size and operation (Simple, sum or max) and run:

```bash
python3 trainer.py
```
Return: folder (in Model_n_Res) with the model, the losses, the architecture and reconstruction figure.



### Figure 2/3/4
To decompose an image via a given atomic autoencoder and obtain the histograms, choose the desired options in Hyperparameters.py 
ie datasets_NAME (for the dataset MNIST, FASHION MNIST or CIFAR), subset_size and operation (max or sum) and run:

```bash
python3 decomposition.py
```
Return: A decomposition image, and the histogram of activated atoms for the chosen dataset (of size subset_size)


### Figure 5 Table 2
TO solve the super resolution inverse problem with a desired autoencoder, choose the dataset and the autoencoder type (simple, sum or min) in Hyperparameters.py. Inverse_problem_hyperparameter.py contains the parameters of the inverse problem (number of iterations, step size etc...). Run:

```bash
python3 inverse_problem.py
```
Return: Solutions of the super resolution inverse problem for (n="number_of_images_to_recover") images

If you want to compute the average PSNR of the recovery of a whole testset uncomment the last line of inverse_problem.py (and rerun).


## Expected execution time
- trainer.py: as it trains an atomic autoencoder, it depends on the dataset size, the computing ressources, the number of epochs. For example, on MNIST of a size 30k, using A100 GPUs, and for a training of 3000 epochs, It needed ~6 hours.
- decomposition.py: does not need more than a minute.
- Inverse_problem.py: depends on the hyperparameters (size of the considered dataset, number of iterations of the dataset). For my experiments, I needed 15min to compute the average PSNRs of the recovered images for 600 images with 150 iterations of PGD algorithm. This visual results are outputed instantly.

However, note that to reproduce the figures, one does not need more than a minute for all the command lines.




