# Attentive Neural Models for Algorithmic Trading
## CS 230, Stanford University

This repository requires Tensorflow 1.13.1 and NumPy to run.

We experimented with reinforcement learning approaches and attentive neural processes before settling on a supervised SNAIL approach. All the code has been included for context, but the relevant files for the final submission are:

- **framework.py**
- **data_utils.py**
- **snail.py**
- **plotting.py**
- **run_framework.sh**

The baseline and experimental models are included in /baseline/ and /experimental/.

## Demos

Run **plotting.py** to see plots of historical performance of the experimental and baseline models compared to the Dow Jones Industrial average and Intel Corporation.

Since the dataset used for this model is private, we are unfortunately unable to provide any samples. Run **run_framework.sh** to watch models train on dummy data.

Use **load=True** in **framework.py** to load the experimental and baseline models and run them on the dummy data (they are trained on real data so they will not be successful).
