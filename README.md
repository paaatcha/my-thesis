# My thesis repository

In this repository I share the code I developed to my PhD thesis. I provide some intructions of how to use it in the following.
If you find any bug or have any observation, please, let me know. Don't hesitate to get in touch with me.

- Thesis: `Combining heterogeneous data and deep learning models for skin cancer detection`
- Link to the thesis: soon

## Dependencies
All code is done using Python enviroment. Machine/Deep leaning models are implemented using [Scikit-Learn](https://scikit-learn.org/stable/) and [Pytorch](https://pytorch.org/).

If you already have the Python enviroment set in your **Linux** machine, all you need to do in order to install all dependencies is to run the command `pip install -r requirements.txt`

If you're in the wrong side of the force and are using Windows, I suggest you to use [Anaconda](https://www.anaconda.com/) to set up your enviroment. However, I don't know if everything will work properly since I use Linux.

## Other repositories
To run this code you're going to need to clone other repositories from my Github:
- [Raug](https://github.com/paaatcha/raug): this is reposible to train the deep learning models. You may find mode instruction on its own `Readme.md`
- [Decision-Making](https://github.com/paaatcha/decision-making): this repository contains the code for decision-making approaches. Essecialy, you're going use it only if you want to use A-TOPSIS. I also implemented a method to compute an ensemble aggregation using TOPSIS, but it was just a quick experiment and is not reported on my thesis.

After cloning these repositories, you must set the paths on `constants.py` file. You're going to find instructions there.

## Organization
- In `my_models` folder you find the CNN models implementations, the combination baseline, MetaNet, and MetaBlock.
- In `benchmarks` folder are the scripts for the experiments for each dataset
- In `LewDir`, as the name suggests, are the codes related to the aggregation ensemble methods.

To run the benchmarks, I used [Sacred](https://sacred.readthedocs.io/en/stable/index.html). A tool to organize experiments.
You don't need to kown how to use in order to run the code, although I strong recommend you to learn it.

Using Sacred, you may run an experiment in the following way:
`python pad.py with _lr=0.001 _batch_size=50`

If you don't want to use it, you can change the parameters directly in the code.

**Important**: you must set the path to the dataset for each benchmark within the `benchmarks` folders.


## Where can I find the datasets?
You may find the link to all datasets I used in my thesis in the following list:
- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2)
- [ISIC 2019](https://challenge2019.isic-archive.com/)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [NCT](https://zenodo.org/record/1214456)
- [OCT](https://data.mendeley.com/datasets/rscbjbr9sj/3)

