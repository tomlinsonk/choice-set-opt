# Choice Set Optimization
This repository accompanies the paper:
> Kiran Tomlinson and Austin R. Benson. Choice Set Optimization Under Discrete Choice Models of Group Decisions. ICML 2020. [arXiv:2002.00421](https://arxiv.org/abs/2002.00421)

### Libraries 
We used:
- Python 3.6.8
- NumPy 1.17.2
- SciPy 1.3.1 
- PyTorch 1.4.0
- Matplotlib 3.1.1
- tqdm 4.35.0
- python-dateutil 2.8.0
- gurobipy 9.0.1 
- Gurobi 9.0.1 

### Files
- ``choice_models.py``: implementations of MNL, CDM, and NL models
- ``inference.py``: maximum-likelihood inference of MNL, low-rank CDM, and NL parameters from data
- ``optimize_choice_sets.py``: implementations of approximation, greedy, and brute force algorithms for choice set optimization
- ``real_data.py``: experiments on SFWork, YOOCHOOSE, and Allstate data
- ``make_plots.py``: generates all plots in the paper

The files ``choice_models.py``, ``inference.py``, and ``optimize_choice_sets.py`` contain example usages of our models,
inference process, and optimization algorithms that can be run with ``python3 [FILENAME]``.

The code in ``inference.py`` was gratefully adapted from [Arjun Seshadri's CDM code](https://github.com/arjunsesh/cdm-icml), which
accompanies the paper:
>Seshadri, A., Peysakhovich, A., and Ugander, J. Discovering context effects from raw choice data. 
In International Conference on Machine Learning, pp. 5660–5669, 2019.

### Data
The ``data/`` directory contains the SFWork and YOOCHOOSE datasets as well as supporting
documentation. The ``real_data.py`` script parses the data and contains additional info in comments. We do not include
the Allstate data in this repository, since it can be downloaded directly from Kaggle (see below).

#### SFWork
This dataset is provided in ``data/SFWork.mat.`` The fields in this file are described in ``SFWork-doc.xls``, which
was provided in the original dataset. For more information about the SFWork dataset, see:
>Koppelman, F. S. and Bhat, C. A self instructing course in mode choice modeling: Multinomial and nested logit models, 2006.

#### YOOCHOOSE
To uncompress this dataset, run ``tar -zxvf yoochoose-data.tar.gz`` from the ``data/`` directory.

This dataset is provided in ``data/yoochoose-data/yoochoose-filtered-clicks.dat`` and ``data/yoochoose-data/yoochoose-buys.dat``.
To reduce the file size, we selected all rows from the original ``yoochoose-clicks.dat`` file where
the ``category`` field is not ``S`` or ``0``, since our analysis uses only clicks with ``category > 0``. The original
dataset also contains the file ``yoochoose-test.dat``, which we did not use and omit form this repository to save space.
The fields in ``yoochoose-filtered-clicks.dat`` and ``yoochoose-buys.dat`` are described in ``data/yoochoose-data/dataset-README.txt``.

For more information about the YOOCHOOSE dataset, see:
> Ben-Shimon, D., et al. RecSys Challenge 2015 and the YOOCHOOSE Dataset. In Proceedings of the 9th ACM Conference on Recommender Systems, 2015.

#### Allstate

The Allstate dataset can be downloaded from Kaggle [here](https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data).
Place the uncompressed ``allstate-purchase-prediction-challenge/`` directory inside ``data/`` and uncompress ``train.csv``.
The Kaggle page contains info about the dataset, including all of its fields.

### Reproducibility
Our results are in the ``results/`` directory and our inferred models are saved in the ``models/`` directory (to use
them rather than inferring new models, use the ``--saved-models`` flag). Running ``python3 make_plots.py`` will
generate the all plots from the saved results. To re-run the experiments, begin by downloading the Allstate dataset as
specified above. Then:
```bash
python3 real_data.py --threads [NUM THREADS]
python3 make_plots.py
```
Running ``real_data.py`` produces the Allstate and YOOCHOOSE results files, saves tables showing optimal sets for 
SFWork, and prints out the SFWork summary statistics. In total, the experiments take about half a day to run (on my 
Intel Core i7-6700T CPU running Ubuntu 18.04.3 LTS). Since PyTorch uses numerical methods, it is possible that slightly
different models will inferred on different computer architectures or PyTorch/Python versions; however, this should have
no effect on the trends we observe.

If you don't have access to a Gurobi license (free for academic use), you can check out commit b03f43e3c33abab96522d010b7f706f88babb234
to run the old version of the code, which has everything except the MIBLP code.

Running ``python3 optimize_choice_sets.py`` will show details about the bad instance of Agreement for Greedy from the appendix.





