# tq_for_CMS_L1Tk

Authors: Claire Savard & Chris Brown 
Email: claire.savard@colorado.edu

This package will get you started with creating a machine
learning classifier that computes track quality for L1Tk
reconstructed tracks. Included are files that give examples
for how to create a keras neural network (tq_NN.ipynb) or a 
scikit-learn or xgboost gradient-boosted decision tree
(tq_GBDT.ipynb), how to evaluate these models, and then how
to save them to file.

Before running, you will need to install:
1. *jupyter notebook (https://jupyter.org/)
2. *scikit-learn (https://scikit-learn.org/stable/install.html)
3. *keras (https://keras.io/#installation)
4. uproot (https://pypi.org/project/uproot/)  
5. xgboost (https://xgboost.readthedocs.io/en/release_0.72/build.html)
*I suggest you install anaconda 
(https://www.anaconda.com/distribution/) which will install
all packages 1-3 necessary from python.

You can also run this as a python (.py) file if your prefer
that to a jupyter notebook. To do that, you need to create a
.py file and copy and paste the code into it, then you can
run it using "python <filename>.py".
