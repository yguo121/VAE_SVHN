try:    # Works for python 3
    from dataset.dataset import *
    from dataset.dataset_svhn import SVHNDataset
except: # Works for python 2
    from dataset import *
    from dataset_svhn import SVHNDataset