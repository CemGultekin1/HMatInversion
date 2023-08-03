
import logging
from sparse import SparseHierarchicalInversion
import numpy as np
import scipy.sparse as sp
import os


def main():
    n = 2**14
    m = 2**9
    np.random.seed(0)
    mat = sp.eye(n,n,) 
    mat.setdiag(1/2,k = 1)
    mat.setdiag(1/2,k = -1)
    mat.setdiag(1/4,k = 2)
    mat.setdiag(1/4,k = -2)
    mat.setdiag(1/8,k = 3)
    mat.setdiag(1/8,k = -3)
    
    save_dir = 'sparse_mats/example'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    hm = SparseHierarchicalInversion(mat.tocoo(),m,\
                                            tol = 1e-9,\
                                            verbose = True,\
                                            continue_flag=True,\
                                            save_dir = save_dir,\
                                            milestones=[0,0.25,0.5,0.75,0.9, 0.95, 0.99999,1])
    logging.basicConfig(level=logging.INFO,\
                format = ' %(message)s',)
    invmat = hm.invert(inplace = True,)
    err = invmat @ mat  - sp.eye(n)
    err =  np.mean(np.abs(err))
    logging.info(f'err = {err}')
if __name__ == '__main__':
    main()