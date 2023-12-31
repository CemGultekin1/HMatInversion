import logging
from typing import Any, Tuple
import numpy as np
import time
class BinaryCounter:
    def __init__(self, numdigs:int,) :
        self.index= 0 
        self.numdigs =numdigs
    @property
    def binary(self,):
        x = format(self.index,'b')
        zs = self.numdigs - len(x)
        return "0"*zs + x
    @property
    def array(self,):
        bin = self.binary
        return tuple([int(x) for x in bin])
    def __str__(self,):
        return self.binary
    def increment(self,):
        self.index+= 1
        self.index = self.index % 2**self.numdigs
    def decrement(self,):
        self.index -= 1

class Reporter:
    def __init__(self,message:str = '',significance = 2.) -> None:
        self.message = message
        self.header_length = 0
        self.t0 = 0
        self.times = []
        self.significance = significance
    def take_message(self,msg:str):
        self.message= msg        
    def msg_print(self,):
        self.header_length = np.maximum(len(self.message),self.header_length)
        # logging.info(self.message)
        self.t0 = time.time()
    def time_print(self,):
        t1 = time.time()
        dt = t1 - self.t0
        self.times.append(dt)
        avgtime = sum(self.times)/len(self.times)
        formatter = "{:.2e}"
        if dt/avgtime > self.significance:
            dtmess = self.message + ' '*(self.header_length - len(self.message))  + f'\t\tdt = {formatter.format(dt)}, avgdt = {formatter.format(avgtime)}'
            logging.info(dtmess)
reporter = Reporter()
def report_decorator(fun):
    def wrapped_fun(self, *args: Any, **kwds: Any) -> Any:
        vb = self.__dict__.get('verbose',False)
        
        if vb:
            reporter.take_message(self.message_writer(fun.__name__,args[0]))
            reporter.msg_print()
        outputs =  fun.__call__(self,*args, **kwds)
        if vb:
            reporter.time_print()
        return outputs
    return wrapped_fun
class HierarchicalMatrixInverter:
    invertible_size:int
    size:int
    nlevels:int
    mat:np.ndarray
    def __init__(self,mat:np.ndarray,invertible_size:int,verbose:bool = False) -> None:
        size = mat.shape[0]
        self.invertible_size = invertible_size
        self.nlevels = int(np.log2(size//invertible_size))
        self.mat = mat
        self.verbose = verbose
        self.pows = np.power(2,np.arange(self.nlevels)[::-1])
    def arr2slc(self,arr:Tuple[int,...]):
        ind =  self.pows[:len(arr)] @ arr * self.invertible_size
        ind1 = ind + self.pows[len(arr)-1] *  self.invertible_size
        return slice(ind,ind1)
    def message_writer(self,fun_name:str,arr:Tuple[int,...]):
        return f'{fun_name}(' + ''.join([str(x) for x in arr])+ '0'*(self.nlevels - len(arr)) + ')'
        
    def __getitem__(self,arr:Tuple[int,...]):
        if not bool(arr):
            return self.mat
        slc = self.arr2slc(arr)
        return self.mat[slc,slc]
    def __setitem__(self,arr:Tuple[int,...],mat):
        if not bool(arr):
            self.mat = mat
            return
        slc = self.arr2slc(arr)
        self.mat[slc,slc] = mat
    def divide(self,mat):
        hside = mat.shape[0]//2
        s0,s1 = slice(0,hside),slice(hside,2*hside)
        m00 = mat[s0,s0]
        m01 = mat[s0,s1]
        m10 = mat[s1,s0]
        m11 = mat[s1,s1]
        return m00,m01,m10,m11

    def act(self,arr:Tuple[int,...]):
        if arr[-1] == 0:
            self.m_a_placement(arr)
        else:
            while bool(arr):
                if arr[-1] == 1:
                    self.complete_inversion(arr)
                    arr = arr[:-1]
                else:
                    self.m_a_placement(arr)
                    break
                
    @staticmethod
    def invert_mat(mat):
        return np.linalg.inv(mat)
    @report_decorator
    def m_a_placement(self,arr:Tuple[int,...]):
        '''
        at 'level'
        mat[ind,ind] <- A^{-1}
        mat[ind+1,ind+] <- (D - C@A^{-1}@B)^{-1}
        '''
        mat = self[arr[:-1]]
        ainv,b,c,d = self.divide(mat)
        level = len(arr)
        if level == self.nlevels: 
            ainv = self.invert_mat(ainv)
            self[arr] = ainv
        m_a = d - c@ ainv @ b
        if level == self.nlevels : 
            m_a = self.invert_mat(m_a)
        arr0 = list(arr)
        arr0[-1] = 1
        self[tuple(arr0)] = m_a
        
    @staticmethod
    def merge_quad(a,b,c,d):
        mother_inverse = np.block([[a,b],[c,d]])
        return mother_inverse
    @report_decorator
    def complete_inversion(self,arr:Tuple[int,...]):
        mat = self[arr[:-1]]
        ainv,b,c,m_a_inv = self.divide(mat)

            
        new_a = ainv + ainv @ b @ m_a_inv @ c @ ainv
        new_b = - ainv @ b @ m_a_inv
        new_c = - m_a_inv @ c @ ainv
        new_d = m_a_inv
        
        self[arr[:-1]] = self.merge_quad(new_a,new_b,new_c,new_d)
        
    def invert(self,inplace:bool = True):
        if not inplace:
            self.mat = self.mat.copy()
        bc = BinaryCounter(self.nlevels)
        self.act(bc.array)
        bc.increment()
        while bc.index !=  0:
            self.act(bc.array)
            bc.increment()
        return self.mat
    
class SizeChagingHierarchicalInversion(HierarchicalMatrixInverter):
    def __init__(self, mat: np.ndarray, invertible_size: int) -> None:
        self.org_size = mat.shape[0]
        mat = self.extend(mat,invertible_size)
        super().__init__(mat, invertible_size)
    @staticmethod
    def get_expansion_size(mat,invsize,):
        k = np.log2(mat.shape[0]) - np.log2(invsize) - 1
        k = int(np.ceil(k))
        expansion = (2**(k+1))*invsize - mat.shape[0]
        return expansion
    @staticmethod
    def extend(mat,invsize):
        expansion = SizeChagingHierarchicalInversion.get_expansion_size(mat,invsize)
        if expansion == 0:
            return mat
        z = np.zeros((mat.shape[0],expansion))
        e = np.eye(expansion)
        return np.block(
            [[mat,z],[z.T,e]]
        )
    @staticmethod
    def submatrix(mat,arr0,arr1):
        arr0 = slice(arr0[0],arr0[-1] + 1)
        arr1= slice(arr1[0],arr1[-1] + 1)
        return mat[arr0,arr1]
    def invert(self,**kwargs):
        mat =  super().invert(**kwargs)
        arr = np.arange(self.org_size)
        return self.submatrix(mat,arr,arr)
def main():
    n = 2**8
    m = 2**1
    np.random.seed(0)
    mat = np.random.randn(n,n)
    mat = mat.T@mat + np.eye(n) * 1e-3
    hm = SizeChagingHierarchicalInversion(mat,m)
    logging.basicConfig(level=logging.INFO,\
                format = '%(message)s',)
    invmat = hm.invert(inplace = False)
    err = invmat @ mat  - np.eye(n)
    err =  np.mean(np.abs(err))
    logging.info(f'err = {err}')
if __name__ == '__main__':
    main()