from __future__ import division
from numpy.linalg import lstsq,eig
from numpy import cov,dot,arange,c_
import numpy as np    

def cca(x_tn,y_tm, reg=0.00000001):
    x_tn = x_tn-x_tn.mean(axis=0)
    y_tm = y_tm-y_tm.mean(axis=0)
    N = x_tn.shape[1]
    M = y_tm.shape[1]
    xy_tq = c_[x_tn,y_tm]
    cqq = cov(xy_tq,rowvar=0)
    cxx = cqq[:N,:N]+reg*np.eye(N)+0.000000001*np.ones((N,N))
    cxy = cqq[:N,N:(N+M)]+0.000000001*np.ones((N,N))
    cyx = cqq[N:(N+M),:N]+0.000000001*np.ones((N,N))
    cyy = cqq[N:(N+M),N:(N+M)]+reg*np.eye(N)+0.000000001*np.ones((N,N))
    
    K = min(N,M)
    
    xldivy = lstsq(cxx,cxy)[0]
    yldivx = lstsq(cyy,cyx)[0]
    #print xldivy
    #print dot(np.linalg.inv(cxx),cxy)
    _,vecs = eig(dot(xldivy,yldivx))
    a_nk = vecs[:,:K]
    #print normr(vecs.T)
    b_mk = dot(yldivx,a_nk)
    
    u_tk = dot(x_tn,a_nk)
    v_tk = dot(y_tm,b_mk)
    
    return a_nk,b_mk,u_tk,v_tk
    
def normr(a):
    return a/np.sqrt((a**2).sum(axis=1))[:,None]

def test_cca():
    x_tn = 1/np.arange(1,31).reshape(6,5)
    y_tm = 1/np.arange(1,19).reshape(6,3)
    x_tn=np.random.random((10000,50))
    y_tm=np.random.random((10000,50))

    a,b,u,v = cca(x_tn,y_tm)
    print normr(a)
    print x_tn
    
if __name__ == "__main__":
    test_cca()
