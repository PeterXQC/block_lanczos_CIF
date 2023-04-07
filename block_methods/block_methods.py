import numpy as np
import scipy as sp
from .contour import *

def block_lanczos(H,V,k,reorth = 0):
    """
    Input
    -----
    
    H    : d x d matrix
    V    : d x b starting block
    k    : number of iterations
    reorth : how many iterations to apply reorthogonalization on 
    
    Returns
    -------
    Q1k  : First k blocks of Lanczos vectors
    Qkp1 : final block of Lanczos vetors
    A    : diagonal blocks
    B    : off diagonal blocks (incuding block for starting with non-orthogonal block)
    """

    Z = np.copy(V)
    
    d = Z.shape[0]
    if np.shape(Z.shape)[0] == 1:
         b = 1
    else:
        b = Z.shape[1]
    
    A = [np.zeros((b,b),dtype=H.dtype)] * k
    B = [np.zeros((b,b),dtype=H.dtype)] * k
    
    Q = np.zeros((d,b*(k+1)),dtype=H.dtype)

    # B_0 accounts for non-orthogonal V and is not part of tridiagonal matrix
    Q[:,0:b],B_0 = np.linalg.qr(Z)
    for j in range(0,k):
        
#       Qj is the next column of blocks
        Qj = Q[:,j*b:(j+1)*b]

        if j == 0:
            Z = H@Qj
        else:
            Qjm1 = Q[:,(j-1)*b:j*b]
            Z = H @ Qj - Qjm1 @ (B[j-1].conj().T)
     
        A[j] = Qj.conj().T @ Z
        Z -= Qj @ A[j]
        
        # double reorthogonalization if needed
        if reorth > j:
            Z -= Q[:,:j*b]@(Q[:,:j*b].conj().T@Z)
            Z -= Q[:,:j*b]@(Q[:,:j*b].conj().T@Z)
        
        Q[:,(j+1)*b:(j+2)*b],B[j] = np.linalg.qr(Z)
    
    Q1k = Q[:,:b*k]
    Qkp1 = Q[:,b*k:]

    return Q1k, Qkp1, A, B, B_0


def get_block_tridiag(A,B):
    """
    Input
    -----
    
    A  : diagonal blocks
    B  : off diagonal blocks
        Without the first block B[0].
    
    Returns
    -------
    T  : block tridiagonal matrix
    """
    
    q = len(A)
    b = len(A[0])
    
    T = np.zeros((q*b,q*b),dtype=A[0].dtype)

    for k in range(q):
        T[k*b:(k+1)*b,k*b:(k+1)*b] = A[k]

    for k in range(q-1):
        T[(k+1)*b:(k+2)*b,k*b:(k+1)*b] = B[k]
        T[k*b:(k+1)*b,(k+1)*b:(k+2)*b] = B[k].conj().T
    
    return T

def orthTest(Q, b, K, thresh):
    """
    Input
    -----    
    Q       : First K-1 blocks of Lanczos vectors
    b       : block size
    K       : maximum iteration
    thresh  : objective error to test orthogonalization against
    
    Returns
    -------
    """
    if np.max(np.abs(Q.T@Q-np.eye(b*K))) > thresh: 
        raise ValueError("Orthogonality test failed.")

def threeTermTest(H, Q, T, Qkp1, B, b, K, thresh):
    """
    Input
    -----    
    H       : d x d matrix
    Q       : First K-1 blocks of Lanczos vectors
    T       : block tridiagonal matrix
    Qkp1    : final blocks of Lanczos vectors
    B       : off diagonal blocks
    b       : block size
    K       : maximum iteration
    thresh  : objective error to test orthogonalization against
    
    Returns
    -------
    """
    
    E = H@Q-Q@T
    E[:,-b:] -= Qkp1@B[K-1]
    
    if np.max(np.linalg.norm(E,axis=0)) > thresh: 
        raise ValueError("Three-term recurrence test failed.")

def Ei(n, b, i):
    """
    Input
    -----    
    n  : matrix size
    b  : block size
    i  : position of diagonal block (the first block is when i = 1)
    
    Returns
    -------
    Ei  : block zero vector with identity in i-th position
    """
    
    if (i == 0 or i > n/b):
        raise ValueError("Illegal Index: ", i, n, b, n/b)

    Ei = np.zeros((n,b))
    Ei[(i-1)*b:i*b,:] = np.identity(b)
    
    return Ei


def get_Cz(Eval,Evec,z,b,B_0):
    """
    Input
    -----
    Eval : eigevnalues of T
    Evec : eigenvectors of T
    z    : shift z
    b    : block size
    B_0  : first block
    
    Output
    ------
    Cz = -Ek^T(T-zI)^{-1}E_1B_0
    """
    
    K = len(Eval)//b

    Cz = -(Evec[-b:]@((1/(Eval-z))[:,None]*Evec.T[:,:b]))@B_0

    return Cz

def get_CwinvCz(Eval,Evec,z,w,b,B_0):
    """
    Input
    -----
    Eval : eigevnalues of T
    Evec : eigenvectors of T
    z    : shift z
    w    : shift w
    b    : block size
    B_0  : first block
    
    Output
    ------
    CwinvCz = C(w)^{-1}C(z)
    """
        
    Cz = get_Cz(Eval,Evec,z,b,B_0)
    Cw = get_Cz(Eval,Evec,w,b,B_0)

    CwinvCz = np.linalg.solve(Cw,Cz)
    
    return CwinvCz

def H_wz(w,z,λmin,λmax):
    """
    Input
    -----
    w    : shift w
    z    : shift z
    λmin : minimum eigenvalue
    λmax : maximum eigenvalue
    
    Output
    ------
    H_wz = max_{x\in[λmin,λmax]} |x-w|/|x-z|
    """
        
    
    if np.real(z) - w != 0:
        b_hat = ( np.abs(z)**2 - np.real(z)*w ) / (np.real(z) - w)
    else:
        b_hat = np.inf
    
    if λmin < b_hat <= λmax:
        return np.abs((z-w)/np.imag(z))
    return np.max([np.abs((λmax-w)/(λmax-z)), np.abs((λmin-w)/(λmin-z))])


def get_lanf(Eval, Evec, b, B_0, f, Q, k):
    """
    get lanczos iterate for f(A)V
    """
   
    return Q[:,:b*k]@(Evec@(f(Eval)[:,None]*(Evec.T@(Ei(b*k,b,1)@B_0))))
  
    
def get_lan_wLS(Eval, Evec, b, B_0, w, Q, k):
    """
    get lanczos iterate for (H-wI)^{-1}V
    """

    f = lambda x: 1/(x-w)
    
    return get_lanf(Eval, Evec, b, B_0, f, Q, k)
    
    
def trig_ineq_bound_integrand(t, contour, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, Λ, V, Q, k, hnorm):
    """
    Input
    -----
    t       : time variable to integrate along
    contour : contour to integrate along
    angle   : angle(Theta) of the Pacman contour
    r       : radius of the Pacman contour
    Eval    : eigevnalues of T
    Evec    : eigenvectors of T
    b       : block size
    B_0     : first block
    λmin    : minimum eigenvalue
    f       : function to integrat
    Λ       : specturm of matrix H
    V       : d x b block vector
    Q       : First K-1 blocks of Lanczos vectors
    k       : number of iterations
    hnorm   : norm function
    
    Returns
    -------
    trig_ineq_bound : |f(z)| ||err_k(z)|| |dz|
    """
    
    z,dz = contour(t, angle, r, λmin, c)

    lan_zLS = get_lan_wLS(Eval, Evec, b, B_0, z, Q, k)
    
    errz = 1/(Λ-z)[:,None]*V - lan_zLS
    
    trig_ineq_bound = np.abs(f(z))*hnorm(errz)*np.abs(dz)

    return trig_ineq_bound


def a_posteriori_bound_integrand(t, contour, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, λmax):
    """
    Input
    -----
    t       : time variable to integrate along
    contour : contour to integrate along
    angle   : angle(Theta) of the Pacman contour
    r       : radius of the Pacman contour
    Eval    : eigevnalues of T
    Evec    : eigenvectors of T
    b       : block size
    B_0     : first block
    λmin    : minimum eigenvalue
    f       : function to integrate
    w       : shift w
    λmax    : maximum eigenvalue
    
    Returns
    -------
    a_posteriori_bound : |f(z)| ||h_{w, z}(H)||_2 ||C(w)^{-1}C(z)||_2 |dz|
    """
    
    z,dz = contour(t, angle, r, λmin, c)

    CwinvCz = get_CwinvCz(Eval,Evec,z,w,b,B_0)
    
    a_posteriori_bound = np.abs(f(z))*H_wz(w,z,λmin,λmax)*np.linalg.norm(CwinvCz,ord=2)*np.abs(dz)

    return a_posteriori_bound

def get_trig_ineq_bound(pts, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, Λ, V, Q, k, hnorm):
    result = sp.integrate.quad(trig_ineq_bound_integrand, 0, angle, args=(Γ, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, Λ, V, Q, k, hnorm))[0]
    result += sp.integrate.quad(trig_ineq_bound_integrand, 0, 1, args=(Γl, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, Λ, V, Q, k, hnorm), points = pts)[0]
    result /= np.pi
    return result


def get_a_posteriori_bound(pts, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, λmax):
    result = sp.integrate.quad(a_posteriori_bound_integrand,0, angle, args=(Γ, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, λmax))[0]
    result += sp.integrate.quad(a_posteriori_bound_integrand, 0, 1, args=(Γl, angle, r, Eval, Evec, b, B_0, λmin, f, c, w, λmax), points = pts)[0]
    result /= np.pi
    return result

# H-wI
def h_w(Λ, w):
    h_of_H = Λ-w
    return h_of_H

def h_norm(X, Λ, h, *args):
    norm = np.linalg.norm(np.sqrt(h(Λ, *args))[:,None]*X)
    return norm

def get_hnorm(Λ,h):
    """
    Input
    -----
    Λ       : spectrum
    h       : scalar function
    
    Returns
    -------
    induced norm : X -> \|X\|_{h(Λ)}
    """
    
    def h_norm(X):
        return np.linalg.norm(np.sqrt(h(Λ))[:,None]*X)
    
    return h_norm