'''
DLM  2D
------------------------------------------------------------
This code computes unsteady aerodynamic forces acting on
a rigid airfoil due to harmonic pitch and plunge motion.

The aerodynamic forces are computed by solving the Possio
integral equation with a collocation method.
The Possio Kernel is computed using the reformulation due
to Fromme and Golberg (AIAA Journal, Vol. 18, 1980).

A.N. Marques
------------------------------------------------------------
Distributed under the MIT License (see LICENSE.md)
Copyright 2019 Alexandre Marques
'''

import numpy as np
from scipy.special import jv, yv
from scipy.integrate import quad
import time

def aerodynamicForce(Mach, k, xea, nPannels, integrals=None):

    '''
    This function computes aerodynamic forces
    
    Q = [[CL_h, CL_a], [CM_h, CM_a]]
    
    due to harmonic motion of a rigid airfoil in plunge (h)
    and pitch (a).
    Pitch motion occurs about location xea, measured from the
    center of the airfoil in semi-chords.
    
    The aerodynamic forces are computed by solving the Possio integral
    equation using a collocation method.
    The airfoil is divided into "nPannels" pannels.
    Potential doublets are located on the 1/4 chord location
    within each pannel, whereas the integral equation is satisfied
    at the 3/4 chord location within each pannel.
    
    The most expensive part of the solution procedure is computing
    integrals I1, I2, I3, and I4 in the reformulation of Fromme
    and Golberg.
    This implementation speeds up the calculation by pre-computing
    these integrals in a large number of points, and interpolating
    the results.
    If several evaluations are needed at the same Mach number, it is
    recommended to pre-compute these integrals and pass them as
    an extra argument.
    '''


    if integrals is None:
        integrals = defineIntegrals(Mach, k=k)

    Q = np.zeros((2, 2), dtype=complex)

    dx = 2./nPannels
    xi = np.arange(nPannels)*dx + dx/4. - 1.
    x = xi + dx/2.
    
    y = np.repeat(np.reshape(x, (nPannels, 1)), nPannels, axis=1) - np.repeat(np.reshape(xi, (1, nPannels)), nPannels, axis=0)
    D = PossiosKernel(y, Mach, k, integrals)

    wok = -1j*np.ones(nPannels)
    p = 2.*np.linalg.solve(D, wok)
    Q[0, 0] = 0.5*np.sum(p)
    Q[1, 0] = -0.25*np.dot(xi - xea, p)

    wok = -(1./k + 1j*(x - xea))
    p = 2.*np.linalg.solve(D, wok)
    Q[0, 1] = 0.5*np.sum(p)
    Q[1, 1] = -0.25*np.dot(xi - xea, p)

    return Q


def PossiosKernel(x, mach, k, integrals=None):
    '''
    Possios kernel reformulated by Fromme and Golberg, 1980
    '''

    if integrals is None:
        integrals = defineIntegrals(Mach, k=k)
        
    I1, I2, I3, I4 = integrals(k*x)
    
    M2 = Mach**2
    beta2 = 1. - M2
    beta = np.sqrt(beta2)
    y = Mach*k*x/beta2 + 0.*1j
    
    j0y = jv(0, y)
    j1y = jv(1, y)
    g0y = g0(y)
    g1y = g1(y)
    e = np.exp(1j*k*x/beta2)
    c = 1j*0.5*np.pi + np.log(0.5*Mach*k/beta2)
    
    F1 = 1. + M2*(j0y*e - 1.) - 1j*Mach*j1y*e - Mach*k*x*I1
    F2 = 1j*k*(np.log(k) + np.euler_gamma + 1j*0.5*np.pi + I2)
    F2 += 1j*k*e*( (np.exp(-1j*k*M2*x/beta2) - 1.)/(1j*k*x) + (M2/beta2)*(c + np.euler_gamma)*j0y - 1j*(Mach/beta2)*c*j1y + (g0y + 1j*Mach*g1y)/beta2 )
    F2 -= 1j*( (k**2)*x/beta2 )*( Mach*(c + np.euler_gamma)*I1 + 1j*I3 )
    F2 += 1j*( Mach*(k**2)*x/beta2 )*I4
    F2 += 1j*(k/beta)*( (M2/(1. + beta))*np.log(2.*beta2/Mach) + np.log(0.5*(1. + beta)/beta2) )
    
    K1 = -1j*(k/beta2)*np.exp(-1j*k*x)*F1
    K2 = -np.exp(-1j*k*x)*F2
    A = np.divide(np.ones_like(x), x) + K1*np.log(np.abs(x)) + K2
    
    return -A*beta/(2.*np.pi*k)
    
    
def g0(z):
    '''
    Function G0 (Fromme and Golberg, 1980)
    '''
    
    z = z + 0.*1j
    return 0.5*np.pi*yv(0, z) - (np.euler_gamma + np.log(0.5*z))*jv(0, z)


def g1(z):
    '''
    Function G1 (Fromme and Golberg, 1980)
    '''

    z = z + 0.*1j
    return -0.5*np.pi*yv(1, z) - 1./z + np.log(0.5*z)*jv(1, z)


def PoissosIntegrals(Mach, kx):
    '''
    Computation of integrals I1, I2, I3, and I4 (Fromme and Golberg, 1980)
    '''

    M2 = Mach**2
    beta2 = 1. - M2
    y = kx/beta2 + 0.*1j

    def complexQuad(f, weight=None, wvar=None):
        def fr(u):
            return np.real(f(u))
        def fi(u):
            return np.imag(f(u))
        intReal = quad(fr, 0., 1., weight=weight, wvar=wvar)
        intImag = quad(fi, 0., 1., weight=weight, wvar=wvar)
        return intReal[0] + 1j*intImag[0]

    I = np.zeros((kx.size, 4), dtype=complex)
    for i in range(kx.size):
        def f1(u):
            return np.exp(1j*y[i]*u)*jv(1, Mach*y[i]*u)
        def f2(u):
            return (np.exp(1j*y[i]*u) - 1.)*jv(0, Mach*y[i]*u)/u
        def f3(u):
            return np.exp(1j*y[i]*u)*g0(Mach*y[i]*u)
        def f4(u):
            return (np.exp(1j*y[i]*u) - 1.)*jv(1, Mach*y[i]*u)
    
        I[i, 0] = complexQuad(f1)
        I[i, 1] = complexQuad(f2)
        I[i, 2] = complexQuad(f3)
        I[i, 3] = complexQuad(f4, weight='alg-loga', wvar=(0., 0.))

    return I

    
def defineIntegrals(Mack, k=2., n=3000):
    '''
    Pre-computes integrals I1, I2, I3, and I4 (Fromme and Golberg, 1980)
    By default, reduced frequencies up to k=2 are considered.
    The user can change this range
    '''

    kx = np.linspace(-2.*k, 2.*k, n)
    I = PoissosIntegrals(Mach, kx)

    def integrals(y):
        I1 = np.interp(y, kx, I[:, 0])
        I2 = np.interp(y, kx, I[:, 1])
        I3 = np.interp(y, kx, I[:, 2])
        I4 = np.interp(y, kx, I[:, 3])
        return I1, I2, I3, I4
        
    return integrals
