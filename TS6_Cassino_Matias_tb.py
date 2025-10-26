# -*- coding: utf-8 -*-
"""
@author: Matías Cassino
"""
#%% Módulos
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import patches


#%% Funciones

plt.close('all')

def rta_frec(b,a,worN,label):
    w,h=signal.freqs(b,a,worN)
    
    # Gráfico de módulo
    plt.figure()
    plt.semilogx(w, 20*np.log10(abs(h)),label=label)
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    return w,h

def fase(w,h,label):
    phase=np.unwrap(np.angle(h))
    
    # Gráfico de fase
    plt.figure()
    # plt.semilogx(w, np.degrees(phase),label=f_aprox)
    plt.semilogx(w, phase,label=label)
    plt.title('Respuesta de fase')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    return phase

def polos_y_ceros(b,a,axis,label):
    z,p,k=signal.tf2zpk(b, a) 
    
    # Diagrama de polos y ceros
    plt.figure()
    plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{label} Polos')
    axes_hdl = plt.gca()
    if len(z) > 0:
        plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{label} Ceros')
        plt.axhline(0, color='k', lw=0.5)
        plt.axvline(0, color='k', lw=0.5)
        unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                                   color='gray', ls='dotted', lw=2)
        axes_hdl.add_patch(unit_circle)
        axes_hdl.add_patch(unit_circle)
        plt.axis(axis)
        plt.title('Diagrama de Polos y Ceros (plano s)')
        plt.xlabel('σ [rad/s]')
        plt.ylabel('jω [rad/s]')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()    
    return z,p,k

#%% Función transferencia y parámetros
b1=[1,0,9]
a1=[1,np.sqrt(2),1]
worN1=np.logspace(-1,2,1000)
axis1=[-1.1, 1.1, -3.1, 3.1]


b2=[1,0,1/9]
a2=[1,1/5,1]
worN2=np.logspace(-1,2,1000)
axis2=[-1.1, 1.1, -1.1, 1.1]


b3=[1,1/5,1]
a3=[1,np.sqrt(2),1]
worN3=np.logspace(-1,2,1000)
axis3=[-1.1, 1.1, -1.1, 1.1]

# Producto en cascada: T_total(s) = T1(s)*T2(s)
b_total = np.polymul(b1, b2)
a_total = np.polymul(a1, a2)
worN_total = np.logspace(-1, 2, 1000)
axis_total=[-1.1, 1.1, -3.1, 3.1]

label=['T1(S)','T2(S)','T3(S)','T_total(S) = T1(S)*T2(S)']

#%% RTA en frec
w1,h1=rta_frec(b=b1,a=a1,worN=worN1,label=label[0])
w2,h2=rta_frec(b=b2,a=a2,worN=worN2,label=label[1])
w3,h3=rta_frec(b=b3,a=a3,worN=worN3,label=label[2])

#%% RTA de fase
phase1=fase(w=w1,h=h1,label=label[0])
phase2=fase(w=w2,h=h2,label=label[1])
phase3=fase(w=w3,h=h3,label=label[2])

#%% Polos y ceros
z1,p1,k1=polos_y_ceros(b=b1,a=a1,axis=axis1,label=label[0])
z2,p2,k2=polos_y_ceros(b=b2,a=a2,axis=axis2,label=label[1])
z3,p3,k3=polos_y_ceros(b=b3,a=a3,axis=axis3,label=label[2])

#%% Bonus 1
w_total, h_total = rta_frec(b_total, a_total, worN_total, label=label[3])
phase_total=fase(w=w_total,h=h_total,label=label[3])
z_total,p_total,k_total=polos_y_ceros(b=b_total,a=a_total,axis=axis_total,label=label[3])
