# -*- coding: utf-8 -*-
"""
@author: Matías Cassino
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from matplotlib import patches

plt.close('all')

def rta_mod_y_fase(b,a,worN,funcion,fs):
    w,h=sig.freqz(b,a,worN=worN,fs=fs) 
    phase=np.unwrap(np.angle(h)) 
    plt.figure()
    # Magnitud
    plt.subplot(2,1,1)
    #plt.plot(w, 20*np.log10(abs(h)),label=funcion)
    plt.plot(w, abs(h),label=funcion)
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Frecuencia [rad/muestra]')
    plt.ylabel('|H(jω)|')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # Fase
    plt.subplot(2,1,2)
    plt.plot(w, phase,label=funcion)
    plt.title('Fase')
    plt.xlabel('Frecuencia [rad/muestra]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    return w, h, phase

def zpk(b,a,funcion):
    z,p,k=sig.tf2zpk(b,a)
    
    # Diagrama de polos y ceros
    plt.figure()
    plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{funcion} Polos')
    axes_hdl = plt.gca()
    if len(z) > 0:
      plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{funcion} Ceros')
      plt.axhline(0, color='k', lw=0.5)
      plt.axvline(0, color='k', lw=0.5)
      unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                                 color='gray', ls='dotted', lw=2)
      axes_hdl.add_patch(unit_circle)
      plt.axis([-1.1, 1.1, -1.1, 1.1])
      plt.title('Diagrama de Polos y Ceros (plano z)')
      plt.xlabel(r'$\Re(z)$')
      plt.ylabel(r'$\Im(z)$')
      plt.legend()
      plt.grid(True)

    plt.tight_layout()
    plt.show()
    return z, p, k

#%% Coeficientes de las funciones transferencia
b_a=[1,1,1,1]
a_a=[1]

b_b=[1,1,1,1,1]
a_b=[1]

b_c=[1,-1]
a_c=[1]

b_d=[1,0,-1]
a_d=[1]

funcion=['Ta(Z)','Tb(Z)','Tc(Z)','Td(Z)']
# worN=np.logspace(-2,1.9,1000)
worN=None
fs=np.pi*2

#%% RTA de módulo y de fase
w_ta,h_ta,phase_ta=rta_mod_y_fase(b=b_a,a=a_a,worN=worN,funcion=funcion[0],fs=fs)
w_tb,h_tb,phase_tb=rta_mod_y_fase(b=b_b,a=a_b,worN=worN,funcion=funcion[1],fs=fs)
w_tc,h_tc,phase_tc=rta_mod_y_fase(b=b_c,a=a_c,worN=worN,funcion=funcion[2],fs=fs)
w_td,h_td,phase_td=rta_mod_y_fase(b=b_d,a=a_d,worN=worN,funcion=funcion[3],fs=fs)

# #%% Polos y ceros
# z_ta,p_ta,k_ta=zpk(b=b_a,a=a_a,funcion=funcion[0])
# z_tb,p_tb,k_tb=zpk(b=b_b,a=a_b,funcion=funcion[1])
# z_tc,p_tc,k_tc=zpk(b=b_c,a=a_c,funcion=funcion[2])
# z_td,p_td,k_td=zpk(b=b_d,a=a_d,funcion=funcion[3])




