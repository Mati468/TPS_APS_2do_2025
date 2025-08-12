#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matías Cassino
"""

##% Consigna
"""

En este primer trabajo comenzaremos por diseñar un generador de señales que utilizaremos 
en las primeras simulaciones que hagamos. 
La primer tarea consistirá en programar una función que genere señales
senoidales y que permita parametrizar:

la amplitud máxima de la senoidal (volts)
su valor medio (volts)
la frecuencia (Hz)
la fase (radianes)
la cantidad de muestras digitalizada por el ADC (# muestras)
la frecuencia de muestreo del ADC.
es decir que la función que uds armen debería admitir se llamada de la siguiente manera

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)
Recuerden que tanto xx como tt deben ser vectores de Nx1. Puede resultarte útil el módulo de visualización matplotlib.pyplot donde encontrarán todas las funciones de visualización estilo Matlab. Para usarlo:

plt.plot(tt, xx)

Bonus:

Ser el primero en subir un enlace a tu notebook en esta tarea
Realizar los experimentos que se comentaron en clase. Siguiendo la notación de la función definida más arriba:
ff = 500 Hz
ff = 999 Hz
ff = 1001 Hz
ff = 2001 Hz
Implementar alguna otra señal propia de un generador de señales. 

"""

#%% Modulos
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds
from scipy import signal

# Senoidal
def mi_funcion_sen( vmax, dc, ff, ph, nn, fs):
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    tt = np.linspace(0, (N-1)*ts, N).flatten() # grilla de sampleo temporal
    arg = 2*np.pi*ff*tt + ph # argumento
    xx = vmax*(np.sin(arg) + dc).flatten() # señal
    
    #%% Presentación gráfica de los resultados
    plt.figure()
    plt.plot(tt, xx, label=f'f= {ff}Hz')
    plt.title('Señal: senoidal')
    plt.xlabel('tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid()
    plt.legend()
    plt.show() 
    
    return tt,xx

# Triangular
def mi_funcion_triangular( vmax, dc, ff, ph, nn, fs):
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    tt_aux = np.linspace(0, (N-1)*ts, N).flatten() # grilla de sampleo temporal
    arg = 2*np.pi*ff*tt_aux + ph # argumento
    xx_aux = vmax*(signal.sawtooth(arg) + dc).flatten() # señal
    
    #%% Presentación gráfica de los resultados
    plt.figure()
    plt.plot(tt_aux, xx_aux, label=f'f= {ff}Hz')
    plt.title('Señal: triangular')
    plt.xlabel('tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid()
    plt.legend()
    plt.show() 
    
    return tt_aux,xx_aux

#%% Parámetros
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras

#%% Ejemplo de invocación de las funciones
#tt,xx = mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph=0, nn=N,fs=fs)
#tt_aux,xx_aux = mi_funcion_triangular(vmax = 1, dc = 0, ff = 1, ph=0, nn=N,fs=fs)
   