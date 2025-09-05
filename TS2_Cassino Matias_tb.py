# -*- coding: utf-8 -*-
"""
@author: Matías Cassino
"""
##% Consigna
"""
1) Dada la siguiente ecuación en diferencias que modela un sistema LTI:

y[n]=3⋅10−2⋅x[n]+5⋅10−2⋅x[n−1]+3⋅10−2⋅x[n−2]+1.5⋅y[n−1]−0.5⋅y[n−2]

A) Graficar la señal de salida para cada una de las señales de entrada que generó en el TS1. Considere que las mismas son causales.
B) Hallar la respuesta al impulso y usando la misma, repetir la generación de la señal de salida para alguna de las señales de entrada consideradas en el punto anterior.
C) En cada caso indique la frecuencia de muestreo, el tiempo de simulación y la potencia o energía de la señal de salida.

2) Hallar la respuesta al impulso y la salida correspondiente a una señal de entrada senoidal en los sistemas definidos mediante las siguientes ecuaciones en diferencias:

y[n]=x[n]+3⋅x[n−10]
y[n]=x[n]+3⋅y[n−10]

Bonus

3) Discretizar la siguiente ecuación diferencial correspondiente al modelo de Windkessel que describe la dinámica presión-flujo del sistema sanguíneo:

C*dP/dt+1/R*P=Q

Considere valores típicos de Compliance y Resistencia vascular

"""

#%% Modulos
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds
from scipy import signal

#%% Funciones
plt.close("all")

# Senoidal
def mi_funcion_sen(vmax, dc, ff, ph, N, fs, plot=True):
    
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    tt = np.linspace(0, (N-1)*ts, N).flatten() # grilla de sampleo temporal
    arg = 2*np.pi*ff*tt + ph # argumento
    xx = (vmax*(np.sin(arg)) + dc).flatten() # señal
    pot = (1/N)*np.sum(xx**2)
    
    if plot:
        
        #Presentación gráfica de los resultados
        plt.figure()
        plt.plot(tt, xx, label=f"f = {ff} Hz\nN = {N}\nTs = {ts} s\nPotencia = {pot:.3f} W")
        plt.title('Señal: senoidal')
        plt.xlabel('tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.grid()
        plt.xlim([tt.min() - 0.1*(tt.max()-tt.min()), tt.max() + 0.1*(tt.max()-tt.min())])
        plt.ylim([xx.min() - 0.1*(xx.max()-xx.min()), xx.max() + 0.1*(xx.max()-xx.min())])
        plt.legend()
        plt.show() 
        
    return tt,xx

# Cuadrada
def funcion_cuadrada(vmax, dc, ff, ph, N, fs, plot=True):
    
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    tt = np.linspace(0, (N-1)*ts, N).flatten() # grilla de sampleo temporal
    arg = 2*np.pi*ff*tt + ph # argumento
    xx = (vmax*(signal.square(arg)) + dc).flatten() # señal
    pot = (1/N)*np.sum(xx**2)
    
    if plot:
    
        #Presentación gráfica de los resultados
        plt.figure()
        plt.plot(tt, xx, label  = f"N = {N}\nTs = {ts} s\nPotencia = {pot:.3f} W")
        plt.title('Señal: cuadrada')
        plt.xlabel('tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.grid()
        plt.xlim([tt.min() - 0.1*(tt.max()-tt.min()), tt.max() + 0.1*(tt.max()-tt.min())])
        plt.ylim([xx.min() - 0.1*(xx.max()-xx.min()), xx.max() + 0.1*(xx.max()-xx.min())])
        plt.legend()
        plt.show() 
        
    return tt,xx

# Pulso rectangular
def funcion_pulso_rectangular(vmax, dc, t_inicio, duracion, N, fs, plot=True):
    
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    tt = np.linspace(0, (N-1)*ts, N).flatten() # grilla de sampleo temporal
    xx=(vmax*(np.heaviside(tt - t_inicio, 1) - np.heaviside(tt - (t_inicio + duracion), 1)) + dc).flatten() #señal con valor 1 en el salto
    pot = (1/N)*np.sum(xx**2)
    
    if plot:
            
        #Presentación gráfica de los resultados
        plt.figure()
        plt.plot(tt*10, xx, label  = f"N = {N}\nTs = {ts} s\nPotencia = {pot:.3f} W")
        plt.title('Señal: pulso rectangular')
        plt.xlabel('tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.grid()
        #plt.xlim([tt.min() - 0.1*(tt.max()-tt.min()), tt.max() + 0.1*(tt.max()-tt.min())])
        #plt.ylim([xx.min() - 0.1*(xx.max()-xx.min()), xx.max() + 0.1*(xx.max()-xx.min())])
        plt.legend()
        plt.show()    
    return tt,xx

# Senoidal modulada
def sen_modulada(vmax_1, vmax_2, dc_1, dc_2, ff_1, ff_2, ph_1, ph_2, N, fs, plot=True):
    
    # Datos generales de la simulación
    tt_1,xx_1 = mi_funcion_sen(vmax = vmax_1, dc = dc_1, ff = ff_1, ph = ph_1, N=N,fs=fs, plot=False)
    tt_2,xx_2 = mi_funcion_sen(vmax = vmax_2, dc = dc_2, ff = ff_2, ph = ph_2, N=N,fs=fs, plot=False)
    
    tt_aux=tt_1
    xx_aux = xx_1*xx_2
    pot_mod = (1/N)*np.sum(xx_aux**2)  
    
    if plot:
    
        #Presentación gráfica de los resultados
        plt.figure()
        plt.plot(tt_1, xx_aux, label  = f"N = {N}\nTs = {1/fs} s\nPotencia = {pot_mod:.3f} W")
        plt.title('Señal: senoidal de 2KHz modulada por senoidal de 1KHz')
        plt.xlabel('tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.legend()
        plt.xlim([tt_aux.min() - 0.1*(tt_aux.max()-tt_aux.min()), tt_aux.max() + 0.1*(tt_aux.max()-tt_aux.min())])
        plt.ylim([xx_aux.min() - 0.1*(xx_aux.max()-xx_aux.min()), xx_aux.max() + 0.1*(xx_aux.max()-xx_aux.min())])
        plt.grid()
        plt.show() 
        
    return tt_aux,xx_aux

def potencia_modificada(xx, tt, N, fs, porcentaje, plot=True):
    pot=(1/N)*np.sum(xx**2)*(porcentaje/100)
    xx_aux=np.clip(xx,-pot,pot)
    tt_aux=tt
    
    if plot:
        
        #Presentación gráfica de los resultados
        plt.figure()
        plt.plot(tt_aux, xx_aux, label  = f"N = {N}\nTs = {1/fs} s\nPotencia = {pot} W")
        plt.title(f'Señal recortada al {porcentaje}% de su potencia')
        plt.xlabel('tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.legend()
        plt.grid()
        plt.show()
    
    return tt_aux, xx_aux

def ec_diferencias_coef(coef_x,coef_y,N,fs,señal, title,y_label="y",plot=True):
    
    # Salida del sistema
    y = signal.lfilter(coef_x, coef_y, señal)
    n = np.arange(N)
    t_simulacion=N*(1/fs)
    pot = (1/N)*np.sum(y**2)
    
    if plot: 
        plt.figure()
        plt.plot(n, y,'--x',label  = f"fs = {fs} Hz\nTiempo de simulación = {t_simulacion:.3f} s\nPotencia de {y_label}[n]= {pot:.3g} W")
        plt.title(title +' por lfilter')
        plt.xlabel("n")
        plt.ylabel(y_label + "[n]")
        plt.xlim([n.min() - 0.1*(n.max()-n.min()), n.max() + 0.1*(n.max()-n.min())])
        plt.ylim([y.min() - 0.1*(y.max()-y.min()), y.max() + 0.1*(y.max()-y.min())])
        plt.legend()
        plt.grid()
        plt.show()
      
    return y

def salida_rta_impulso_conv(coef_x,coef_y,N,fs,señal, plot_y=True, plot_h=True):
    
    delta=signal.unit_impulse(N)
    h = signal.lfilter(coef_x, coef_y, delta) #Respuesta al impulso
    y=np.convolve(señal, h)[:N] #Salida ante respuesta al impulso truncada a la longitud de N ya que tiene el tamaño de señal+delta-1
    
    n = np.arange(N)
    t_simulacion=N*(1/fs)
    
    pot = (1/N)*np.sum(y**2)
    pot_h = (1/N)*np.sum(h**2)
    
    if plot_h: 
        
        plt.figure()
        plt.plot(n, delta)
        plt.legend()
        plt.grid()
        plt.show()
        
        
        plt.figure()
        plt.plot(n, h,'--+',label  = f"fs = {fs} Hz\nTiempo de simulación = {t_simulacion:.3f} s\nPotencia de h[n]= {pot_h:.3g} W")
        plt.title("Respuesta al impulso del sistema LTI por lfilter")
        plt.xlabel("n")
        plt.ylabel("h[n]")
        plt.xlim([n.min() - 0.1*(n.max()-n.min()), n.max() + 0.1*(n.max()-n.min())])
        plt.ylim([h.min() - 0.1*(h.max()-h.min()), h.max() + 0.1*(h.max()-h.min())])
        plt.legend()
        plt.grid()
        plt.show()
    
    if plot_y: 
        plt.figure()
        plt.plot(n, y,'--+',label  = f"fs = {fs} Hz\nTiempo de simulación = {t_simulacion:.3f} s\nPotencia = {pot:.3g} W")
        plt.title("Salida sistema LTI por convolución")
        plt.xlabel("n")
        plt.ylabel("y[n]")
        plt.xlim([n.min() - 0.1*(n.max()-n.min()), n.max() + 0.1*(n.max()-n.min())])
        plt.ylim([y.min() - 0.1*(y.max()-y.min()), y.max() + 0.1*(y.max()-y.min())])
        plt.legend()
        plt.grid()
        plt.show()
    
    return y,h

# def ec_diferencias(x,N):
    
#     # Inicialización de la salida
#     y = np.zeros(N)
#     n = np.arange(N)
    
#     for i in range(2, N):
#         y[i] = (0.03*x[i] + 0.05*x[i-1] + 0.03*x[i-2] + 1.5*y[i-1] - 0.5*y[i-2])

#     plt.figure()
#     plt.stem(n, y)
#     plt.title("Respuesta del sistema LTI")
#     plt.xlabel("n")
#     plt.ylabel("y[n]")
#     plt.xlim([n.min() - 0.1*(n.max()-n.min()), n.max() + 0.1*(n.max()-n.min())])
#     plt.ylim([y.min() - 0.1*(y.max()-y.min()), y.max() + 0.1*(y.max()-y.min())])
#     plt.legend()
#     plt.grid()
#     plt.show()
    
#     return y

def euler(Q, N, fs, R, C, P0):
    ts=1/fs
    n = np.arange(N)
    P = np.zeros(N)
    P[0]=P0
    for i in range(1,N):
        P[i] = P[i-1]*(1-(ts/(R*C))) + Q[i-1]*ts/C
        
    plt.figure()
    plt.suptitle("Modelo Windkessel por método de Euler")
    plt.subplot(2,1,1)
    plt.plot(n, P, '--+', label="Presión P[n]")
    plt.xlabel("Índice n")
    plt.ylabel("P[n] [mmHg]")
    plt.ylim([P.min() - 0.1*(P.max()-P.min()), P.max() + 0.1*(P.max()-P.min())])
    plt.grid()
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(n, Q, '--*', label="Flujo Q[n]")
    plt.xlabel("Índice n")
    plt.ylabel("Q[n] [mL/s]")
    plt.ylim([Q.min() - 0.1*(Q.max()-Q.min()), Q.max() + 0.1*(Q.max()-Q.min())])
    plt.grid()
    plt.legend()
    
    #plt.xlim([n.min() - 0.1*(n.max()-n.min()), n.max() + 0.1*(n.max()-n.min())])
    plt.tight_layout()
    plt.show()
    
    return P
    
#%% Parámetros
# Punto 1
fs = 50000 # frecuencia de muestreo (Hz)
N = 750   # cantidad de muestras
coef_x= [0.03, 0.05, 0.03]
coef_y= [1, -1.5, 0.5]
# Punto 2
N_2=1000
fs_2=1000
coef_x_I= [1] + [0]*9 + [3]
coef_y_I= [1]
coef_x_II= [1]
coef_y_II= [1] + [0]*9 + [-3]
# Punto 3
C= 2 #mL/mmHg
R=1.145 #mmHg.s/mL
P0=95 #mmHg

#%% Invocación de las funciones del punto 1A (y 1C implícito)
tt_a,xx_a = mi_funcion_sen(vmax = 1, dc = 0, ff = 2000, ph=0, N=N,fs=fs, plot=None)

tt_b,xx_b = mi_funcion_sen(vmax = 2, dc = 0, ff = 2000, ph=np.pi/2, N=N,fs=fs, plot=None)

tt_c,xx_c = sen_modulada(vmax_1 = 1, vmax_2 = 1, dc_1 = 0, dc_2 = 0, ff_1 = 2000, ff_2 = 1000, ph_1 = 0, ph_2 = 0, N=N,fs=fs, plot=None)

tt_d,xx_d = potencia_modificada(xx = xx_a, tt = tt_a, N=N, fs=fs, porcentaje = 75, plot=None)

tt_e,xx_e = funcion_cuadrada(vmax = 1, dc = 0, ff = 4000, ph=0, N=N,fs=fs, plot=None)

tt_f,xx_f = funcion_pulso_rectangular(vmax = 1, dc = 0, t_inicio=0.002, duracion=0.001, N=N,fs=fs, plot=None)

# ec_diferencias(xx_a,N)

salida_a=ec_diferencias_coef(coef_x, coef_y, N, fs, xx_a,title="Salida del sistema LTI para senoidal")
salida_b=ec_diferencias_coef(coef_x, coef_y, N, fs, xx_b,title="Salida del sistema LTI para senoidal ampliada y desfasada")
salida_c=ec_diferencias_coef(coef_x, coef_y, N, fs, xx_c,title="Salida del sistema LTI para senoidal modulada")
salida_d=ec_diferencias_coef(coef_x, coef_y, N, fs, xx_d,title="Salida del sistema LTI para senoidal recortada en potencia")
salida_e=ec_diferencias_coef(coef_x, coef_y, N, fs, xx_e,title="Salida del sistema LTI para señal cuadrada")
salida_f=ec_diferencias_coef(coef_x, coef_y, N, fs, xx_f,title="Salida del sistema LTI para señal pulso rectangular")

#%% Invocación de las funciones del punto 1B (y 1C implícito)
salida_impulso_a,h_impulso_a=salida_rta_impulso_conv(coef_x, coef_y, N, fs, xx_a,plot_y=True)
salida_impulso_b,h_impulso_b=salida_rta_impulso_conv(coef_x, coef_y, N, fs, xx_b,plot_y=True,plot_h=None)
salida_impulso_c,h_impulso_c=salida_rta_impulso_conv(coef_x, coef_y, N, fs, xx_c,plot_y=True,plot_h=None)
salida_impulso_d,h_impulso_d=salida_rta_impulso_conv(coef_x, coef_y, N, fs, xx_d,plot_y=True,plot_h=None)
salida_impulso_e,h_impulso_e=salida_rta_impulso_conv(coef_x, coef_y, N, fs, xx_e,plot_y=True,plot_h=None)
salida_impulso_f,h_impulso_f=salida_rta_impulso_conv(coef_x, coef_y, N, fs, xx_f,plot_y=True,plot_h=None)

# Comparación punto 1A y 1B
print(f"La convolución y la ecuación en diferencias de A tienen la misma salida?: {np.allclose(salida_a, salida_impulso_a)}")
print(f"La convolución y la ecuación en diferencias de B tienen la misma salida?: {np.allclose(salida_b, salida_impulso_b)}")
print(f"La convolución y la ecuación en diferencias de C tienen la misma salida?: {np.allclose(salida_c, salida_impulso_c)}")
print(f"La convolución y la ecuación en diferencias de D tienen la misma salida?: {np.allclose(salida_d, salida_impulso_d)}")
print(f"La convolución y la ecuación en diferencias de E tienen la misma salida?: {np.allclose(salida_e, salida_impulso_e)}")
print(f"La convolución y la ecuación en diferencias de F tienen la misma salida?: {np.allclose(salida_f, salida_impulso_f)}")

#%% Invocación de las funciones del punto 2
tt_I,xx_2 = mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph=0, N=N_2,fs=fs_2, plot=None)
delta_aux=signal.unit_impulse(N_2)

h_I=ec_diferencias_coef(coef_x_I,coef_y_I,N_2,fs_2,delta_aux,y_label="h",title="Respuesta al impulso del sistema I LTI")
h_II=ec_diferencias_coef(coef_x_II,coef_y_II,N_2,fs_2,delta_aux,y_label="h",title="Respuesta al impulso del sistema II LTI")
y_I=ec_diferencias_coef(coef_x_I,coef_y_I,N_2,fs_2,xx_2,title="Salida del sistema I LTI")
y_II=ec_diferencias_coef(coef_x_II,coef_y_II,N_2,fs_2,xx_2,title="Salida del sistema II LTI")

#%% Invocación de la función del Bonus (punto 3)
tt_q,xx_q = mi_funcion_sen(vmax = 2.5, dc = 83, ff = 1.333, ph=0, N=N_2,fs=fs_2,plot=None)

presion=euler(xx_q, N=N_2, fs=fs_2, R=R, C=C, P0=P0)
