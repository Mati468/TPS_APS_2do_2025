# -*- coding: utf-8 -*-
"""
@author: Matías Cassino
"""
##% Consigna
"""
En esta tarea semanal analizaremos un fenómeno muy particular que se da al calcular la DFT, el efecto de desparramo espectral.  

Luego, haremos el siguiente experimento:

    - Senoidal de frecuencia f0=k0∗fS/N=k0.Δf
    - Potencia normalizada, es decir energía (o varianza) unitaria

1. Se pide:

a) Sea k0
 
    - N4
 
    - N4+0.25
 
    - N4+0.5
 
Notar que a cada senoidal se le agrega una pequeña desintonía respecto a  Δf.
a) Graficar las tres densidades espectrales de potencia (PDS's) y discutir cuál es el efecto de dicha desintonía en el espectro visualizado.

b) Verificar la potencia unitaria de cada PSD, puede usar la identidad de Parseval. 
En base a la teoría estudiada. Discuta la razón por la cual una señal senoidal tiene un espectro tan diferente respecto a otra de muy pocos Hertz de diferencia. 

c) Repetir el experimento mediante la técnica de zero padding. Dicha técnica consiste en agregar ceros al final de la señal para aumentar Δf
 de forma ficticia. Probar agregando un vector de 9*N ceros al final. Discuta los resultados obtenidos.

Bonus
2. Calcule la respuesta en frecuencia de los sistemas LTI de la TS2.
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

def normalizacion(x):
    media_x=np.mean(x) #media
    desvio_x=np.std(x) #desvio
    xx_norm=(x-media_x)/desvio_x #señal normalizada
    varianza_x=np.var(xx_norm) #varianza
    
    return xx_norm,varianza_x

def densidad_potencia(x):
    densidad_pot_x=fft_abs_func(x)**2 # Densidad de potencia
    
    return densidad_pot_x

def parseval(x,densidad_potencia_x,N):
    energia_t_x=np.sum(np.abs(x)**2)/N
    potencia_f_x=(1/N**2)*np.sum(densidad_potencia_x)
    
    if np.isclose(energia_t_x, potencia_f_x, rtol=1e-6):  # comparación con tolerancia numérica
        print(f"Se cumple la identidad de Parseval: Energía = {energia_t_x:.4f} = Densidad espectral de potencia = {potencia_f_x:.4f}")
    else:
        print(f"No se cumple la identidad de Parseval: Energía = {energia_t_x:.4f} != Densidad espectral de potencia = {potencia_f_x:.4f}")
        
        
    return energia_t_x,potencia_f_x

def zero_padding_fft_abs(M,N,x_norm):
    x_zp=np.zeros(M)
    x_zp[0:N]=x_norm
    
    return x_zp

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
    
def fft_abs_func(x):
    fft_x=np.fft.fft(x) # FFT
    fft_x_abs=np.abs(fft_x) # Módulo de la FFT
    
    return fft_x_abs

#%% Parámetros
# Punto 1
N=1000
fs=1000
df=fs/N
k=np.arange(N)*df
k0_1=N/4
k0_2=(N/4)+0.25
k0_3=(N/4)+0.5
M=10*N
k_M=np.arange(M)*df

# Bonus (punto 2)
# Sistema 1
coef_x_I= [0.03, 0.05, 0.03]
coef_y_I= [1, -1.5, 0.5]
# Sistema 2
coef_x_II= [1] + [0]*9 + [3]
coef_y_II= [1]
# Sistema 3
coef_x_III= [1]
coef_y_III= [1] + [0]*9 + [-3]

#%% Invocación de las funciones del punto 1
tt_1,xx_1 = mi_funcion_sen(vmax = 1, dc = 0, ff = k0_1*df, ph=0, N=N,fs=fs,plot=None)
tt_2,xx_2 = mi_funcion_sen(vmax = 1, dc = 0, ff = k0_2*df, ph=0, N=N,fs=fs,plot=None)
tt_3,xx_3 = mi_funcion_sen(vmax = 1, dc = 0, ff = k0_3*df, ph=0, N=N,fs=fs,plot=None)

# Normalización de la señal
xx_1_norm,varianza_1=normalizacion(xx_1)
xx_2_norm,varianza_2=normalizacion(xx_2)
xx_3_norm,varianza_3=normalizacion(xx_3)

# Densidad de potencia
densidad_pot_1=densidad_potencia(xx_1_norm) 
densidad_pot_2=densidad_potencia(xx_2_norm) 
densidad_pot_3=densidad_potencia(xx_3_norm) 

# Presentación gráfica de los resultados en conjunto
plt.figure()
plt.plot(k,10*np.log10(densidad_pot_1),'c--+',label=f'PDS_{k0_1*df}')
plt.plot(k,10*np.log10(densidad_pot_2),'m--*',label=f'PDS_{k0_2*df}')
plt.plot(k,10*np.log10(densidad_pot_3),'y--o',label=f'PDS_{k0_3*df}')
plt.title('Densidad Espectral de Potencia de FFT{h(n)}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.xlim([0, fs/2]) 
plt.show() 

# Presentación gráfica de los resultados para K0_1
plt.figure()
plt.plot(k,10*np.log10(densidad_pot_1),'c--+',label=f'PDS_{k0_1*df}')
plt.title('Densidad Espectral de Potencia de FFT{h(n)}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.xlim([0, fs/2]) 
plt.show() 

# Presentación gráfica de los resultados para K0_2
plt.figure()
plt.plot(k,10*np.log10(densidad_pot_2),'m--*',label=f'PDS_{k0_2*df}')
plt.title('Densidad Espectral de Potencia de FFT{h(n)}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.xlim([0, fs/2]) 
plt.show() 

# Presentación gráfica de los resultados para K0_3
plt.figure()
plt.plot(k,10*np.log10(densidad_pot_3),'y--o',label=f'PDS_{k0_3*df}')
plt.title('Densidad Espectral de Potencia de FFT{h(n)}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.xlim([0, fs/2]) 
plt.show() 

#%% Invocación de las funciones del punto 1B
# Verifico Parseval
energia_t_1,potencia_f_1=parseval(xx_1_norm,densidad_pot_1,N)
energia_t_2,potencia_f_2=parseval(xx_2_norm,densidad_pot_2,N)
energia_t_3,potencia_f_3=parseval(xx_3_norm,densidad_pot_3,N)

#%% Invocación de las funciones del punto 1C
xx_1_zp=zero_padding_fft_abs(M,N,xx_1_norm)
xx_2_zp=zero_padding_fft_abs(M,N,xx_2_norm)
xx_3_zp=zero_padding_fft_abs(M,N,xx_3_norm)

# Densidad de potencia de las señales con zero padding
densidad_pot_1_zp=densidad_potencia(xx_1_zp) 
densidad_pot_2_zp=densidad_potencia(xx_2_zp) 
densidad_pot_3_zp=densidad_potencia(xx_3_zp)

# Presentación gráfica de los resultados de zero padding en conjunto
plt.figure()
plt.plot(k_M,10*np.log10(densidad_pot_1_zp),'c--+',label=f'PDS_zp_{k0_1*df}')
plt.plot(k_M,10*np.log10(densidad_pot_2_zp),'m--*',label=f'PDS_zp_{k0_2*df}')
plt.plot(k_M,10*np.log10(densidad_pot_3_zp),'y--o',label=f'PDS_zp_{k0_3*df}')
plt.title('Densidad Espectral de Potencia con zero padding de FFT{h(n)}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.xlim([0, M/2]) 
plt.show() 

# Presentación gráfica de los resultados de zero padding para K0_1
plt.figure()

plt.subplot(2,1,1)
plt.plot(k_M,10*np.log10(densidad_pot_1_zp),'c--+',label=f'PDS_zp_{k0_1*df}')
plt.title('Densidad Espectral de Potencia con zero padding de FFT{h(n)}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.xlim([0, M/2]) 

plt.subplot(2,1,2)
plt.plot(k_M,10*np.log10(densidad_pot_1_zp),'c--+',label=f'PDS_zp_{k0_1*df}')
plt.title('Zoom')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.ylim([20,60])
plt.xlim([2400, 2600]) 

plt.tight_layout()
plt.show() 

# Presentación gráfica de los resultados de zero padding para K0_2
plt.figure()
plt.suptitle('Densidad Espectral de Potencia con zero padding de FFT{h(n)}')


plt.subplot(2,1,1)
plt.plot(k_M,10*np.log10(densidad_pot_2_zp),'m--*',label=f'PDS_zp_{k0_2*df}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.xlim([0, M/2]) 

plt.subplot(2,1,2)
plt.plot(k_M,10*np.log10(densidad_pot_2_zp),'m--*',label=f'PDS_zp_{k0_2*df}')
plt.title('Zoom')
plt.ylabel('Amplitud |X(k)|^2 en dB')
plt.grid()
plt.legend()
plt.ylim([20,60])
plt.xlim([2400, 2600]) 

plt.tight_layout()
plt.show() 

# Presentación gráfica de los resultados de zero padding para K0_3
plt.figure()
plt.suptitle('Densidad Espectral de Potencia con zero padding de FFT{x(n)}')

plt.subplot(2,1,1)
plt.plot(k_M,densidad_pot_3_zp,'y--o',label=f'PDS_zp_{k0_3*df}')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |X(k)|^2 [dB]')
plt.grid()
plt.legend()
plt.xlim([0, M/2])
plt.ylim([-70,70])

plt.subplot(2,1,2)
plt.plot(k_M,10*np.log10(densidad_pot_3_zp),'y--o',label=f'PDS_zp_{k0_3*df}')
plt.title('Zoom')
plt.ylabel('Amplitud |X(k)|^2 [dB]')
plt.grid()
plt.legend()
plt.ylim([20, 60])
plt.xlim([2400,2600])

plt.tight_layout()
plt.show() 

# Verifico Parseval
energia_t_1_zp,potencia_f_1_zp=parseval(xx_1_zp,densidad_pot_1_zp,M)
energia_t_2_zp,potencia_f_2_zp=parseval(xx_2_zp,densidad_pot_2_zp,M)
energia_t_3_zp,potencia_f_3_zp=parseval(xx_3_zp,densidad_pot_3_zp,M)

#%% Invocación de la función del Bonus (punto 2)
delta_aux=signal.unit_impulse(N)

h_I=ec_diferencias_coef(coef_x_I,coef_y_I,N,fs,delta_aux,y_label="h",title="Respuesta al impulso del sistema I LTI",plot=None)
h_II=ec_diferencias_coef(coef_x_II,coef_y_II,N,fs,delta_aux,y_label="h",title="Respuesta al impulso del sistema II LTI",plot=None)
h_III=ec_diferencias_coef(coef_x_III,coef_y_III,N,fs,delta_aux,y_label="h",title="Respuesta al impulso del sistema III LTI",plot=None)

fft_abs_h_I=fft_abs_func(h_I)
fft_abs_h_II=fft_abs_func(h_II)
fft_abs_h_III=fft_abs_func(h_III)

polos_I = np.roots(coef_y_I)
ceros_I = np.roots(coef_x_I)
polos_II = np.roots(coef_y_II)
ceros_II = np.roots(coef_x_II)
polos_III = np.roots(coef_y_III)
ceros_III = np.roots(coef_x_III)

print(f'Los polos del sitema I son: {polos_I}\n')
print(f'Los ceros del sitema I son: {ceros_I}\n')
print(f'Los polos del sitema II son: {polos_II}\n')
print(f'Los ceros del sitema II son: {ceros_II}\n')
print(f'Los polos del sitema III son: {polos_III}\n')
print(f'Los ceros del sitema III son: {ceros_III}\n')

plt.figure()
plt.suptitle('Respuesta en Frecuencia de FFT{h(n)}')

plt.subplot(3,1,1)
plt.plot(k,20*np.log10(fft_abs_h_I),'r--+',label='RTA_frec_h_I')
plt.ylabel('Amplitud |H(k)| [dB]')
plt.xlim([k.min() - 0.1*(k.max()-k.min()), k.max() + 0.1*(k.max()-k.min())])
plt.grid()
plt.legend()

plt.subplot(3,1,2)
plt.plot(k,20*np.log10(fft_abs_h_II),'g--*',label='RTA_frec_h_II')
plt.ylabel('Amplitud |H(k)| [dB]')
plt.xlim([k.min() - 0.1*(k.max()-k.min()), k.max() + 0.1*(k.max()-k.min())])
plt.grid()
plt.legend()

plt.subplot(3,1,3)
plt.plot(k,20*np.log10(fft_abs_h_III),'b--o',label='RTA_frec_h_III')
plt.xlabel('Frecuencia discreta k.Δf [Hz]')
plt.ylabel('Amplitud |H(k)| [dB]')
plt.xlim([k.min() - 0.1*(k.max()-k.min()), k.max() + 0.1*(k.max()-k.min())])
plt.grid()
plt.legend()

plt.tight_layout()
plt.show() 