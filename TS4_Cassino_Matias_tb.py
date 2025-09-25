# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 19:26:48 2025

@author: Nancy
"""
#%% Consigna
"""
Comenzaremos con la generación de la siguiente señal:

x(k)=a0⋅sen(Ω1⋅n)+na(n) siendo a0=2, Ω1=Ω0+fr⋅2πN y Ω0=π2

siendo la variable aleatoria definida por la siguiente distribución de probabilidad

fr∼U(−2,2) y na∼N(0,σ2)

Diseñe los siguientes estimadores,  de amplitud a1:  a^i1=|Xiw(Ω0)|=|F{x(n)⋅wi(n)}|

 y de frecuencia Ω1:  Ω^i1=arg maxf{|Xiw(Ω)|}

para cada una de las ventanas: rectangular (sin ventana), flattop, blackmanharris y otra que elija de scipy.signal.windows

Y siguiendo las siguientes consignas para su experimentación:

Considere 200 realizaciones (muestras tomadas de fr) de 1000 muestras para cada experimento.
Parametrice para SNR's de 3 y 10 db (Ayuda: calibre a1 para que la potencia de la senoidal sea 1 W).

Se pide:

1) Realizar una tabla por cada SNR, que describa el sesgo y la varianza de cada estimador para cada ventana analizada. 
Recuerde incluir las ventanas rectangular (sin ventana), flattop y blackmanharris y otras que considere.

Ayuda: Puede calcular experimentalmente el sesgo y la varianza de un estimador:

a0^=|Xiw(Ω0)| siendo sa=E{a0^}−a0 y va=var{a0^}=E{(a0^−E{a0^})2}

y pueden aproximarse cuando consideramos los valores esperados como las medias muestrales

E{a0^}=μa^=1M∑j=0M−1aj^ -> sa=μa^−a0 y va=1M∑j=0M−1(aj^−μa^)2

Bonus:
 2) Analice el efecto del zero-padding para el estimador Ω^1
 
 3) Proponga estimadores alternativos para frecuencia y amplitud de la senoidal y repita el experimento.

 4) Visualizar los 3 histogramas juntos 

"""

#%% Modulos
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds
import scipy.signal as sig
import scipy.stats as st

#%% Funciones
plt.close("all")

def mi_funcion_sen_estocastica_matricial(vmax, dc, ff, fr_matriz, realizaciones, ph, N, fs, plot=True):
    
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    tt = np.linspace(0, (N-1)*ts, N).reshape(-1,1) # grilla de sampleo temporal discreta (n) pasa de vector en 1xN a matriz NX1 (-1 toma el tamaño del último elemento)
    tt_matriz = np.tile(tt, (1, realizaciones))   #1000x200  pq tile replica array en columnas= repeticiones y filas =1 = no repitas
    omega_0=ff
    omega_1=(fs/N)*fr_matriz + omega_0
    arg = 2*np.pi*omega_1*tt_matriz + ph # argumento
    xx_matriz = (vmax*(np.sin(arg)) + dc) # señal
    var_x=np.var(xx_matriz)
    
    print(f'La varianza de la señal sin normalizar es: {var_x}\n')    
    if plot:
        
        #%% Presentación gráfica de los resultados
        plt.figure()
        plt.plot(tt_matriz, xx_matriz, label=f"f = {ff} Hz\nN = {N}\nTs = {ts} s\nPotencia = {var_x:.3} W")
        plt.title('Señal: senoidal')
        plt.xlabel('tiempo [s]')
        plt.ylabel('Amplitud [V]')
        plt.grid()
        plt.xlim([tt_matriz.min() - 0.1*(tt_matriz.max()-tt_matriz.min()), tt_matriz.max() + 0.1*(tt_matriz.max()-tt_matriz.min())])
        plt.ylim([xx_matriz.min() - 0.1*(xx_matriz.max()-xx_matriz.min()), xx_matriz.max() + 0.1*(xx_matriz.max()-xx_matriz.min())])
        plt.legend()
        plt.show() 
        
    return tt_matriz,xx_matriz


def mi_funcion_noise_matricial(N,SNR,media_n,realizaciones):
    var_n=10**(-SNR/10)
    med_n=media_n
    # n=np.random.normal(med_n, np.sqrt(var_n),N).reshape(-1,1)
    # n_matriz=np.tile(n, (1, realizaciones))
    n = np.random.normal(med_n, np.sqrt(var_n), (N, realizaciones))  # ruido independiente por realización
    n_matriz = n.reshape(N, realizaciones) # redundante
    var_n=np.var(n)
    print(f'La varianza del ruido es: {var_n}\n')
    
    return n_matriz,var_n


def frecuencia_random_matricial(a,b,realizaciones):
    fr=np.random.uniform(a,b,realizaciones).reshape(1,-1) # 1x200
    fr_matriz=np.tile(fr, (N, 1)) #1000x200
    return fr_matriz

def normalizacion(x):
    media_x=np.mean(x) #media
    desvio_x=np.std(x) #desvio
    xx_norm=(x-media_x)/desvio_x #señal normalizada
    varianza_x=np.var(xx_norm) #varianza
    print(f'La varianza de la señal normalizada es: {varianza_x}\n')
    
    return xx_norm,varianza_x

def mod_y_fase_fft(fft_x):
    fft_x_abs=np.abs(fft_x)
    fft_x_abs_ph=np.max(np.angle(fft_x_abs))
    return fft_x_abs, fft_x_abs_ph

def estimadores(fft_abs_x,df,ff,N):
    k0= int(np.round(ff*(1/df))) # índice redondeado a entero de la fft que corresponde a ff
    a_estimadas=fft_abs_x[k0,:] # Vector de amplitudes en el índice de ff para c/realización
    index=np.argmax(fft_abs_x[:N//2],axis=0) # Vector de índices donde fft tiene máximo argumento para c/realización
    f_estimadas=index*df # Conversión de índice a frecuencia
    return f_estimadas,a_estimadas

def estadisticas(x_real,estimacion):
    media=np.mean(estimacion)
    sesgo=media-x_real
    varianza=np.var(estimacion,mean=media)
    return sesgo,varianza

def estadisticas_robustas(x_real,estimacion):
    mediana=np.median(estimacion)
    sesgo=mediana-x_real
    mad=st.median_abs_deviation(estimacion, scale='normal') # Normal escala por sqrt(pi/2) para obtener un valor proporcional al desvío estándar para una distribución Gaussiana
    var_robusta=mad**2
    return sesgo,var_robusta

#%% Parámetros
N=1000
M=10*N
fs=1000
df=fs/N
df_M=fs/M
k=np.arange(N)*df
k_M=np.arange(M)*df
ts = 1/fs # tiempo de muestreo
realizaciones=200 # parametro de fr
bins=15
a_real=2 # Amplitud real
f_real=fs/4  # Frecuencia real

#%% Invocación de las funciones del punto 1
n_m_1,var_n_m_1=mi_funcion_noise_matricial(N=N,SNR=3,media_n=0,realizaciones=realizaciones)
n_m_2,var_n_m_2=mi_funcion_noise_matricial(N=N,SNR=10,media_n=0,realizaciones=realizaciones)

fr_m=frecuencia_random_matricial(a=-2,b=2,realizaciones=realizaciones)

t_m_1,x_m_1 = mi_funcion_sen_estocastica_matricial(vmax = 2, dc = 0, ff = fs/4, fr_matriz=fr_m, realizaciones=realizaciones, ph=0, N=N,fs=fs,plot=None)

# Normalización de señal
x_m_norm_1,var_m_norm_1=normalizacion(x_m_1)

# Señal normalizada con ruido
xn_m_1=x_m_norm_1+n_m_1

# Varianza de señal normalizada con ruido
var_xn_m_1=np.var(xn_m_1)

# Ventanas
w_bh=sig.windows.blackmanharris(N).reshape(-1,1)
w_bh_m=np.tile(w_bh, (1, realizaciones))

w_hamming=sig.windows.hamming(N).reshape(-1,1)
w_hamming_m=np.tile(w_hamming, (1, realizaciones))

w_flattop=sig.windows.flattop(N).reshape(-1,1)
w_flattop_m=np.tile(w_flattop, (1, realizaciones))

w_rectangular=sig.windows.get_window("boxcar",N).reshape(-1,1)
w_rectangular_m=np.tile(w_rectangular, (1, realizaciones))

# Señal normalizada con ruido ventaneada
xn_m_1_w_bh=xn_m_1*w_bh_m
xn_m_1_w_hamming=xn_m_1*w_hamming_m
xn_m_1_w_flattop=xn_m_1*w_flattop_m
xn_m_1_w_rectangular=xn_m_1*w_rectangular_m

# FFT normalizada por N
fft_xn_m_1_w_bh=np.fft.fft(xn_m_1_w_bh,axis=0)/N 
fft_xn_m_1_w_hamming=np.fft.fft(xn_m_1_w_hamming,axis=0)/N 
fft_xn_m_1_w_flattop=np.fft.fft(xn_m_1_w_flattop,axis=0)/N 
fft_xn_m_1_w_rectangular=np.fft.fft(xn_m_1_w_rectangular,axis=0)/N 

# Módulo y fase de FFT
fft_xn_m_1_w_bh_abs,fft_xn_m_1_w_bh_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_bh)
fft_xn_m_1_w_hamming_abs,fft_xn_m_1_w_hamming_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_hamming)
fft_xn_m_1_w_flattop_abs,fft_xn_m_1_w_flattop_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_flattop)
fft_xn_m_1_w_rectangular_abs,fft_xn_m_1_w_rectangular_abs_ph=mod_y_fase_fft(fft_xn_m_1_w_rectangular)

# Estimadores de amplitud y frecuencia para c/ventana
f_estimado_1_xn_1_w_bh,a_estimado_1_xn_1_w_bh=estimadores(fft_xn_m_1_w_bh_abs,df=df,ff=fs/4,N=N)
f_estimado_1_xn_1_w_hamming,a_estimado_1_xn_1_w_hamming=estimadores(fft_xn_m_1_w_hamming_abs,df=df,ff=fs/4,N=N)
f_estimado_1_xn_1_w_flattop,a_estimado_1_xn_1_w_flattop=estimadores(fft_xn_m_1_w_flattop_abs,df=df,ff=fs/4,N=N)
f_estimado_1_xn_1_w_rectangular,a_estimado_1_xn_1_w_rectangular=estimadores(fft_xn_m_1_w_rectangular_abs,df=df,ff=fs/4,N=N)

# Estadisticas - Sesgo y varianza
sesgo_a_bh,var_a_bh=estadisticas(a_real, a_estimado_1_xn_1_w_bh)
sesgo_f_bh,var_f_bh=estadisticas(f_real, f_estimado_1_xn_1_w_bh)

print("Ventana Blackman-Harris:")
print(f" Amplitud -> sesgo={sesgo_a_bh:.3f}, var={var_a_bh:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_bh:.3f}, var={var_f_bh:.3e}")

sesgo_a_hamming,var_a_hamming=estadisticas(a_real, a_estimado_1_xn_1_w_hamming)
sesgo_f_hamming,var_f_hamming=estadisticas(f_real, f_estimado_1_xn_1_w_hamming)

print("Ventana Hamming:")
print(f" Amplitud -> sesgo={sesgo_a_hamming:.3f}, var={var_a_hamming:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_hamming:.3f}, var={var_f_hamming:.3e}")

sesgo_a_flattop,var_a_flattop=estadisticas(a_real, a_estimado_1_xn_1_w_flattop)
sesgo_f_flattop,var_f_flattop=estadisticas(f_real, f_estimado_1_xn_1_w_flattop)

print("Ventana Flattop:")
print(f" Amplitud -> sesgo={sesgo_a_flattop:.3f}, var={var_a_flattop:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_flattop:.3f}, var={var_f_flattop:.3e}")

sesgo_a_rectangular,var_a_rectangular=estadisticas(a_real, a_estimado_1_xn_1_w_rectangular)
sesgo_f_rectangular,var_f_rectangular=estadisticas(f_real, f_estimado_1_xn_1_w_rectangular)

print("Ventana Rectangular:")
print(f" Amplitud -> sesgo={sesgo_a_rectangular:.3f}, var={var_a_rectangular:.3e}")
print(f" Frecuencia -> sesgo={sesgo_f_rectangular:.3f}, var={var_f_rectangular:.3e}")

#%% Bonus (punto 2)
# FFT normalizada por N
fft_xn_m_1_w_bh_zp=np.fft.fft(xn_m_1_w_bh,M,axis=0)/N 
fft_xn_m_1_w_hamming_zp=np.fft.fft(xn_m_1_w_hamming,M,axis=0)/N 
fft_xn_m_1_w_flattop_zp=np.fft.fft(xn_m_1_w_flattop,M,axis=0)/N 
fft_xn_m_1_w_rectangular_zp=np.fft.fft(xn_m_1_w_rectangular,M,axis=0)/N 

# Módulo y fase de FFT
fft_xn_m_1_w_bh_abs_zp,fft_xn_m_1_w_bh_abs_ph_zp=mod_y_fase_fft(fft_xn_m_1_w_bh_zp)
fft_xn_m_1_w_hamming_abs_zp,fft_xn_m_1_w_hamming_abs_ph_zp=mod_y_fase_fft(fft_xn_m_1_w_hamming_zp)
fft_xn_m_1_w_flattop_abs_zp,fft_xn_m_1_w_flattop_abs_ph_zp=mod_y_fase_fft(fft_xn_m_1_w_flattop_zp)
fft_xn_m_1_w_rectangular_abs_zp,fft_xn_m_1_w_rectangular_abs_ph_zp=mod_y_fase_fft(fft_xn_m_1_w_rectangular_zp)

# Estimadores de amplitud y frecuencia para c/ventana
f_estimado_1_xn_1_w_bh_zp,_=estimadores(fft_xn_m_1_w_bh_abs_zp,df=df_M,ff=fs/4,N=M)
f_estimado_1_xn_1_w_hamming_zp,_=estimadores(fft_xn_m_1_w_hamming_abs_zp,df=df_M,ff=fs/4,N=M)
f_estimado_1_xn_1_w_flattop_zp,_=estimadores(fft_xn_m_1_w_flattop_abs_zp,df=df_M,ff=fs/4,N=M)
f_estimado_1_xn_1_w_rectangular_zp,_=estimadores(fft_xn_m_1_w_rectangular_abs_zp,df=df_M,ff=fs/4,N=M)

# Estadisticas
sesgo_f_bh_zp,var_f_bh_zp=estadisticas(f_real, f_estimado_1_xn_1_w_bh_zp)

print("Ventana Blackman-Harris con Zero Padding:")
print(f" Frecuencia -> sesgo={sesgo_f_bh_zp:.3f}, var={var_f_bh_zp:.3e}")

sesgo_f_hamming_zp,var_f_hamming_zp=estadisticas(f_real, f_estimado_1_xn_1_w_hamming_zp)

print("Ventana Hamming con Zero Padding:")
print(f" Frecuencia -> sesgo={sesgo_f_hamming_zp:.3f}, var={var_f_hamming_zp:.3e}")

sesgo_f_flattop_zp,var_f_flattop_zp=estadisticas(f_real, f_estimado_1_xn_1_w_flattop_zp)

print("Ventana Flattop con Zero Padding:")
print(f" Frecuencia -> sesgo={sesgo_f_flattop_zp:.3f}, var={var_f_flattop_zp:.3e}")

sesgo_f_rectangular_zp,var_f_rectangular_zp=estadisticas(f_real, f_estimado_1_xn_1_w_rectangular_zp)

print("Ventana Rectangular con Zero Padding:")
print(f" Frecuencia -> sesgo={sesgo_f_rectangular_zp:.3f}, var={var_f_rectangular_zp:.3e}")

#%% Bonus (punto 3)
# Estadisticas robustas
sesgo_robusto_a_bh,var_robusta_a_bh=estadisticas_robustas(a_real, a_estimado_1_xn_1_w_bh)
sesgo_robusto_f_bh,var_robusta_f_bh=estadisticas_robustas(f_real, f_estimado_1_xn_1_w_bh)

print("Ventana Blackman-Harris:")
print(f" Amplitud -> sesgo={sesgo_robusto_a_bh:.3f}, MAD**2={var_robusta_a_bh:.3e}")
print(f" Frecuencia -> sesgo={sesgo_robusto_f_bh:.3f}, MAD**2={var_robusta_f_bh:.3e}")

sesgo_robusto_a_hamming,var_robusta_a_hamming=estadisticas_robustas(a_real, a_estimado_1_xn_1_w_hamming)
sesgo_robusto_f_hamming,var_robusta_f_hamming=estadisticas_robustas(f_real, f_estimado_1_xn_1_w_hamming)

print("Ventana Hamming:")
print(f" Amplitud -> sesgo={sesgo_robusto_a_hamming:.3f}, MAD**2={var_robusta_a_hamming:.3e}")
print(f" Frecuencia -> sesgo={sesgo_robusto_f_hamming:.3f}, MAD**2={var_robusta_f_hamming:.3e}")

sesgo_robusto_a_flattop,var_robusta_a_flattop=estadisticas_robustas(a_real, a_estimado_1_xn_1_w_flattop)
sesgo_robusto_f_flattop,var_robusta_f_flattop=estadisticas_robustas(f_real, f_estimado_1_xn_1_w_flattop)

print("Ventana Flattop:")
print(f" Amplitud -> sesgo={sesgo_robusto_a_flattop:.3f}, MAD**2={var_robusta_a_flattop:.3e}")
print(f" Frecuencia -> sesgo={sesgo_robusto_f_flattop:.3f}, MAD**2={var_robusta_f_flattop:.3e}")

sesgo_robusto_a_rectangular,var_robusta_a_rectangular=estadisticas_robustas(a_real, a_estimado_1_xn_1_w_rectangular)
sesgo_robusto_f_rectangular,var_robusta_f_rectangular=estadisticas_robustas(f_real, f_estimado_1_xn_1_w_rectangular)

print("Ventana Rectangular:")
print(f" Amplitud -> sesgo={sesgo_robusto_a_rectangular:.3f}, MAD**2={var_robusta_a_rectangular:.3e}")
print(f" Frecuencia -> sesgo={sesgo_robusto_f_rectangular:.3f}, MAD**2={var_robusta_f_rectangular:.3e}")



#%% Bonus (punto 4)
plt.figure()
plt.hist(a_estimado_1_xn_1_w_bh-sesgo_a_bh, bins, alpha=0.5, label='Blackman-Harris')
plt.hist(a_estimado_1_xn_1_w_flattop-sesgo_a_hamming, bins, alpha=0.5, label='Flattop')
plt.hist(a_estimado_1_xn_1_w_rectangular-sesgo_a_rectangular, bins, alpha=0.5, label='Rectangular')
plt.axvline(np.mean(a_real), color='m', linestyle='--', label='Amplitud verdadera') # Línea vertical indicando valor verdadero de amplitud
plt.title("Histogramas de estimador de amplitud insesgados")
plt.xlabel("Amplitud estimada")
plt.ylabel("Realizaciones")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.hist(f_estimado_1_xn_1_w_bh-sesgo_f_bh, bins, alpha=0.5, label='Blackman-Harris')
plt.hist(f_estimado_1_xn_1_w_flattop-sesgo_f_flattop, bins, alpha=0.5, label='Flattop')
plt.hist(f_estimado_1_xn_1_w_rectangular-sesgo_f_rectangular, bins, alpha=0.5, label='Rectangular')
plt.axvline(np.mean(f_real), color='m', linestyle='--', label='Frecuencia verdadera') # Línea vertical indicando valor verdadero de frecuencia promedio
plt.title("Histogramas de estimador de frecuencia insesgados")
plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Realizaciones")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.hist(f_estimado_1_xn_1_w_bh_zp-sesgo_f_bh_zp, bins, alpha=0.5, label='Blackman-Harris')
plt.hist(f_estimado_1_xn_1_w_flattop_zp-sesgo_f_flattop_zp, bins, alpha=0.5, label='Flattop')
plt.hist(f_estimado_1_xn_1_w_rectangular_zp-sesgo_f_rectangular_zp, bins, alpha=0.5, label='Rectangular')
plt.axvline(np.mean(f_real), color='m', linestyle='--', label='Frecuencia verdadera') # Línea vertical indicando valor verdadero de frecuencia promedio
plt.title("Histogramas de estimador de frecuencia insesgados con Zero Padding")
plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Realizaciones")
plt.legend()
plt.grid()
plt.show()


#%% Histograma en escala dB
# a_estimado_1_xn_1_w_bh_abs=10*np.log10(a_estimado_1_xn_1_w_bh)
# a_estimado_1_xn_1_w_flattop_abs=10*np.log10(a_estimado_1_xn_1_w_flattop)
# a_estimado_1_xn_1_w_rectangular_abs=10*np.log10(a_estimado_1_xn_1_w_rectangular)

# plt.figure()
# plt.hist(a_estimado_1_xn_1_w_bh_abs, bins, alpha=0.5, label='Blackman-Harris')
# plt.hist(a_estimado_1_xn_1_w_flattop_abs, bins, alpha=0.5, label='Flattop')
# plt.hist(a_estimado_1_xn_1_w_rectangular_abs, bins, alpha=0.5, label='Rectangular')
# plt.axvline(10*np.log10(np.mean(a_real)), color='m', linestyle='--', label='Amplitud verdadera') # Línea vertical indicando valor verdadero de amplitud
# plt.title("Histogramas de estimador de amplitud para 3 ventanas")
# plt.xlabel("Amplitud estimada")
# plt.ylabel("Realizaciones")
# plt.legend()
# plt.grid()
# plt.show()

plt.figure()
plt.hist(a_estimado_1_xn_1_w_bh-sesgo_a_bh, bins, alpha=0.5, label='Blackman-Harris')
plt.hist(a_estimado_1_xn_1_w_flattop-sesgo_a_hamming, bins, alpha=0.5, label='Flattop')
plt.hist(a_estimado_1_xn_1_w_rectangular-sesgo_a_rectangular, bins, alpha=0.5, label='Rectangular')
plt.axvline(np.mean(a_real), color='m', linestyle='--', label='Amplitud verdadera') # Línea vertical indicando valor verdadero de amplitud
plt.title("Histogramas de estimador de amplitud insesgados")
plt.xlabel("Amplitud estimada")
plt.ylabel("Realizaciones")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.hist(f_estimado_1_xn_1_w_bh-sesgo_f_bh, bins, alpha=0.5, label='Blackman-Harris')
plt.hist(f_estimado_1_xn_1_w_flattop-sesgo_f_flattop, bins, alpha=0.5, label='Flattop')
plt.hist(f_estimado_1_xn_1_w_rectangular-sesgo_f_rectangular, bins, alpha=0.5, label='Rectangular')
plt.axvline(np.mean(f_real), color='m', linestyle='--', label='Frecuencia verdadera') # Línea vertical indicando valor verdadero de frecuencia promedio
plt.title("Histogramas de estimador de frecuencia insesgados")
plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Realizaciones")
plt.legend()
plt.grid()
plt.show()

#%% Presentación de las señales con Zero Padding
# Blackman-Harris
plt.figure()
plt.plot(k_M, 10*np.log10(np.abs(fft_xn_m_1_w_bh_abs_zp)**2))
plt.title('FFT de señal con ruido y con zero-Padding para: Blackman-Harris')
plt.xlabel('Frecuencia discreta k·Δf [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid()
# plt.legend()
plt.xlim([0, M/2])
plt.show()

# Hamming
plt.figure()
plt.plot(k_M, 10*np.log10(np.abs(fft_xn_m_1_w_hamming_abs_zp)**2))
plt.title('FFT de señal con ruido y con zero-Padding para: Hamming')
plt.xlabel('Frecuencia discreta k·Δf [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid()
# plt.legend()
plt.xlim([0, M/2])
plt.show()

# Flattop
plt.figure()
plt.plot(k_M, 10*np.log10(np.abs(fft_xn_m_1_w_flattop_abs_zp)**2))
plt.title('FFT de señal con ruido y con zero-Padding para: Flattop')
plt.xlabel('Frecuencia discreta k·Δf [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid()
# plt.legend()
plt.xlim([0, M/2])
plt.show()

# Rectangular
plt.figure()
plt.plot(k_M, 10*np.log10(np.abs(fft_xn_m_1_w_rectangular_abs_zp)**2))
plt.title('FFT de señal con ruido y con zero-Padding para: Rectangular')
plt.xlabel('Frecuencia discreta k·Δf [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid()
# plt.legend()
plt.xlim([0, M/2])
plt.show()



