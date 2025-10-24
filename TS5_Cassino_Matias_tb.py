# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 18:06:18 2025

@author: Matias
"""

#%% Consigna
"""
En el repositorio PDStestbench encontrará tres tipos de señales registradas:

- Electrocardiograma (ECG). En el archivo ECG_TP4.mat encontrará un registro electrocardiográfico (ECG) registrado durante una prueba de esfuerzo, junto con una serie de variables descriptas más abajo.

- Pletismografía (PPG). El archivo PPG.csv contiene una señal registrada en reposo de un estudiante de la materia que ha donado su registro para esta actividad.

- Audio. Tres registros en los que el profesor pronuncia una frase, y otros dos en los que se silba una melodía muy conocida.

Los detalles de cómo acceder a dichos registros los pueden encontrar en lectura_sigs.py

Se pide:

1) Realizar la estimación de la densidad espectral de potencia (PSD) de cada señal mediante alguno de los métodos vistos en clase (Periodograma ventaneado, Welch, Blackman-Tukey).

2) Realice una estimación del ancho de banda de cada señal y presente los resultados en un tabla para facilitar la comparación.

Bonus:

3) Proponga algún tipo de señal que no haya sido analizada y repita el análisis. No olvide explicar su origen y cómo fue digitalizada.

"""
#%% Modulos
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt  
import scipy.io as sio
from scipy.io.wavfile import write
from scipy.signal import periodogram
import matplotlib as mpl
import scipy.stats as st
import pandas as pd
# import sounddevice as sd

#%% Funciones
plt.close('all')

def ancho_de_banda(f,Px,porcentaje):
    pot_acum = np.cumsum(Px) #Potencia acumulada
    pot_acum_norm = pot_acum/pot_acum[-1] # Normalizo
    cond_aux=pot_acum_norm>=(porcentaje/100) # Máscara booleana
    f_umbral=f[cond_aux] # Filtro frecuencias según máscara 
    # idx = np.where(pot_acum_norm >= (porcentaje/100))[0] # Alternativa con índices que devuelve tupla y selecciono la primera en el vector de frecuencias
    f_wb=np.round(f_umbral[0]) # Tomo el primer valor redondeado
    print(f'El ancho de banda {f_wb} Hz contiene el {porcentaje}% de la potencia\n')
    return f_wb 

def periodograma(vector,fs,window,zp):
    N=len(vector)
    nfft=zp*N
    f_p, Px_p= sig.periodogram(vector, fs=fs, window=window, nfft=nfft,scaling='density')
    return f_p, Px_p

def blackman_tukey(vector, fs, window,zp):
    N=len(vector)
    M=N//5
    nfft=zp*N
    Rxx= sig.correlate(vector, vector, mode='full')/N
    aux= len(Rxx)//2
    Rxx=Rxx[aux-M:aux+M+1] # Aprecio solapamiento mayor al 20%
    aux_2=len(Rxx)
    if window == 'rectangular':
        w = np.ones(aux_2)
    elif window == 'hamming':
        w = np.hamming(aux_2)
    elif window == 'hann':
        w = np.hanning(aux_2)
    elif window == 'bartlett':
        w = np.bartlett(aux_2)
    elif window == 'blackman':
        w = np.blackman(aux_2)
    
    Rxx_w=Rxx*w
    Px_bt=np.abs(np.fft.fft(Rxx_w,nfft))/N
    Px_bt=Px_bt[:nfft//2]
    f_bt=np.linspace(0, fs/2,nfft//2)
    return f_bt, Px_bt

def welch(cant_promedio,vector,zp,fs,window):
    N=vector.shape[0] # N=Cantidad de muestras y accede al elemento 0
    nperseg=N//cant_promedio # L=largo del bloque
    print(f'El largo del bloque es: {nperseg}\n')
    nfft=zp*nperseg
    # Periodograma
    f_w, Px_w = sig.welch(vector, fs=fs, nperseg=nperseg, window=window, nfft=nfft)
    return f_w, Px_w

def periodograma(cant_promedio, vector, zp, fs):
    N = vector.shape[0]
    nperseg = N//cant_promedio
    nfft = zp*nperseg
    f_P, Px_P = periodogram(vector, fs = fs, nfft = nfft)
    return f_P, Px_P

def plot_periodograma(f,Px,f_wb,title, f0 = 0):
    # Periodograma
    plt.figure()
    plt.plot(f,Px)
    plt.axvline(f_wb, color='r', linestyle='--', label=f'BW ≈ {f_wb} Hz')
    plt.title(title)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD")
    plt.xlim([f0,f_wb + 1])
    plt.grid()
    return

#%% Parámetros generales
windows=['blackmanharris','hann','hamming','flattop','boxcar']

##################
# Lectura de ECG sin ruido
##################

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead[5000:12000])

ecg_one_lead = np.load('Archivos_aux_TS5/ecg_sin_ruido.npy') # esto es N
# plt.figure()
# plt.plot(ecg_one_lead)

# Parámetros ECG
fs_ecg = 1000 # Hz
cant_promedio_ecg = 14 # K
zp_ecg=2

# ####################################
# # Lectura de pletismografía (PPG) sin ruido
# ##################
ppg = np.load('Archivos_aux_TS5/ppg_sin_ruido.npy') # esto es N
# plt.figure()
# plt.plot(ppg)

# Parámetros PPG
fs_ppg = 400 # Hz
cant_promedio_ppg = 5 # K
zp_ppg=1

####################
# Lectura de audio cucaracha#
####################
# Cargar el archivo CSV como un array de NumPy
fs_cucaracha, wav_data_cucaracha = sio.wavfile.read('sonido/la cucaracha.wav')
# plt.figure()
# plt.plot(wav_data_cucaracha)

# sd.play(wav_data_cucaracha, fs_cucaracha)


# # Parámetros cucaracha
cant_promedio_cucaracha = 9 # K
zp_cucaracha=2

####################
# Lectura de audio prueba#
####################
# Cargar el archivo CSV como un array de NumPy
fs_prueba, wav_data_prueba = sio.wavfile.read('sonido/prueba psd.wav')
# plt.figure()
# plt.plot(wav_data_prueba)

# sd.play(wav_data_prueba, fs_prueba)

# # Parámetros prueba
cant_promedio_prueba = 10 # K = 8
zp_prueba=1

####################
# Lectura de audio silbido#
####################
# Cargar el archivo CSV como un array de NumPy
fs_silbido, wav_data_silbido = sio.wavfile.read('sonido/silbido.wav')
# plt.figure()
# plt.plot(wav_data_silbido)

# sd.play(wav_data_silbido, fs_silbido)

# Parámetros silbido
cant_promedio_silbido = 25 # K = 17
zp_silbido=3

####################
# Lectura de audio bonus#
####################
fs_audio, wav_data_audio = sio.wavfile.read('sonido/knock.wav')
# sd.play(wav_data_audio, fs_audio)
if wav_data_audio.ndim > 1:
#     # promedio de canales (podés usar sum/2 o canal específico)
    wav_data_audio = wav_data_audio.mean(axis=1)
plt.figure()
plt.plot(wav_data_audio)


# # Parámetros bonus
cant_promedio_audio = 15 # K
zp_audio=2

#%% Invocación de las funciones del punto 1 y Bonus
ff_P_ECG_SR, P_ECG_SR = periodograma(cant_promedio = cant_promedio_ecg, vector=ecg_one_lead, fs = fs_ecg, zp = zp_ecg)
f_w_ecg, Px_w_ecg=welch(cant_promedio=cant_promedio_ecg,vector=ecg_one_lead,zp=zp_ecg,fs=fs_ecg,window=windows[1])
f_P_ECG = ancho_de_banda(ff_P_ECG_SR,P_ECG_SR,porcentaje=99)
f_wb_ecg=ancho_de_banda(f_w_ecg,Px_w_ecg,porcentaje=99)

ff_P_PPG, P_PPG = periodograma(cant_promedio = cant_promedio_ppg, vector=ppg, fs = fs_ppg, zp = zp_ppg)
f_w_ppg, Px_w_ppg=welch(cant_promedio=cant_promedio_ppg,vector=ppg,zp=zp_ppg,fs=fs_ppg,window=windows[1])
f_P_PPG = ancho_de_banda(ff_P_PPG,P_PPG,porcentaje=99)
f_wb_ppg=ancho_de_banda(f_w_ppg,Px_w_ppg,porcentaje=99)

ff_P_cucaracha, P_cucaracha = periodograma(cant_promedio = cant_promedio_cucaracha, vector=wav_data_cucaracha, fs = fs_cucaracha, zp = zp_cucaracha)
f_w_cucaracha, Px_w_cucaracha=welch(cant_promedio=cant_promedio_cucaracha,vector=wav_data_cucaracha,zp=zp_cucaracha,fs=fs_cucaracha,window=windows[1])
f_P_cucaracha = ancho_de_banda(ff_P_cucaracha,P_cucaracha,porcentaje=99)
f_wb_cucaracha=ancho_de_banda(f_w_cucaracha,Px_w_cucaracha,porcentaje=99)

ff_P_prueba, P_prueba = periodograma(cant_promedio = cant_promedio_prueba, vector=wav_data_prueba, fs = fs_prueba, zp = zp_prueba)
f_w_prueba, Px_w_prueba=welch(cant_promedio=cant_promedio_prueba,vector=wav_data_prueba,zp=zp_prueba,fs=fs_prueba,window=windows[1])
f_P_prueba = ancho_de_banda(ff_P_prueba,P_prueba,porcentaje=99)
f_wb_prueba=ancho_de_banda(f_w_prueba,Px_w_prueba,porcentaje=99)

ff_P_silbido, P_silbido = periodograma(cant_promedio = cant_promedio_silbido, vector=wav_data_silbido, fs = fs_silbido, zp = zp_silbido)
f_w_silbido, Px_w_silbido=welch(cant_promedio=cant_promedio_silbido,vector=wav_data_silbido,zp=zp_silbido,fs=fs_silbido,window=windows[1])
f_P_silbido = ancho_de_banda(ff_P_silbido,P_silbido,porcentaje=99)
f_wb_silbido=ancho_de_banda(f_w_silbido,Px_w_silbido,porcentaje=99)

ff_P_audio, P_audio = periodograma(cant_promedio = cant_promedio_audio, vector=wav_data_audio, fs = fs_audio, zp = zp_audio)
f_w_audio, Px_w_audio=welch(cant_promedio=cant_promedio_audio,vector=wav_data_audio,zp=zp_audio,fs=fs_audio,window=windows[1])
f_P_audio = ancho_de_banda(ff_P_audio,P_audio,porcentaje=99)
f_wb_audio=ancho_de_banda(f_w_audio,Px_w_audio,porcentaje=99)

#%% Invocación de las funciones del punto 2 y Bonus
plot_periodograma(f_w_ecg, Px_w_ecg, f_wb_ecg, title="Periodograma ECG sin ruido por Welch")
plot_periodograma(ff_P_ECG_SR, P_ECG_SR, f_P_ECG, title="Periodograma ECG sin ruido por método de Periodograma Modificado")


plot_periodograma(f_w_ppg, Px_w_ppg, f_wb_ppg, title="Periodograma PPG sin ruido por Welch")
plot_periodograma(ff_P_PPG, P_PPG, f_P_PPG, title="Periodograma PPG sin ruido por método de Periodograma Modificado")

plot_periodograma(f_w_cucaracha, Px_w_cucaracha, f_wb_cucaracha, f0 = 500, title="Periodograma audio cucaracha sin ruido por Welch")
plot_periodograma(ff_P_cucaracha, P_cucaracha, f_P_cucaracha, f0 = 500, title="Periodograma audio cucaracha sin ruido por método de Periodograma Modificado")

plot_periodograma(f_w_prueba, Px_w_prueba, f_wb_prueba, title="Periodograma audio prueba sin ruido por Welch")
plot_periodograma(ff_P_prueba, P_prueba, f_P_prueba, title="Periodograma audio prueba sin ruido por método de Periodograma Modificado")

plot_periodograma(f_w_silbido, Px_w_silbido, f_wb_silbido, title="Periodograma audio silbido sin ruido por Welch")
plot_periodograma(ff_P_silbido, P_silbido, f_P_silbido, title="Periodograma audio silbido sin ruido por método de Periodograma Modificado")

plot_periodograma(f_w_audio, Px_w_audio, f_wb_audio, title="Periodograma audio bonus por Welch")
plot_periodograma(ff_P_audio, P_audio, f_P_audio, title="Periodograma audio bonus por método de Periodograma Modificado")

tabla_bw = pd.DataFrame({
    'Archivo de Señal': [
        'ECG sin ruido',
        'PPG sin ruido',
        'Audio - La Cucaracha',
        'Audio - Prueba PSD',
        'Audio - Silbido',
        'Audio - Bonus'
    ],
    'Ancho de Banda [Hz]': [
        f_wb_ecg,
        f_wb_ppg,
        f_wb_cucaracha,
        f_wb_prueba,
        f_wb_silbido,
        f_wb_audio
    ],
    'Ancho de Banda - Periodograma Modificado [Hz]': [
        f_P_ECG,
        f_P_PPG,
        f_P_cucaracha,
        f_P_prueba,
        f_P_silbido,
        f_P_audio
    ]
}) # Tabla de 3 columnas (etiquetas) y 6 filas (listas)


# print("\n=== Tabla comparativa de anchos de banda con Welch ===\n")
# print(tabla_bw.to_string(index=False))

#%% Basura
##################
## ECG con ruido
##################
# ecg_one_lead = mat_struct['ecg_lead']
# N = len(ecg_one_lead)
# ecg_sucio=ecg_one_lead[670000:700000].reshape(-1)
# plt.figure()
# plt.plot(ecg_sucio)

# Parámetros ECG con ruido
# cant_promedio_ecg_sucio = 20 # K
# zp_ecg_sucio=2
# f_w_ecg_sucio, Px_w_ecg_sucio=welch(cant_promedio=cant_promedio_ecg_sucio,vector=ecg_sucio,zp=zp_ecg_sucio,fs=fs_ecg,window=window_bh)

# Periodograma
# plt.figure()
# plt.plot(f_w_ecg_sucio,Px_w_ecg_sucio)
# plt.title("Periodograma ECG con ruido por Welch")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("PSD")
# plt.xlim([0,50])
# plt.grid()

# Periodograma en dB
# plt.figure()
# plt.plot(f_w_ecg_sucio,10*np.log10(Px_w_ecg_sucio))
# plt.title("Periodograma ECG con ruido por Welch en dB")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("PSD [dB]")
# plt.xlim([0,50])
# plt.grid()

#%% Periodograma
#%% Invocación de las funciones del punto 1 y Bonus
f_p_ecg, Px_p_ecg=periodograma(vector=ecg_one_lead,zp=zp_ecg,fs=fs_ecg,window=windows[1])
f_wb_ecg_p=ancho_de_banda(f_p_ecg,Px_p_ecg,porcentaje=99)

# f_p_ppg, Px_p_ppg=periodograma(vector=ppg,zp=zp_ppg,fs=fs_ppg,window=windows[1])
# f_wb_ppg_p=ancho_de_banda(f_p_ppg,Px_p_ppg,porcentaje=99)

# f_p_cucaracha, Px_p_cucaracha=periodograma(vector=wav_data_cucaracha,zp=zp_cucaracha,fs=fs_cucaracha,window=windows[1])
# f_wb_cucaracha_p=ancho_de_banda(f_p_cucaracha,Px_p_cucaracha,porcentaje=99)

# f_p_prueba, Px_p_prueba=periodograma(vector=wav_data_prueba,zp=zp_prueba,fs=fs_prueba,window=windows[1])
# f_wb_prueba_p=ancho_de_banda(f_p_prueba,Px_p_prueba,porcentaje=99)

# f_p_silbido, Px_p_silbido=periodograma(vector=wav_data_silbido,zp=zp_silbido,fs=fs_silbido,window=windows[1])
# f_wb_silbido_p=ancho_de_banda(f_p_silbido,Px_p_silbido,porcentaje=99)

# f_p_audio, Px_p_audio=periodograma(vector=wav_data_audio,zp=zp_audio,fs=fs_audio,window=windows[1])
# f_wb_audio_p=ancho_de_banda(f_p_audio,Px_p_audio,porcentaje=99)

#%% Invocación de las funciones del punto 2 y Bonus
plot_periodograma(f_p_ecg, Px_p_ecg, f_wb_ecg_p, title="Periodograma ECG sin ruido")
# plot_periodograma(f_p_ppg, Px_p_ppg, f_wb_ppg_p, title="Periodograma PPG sin ruido")
# plot_periodograma(f_p_cucaracha, Px_p_cucaracha, f_wb_cucaracha_p, title="Periodograma audio cucaracha sin ruido")
# plot_periodograma(f_p_prueba, Px_p_prueba, f_wb_prueba_p, title="Periodograma audio prueba sin ruido")
# plot_periodograma(f_p_silbido, Px_p_silbido, f_wb_silbido_p, title="Periodograma audio silbido sin ruido")
# plot_periodograma(f_p_audio, Px_p_audio, f_wb_audio_p, title="Periodograma audio bonus")


# tabla_bw = pd.DataFrame({
#     'Archivo de Señal': [
#         'ECG sin ruido',
#         'PPG sin ruido',
#         'Audio - La Cucaracha',
#         'Audio - Prueba PSD',
#         'Audio - Silbido',
#         'Audio - Bonus'
#     ],
#     'Ancho de Banda [Hz]': [
#         f_wb_ecg_p,
#         f_wb_ppg_p,
#         f_wb_cucaracha_p,
#         f_wb_prueba_p,
#         f_wb_silbido_p,
#         f_wb_audio_p
#     ]
# }) # Tabla de 2 columnas (etiquetas) y 6 filas (listas)

# print("\n=== Tabla comparativa de anchos de banda ===\n")
# print(tabla_bw.to_string(index=False))

#%% Blackman-Tukey
#%% Invocación de las funciones del punto 1 y Bonus
zp_bt_ecg=1
f_bt_ecg, Px_bt_ecg=blackman_tukey(vector=ecg_one_lead,fs=fs_ecg,window=windows[2],zp=zp_bt_ecg)
f_wb_ecg_bt=ancho_de_banda(f_bt_ecg,Px_bt_ecg,porcentaje=99)

# zp_bt_ppg=1
# f_bt_ppg, Px_bt_ppg=blackman_tukey(vector=ppg,fs=fs_ppg,window=windows[2],zp=zp_bt_ppg)
# f_wb_ppg_bt=ancho_de_banda(f_bt_ppg,Px_bt_ppg,porcentaje=99)

# zp_bt_cucaracha=1
# f_bt_cucaracha, Px_bt_cucaracha=blackman_tukey(vector=wav_data_cucaracha,fs=fs_cucaracha,window=windows[2],zp=zp_bt_cucaracha)
# f_wb_cucaracha_bt=ancho_de_banda(f_bt_cucaracha,Px_bt_cucaracha,porcentaje=99)

# zp_bt_prueba=1
# f_bt_prueba, Px_bt_prueba=blackman_tukey(vector=wav_data_prueba,fs=fs_prueba,window=windows[2],zp=zp_bt_prueba)
# f_wb_prueba_bt=ancho_de_banda(f_bt_prueba,Px_bt_prueba,porcentaje=99)

# zp_bt_silbido=1
# f_bt_silbido, Px_bt_silbido=blackman_tukey(vector=wav_data_silbido,fs=fs_silbido,window=windows[2],zp=zp_bt_silbido)
# f_wb_silbido_bt=ancho_de_banda(f_bt_silbido,Px_bt_silbido,porcentaje=99)

# zp_bt_audio=1
# f_bt_audio, Px_bt_audio=blackman_tukey(vector=wav_data_audio,fs=fs_audio,window=windows[2], zp=zp_bt_audio)
# f_wb_audio_bt=ancho_de_banda(f_bt_audio,Px_bt_audio,porcentaje=99)

#%% Invocación de las funciones del punto 2 y Bonus
plot_periodograma(f_bt_ecg, Px_bt_ecg, f_wb_ecg_bt, title="Periodograma ECG sin ruido por Blackman-Tukey")
# plot_periodograma(f_bt_ppg, Px_bt_ppg, f_wb_ppg_bt, title="Periodograma PPG sin ruido por Blackman-Tukey")
# plot_periodograma(f_bt_cucaracha, Px_bt_cucaracha, f_wb_cucaracha_bt, title="Periodograma audio cucaracha sin ruido por Blackman-Tukey")
# plot_periodograma(f_bt_prueba, Px_bt_prueba, f_wb_prueba_bt, title="Periodograma audio prueba sin ruido por Blackman-Tukey")
# plot_periodograma(f_bt_silbido, Px_bt_silbido, f_wb_silbido_bt, title="Periodograma audio silbido sin ruido por Blackman-Tukey")
# plot_periodograma(f_bt_audio, Px_bt_audio, f_wb_audio_bt, title="Periodograma audio bonus por Blackman-Tukey")


# tabla_bw = pd.DataFrame({
#     'Archivo de Señal': [
#         'ECG sin ruido',
#         'PPG sin ruido',
#         'Audio - La Cucaracha',
#         'Audio - Prueba PSD',
#         'Audio - Silbido',
#         'Audio - Bonus'
#     ],
#     'Ancho de Banda con Blackman-Tukey [Hz]': [
#         f_wb_ecg_bt,
#         f_wb_ppg_bt,
#         f_wb_cucaracha_bt,
#         f_wb_prueba_bt,
#         f_wb_silbido_bt,
#         f_wb_audio_bt
#     ]
# }) # Tabla de 2 columnas (etiquetas) y 6 filas (listas)

# print("\n=== Tabla comparativa de anchos de banda con Blackman-Tukey ===\n")
# print(tabla_bw.to_string(index=False))