# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:39:59 2025

@author: Matías Cassino
"""

#%% Módulos
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from matplotlib import patches
from pytc2.sistemas_lineales import plot_plantilla
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

plt.close('all')

#%% Funciones
def matriz_confusion_qrs(mis_qrs, qrs_det, tolerancia_ms=150, fs=1000):
    """
    Calcula matriz de confusión para detecciones QRS usando solo NumPy y SciPy
    
    Parámetros:
    - mis_qrs: array con tiempos de tus detecciones (muestras)
    - qrs_det: array con tiempos de referencia (muestras)  
    - tolerancia_ms: tolerancia en milisegundos (default 150ms)
    - fs: frecuencia de muestreo (default 360 Hz)
    """
    
    # Convertir a arrays numpy
    mis_qrs = np.array(mis_qrs)
    qrs_det = np.array(qrs_det)
    
    # Convertir tolerancia a muestras
    tolerancia_muestras = tolerancia_ms * fs / 1000
    
    # Inicializar contadores
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    
    # Arrays para marcar detecciones ya emparejadas
    mis_qrs_emparejados = np.zeros(len(mis_qrs), dtype=bool)
    qrs_det_emparejados = np.zeros(len(qrs_det), dtype=bool)
    
    # Encontrar True Positives (detecciones que coinciden dentro de la tolerancia)
    for i, det in enumerate(mis_qrs):
        diferencias = np.abs(qrs_det - det)
        min_diff_idx = np.argmin(diferencias)
        min_diff = diferencias[min_diff_idx]
        
        if min_diff <= tolerancia_muestras and not qrs_det_emparejados[min_diff_idx]:
            TP += 1
            mis_qrs_emparejados[i] = True
            qrs_det_emparejados[min_diff_idx] = True
    
    # False Positives (tus detecciones no emparejadas)
    tp_idx=np.where(mis_qrs_emparejados)[0]
    fp_idx=np.where(~mis_qrs_emparejados)[0] #índices de detecciones falsas
    FP = np.sum(~mis_qrs_emparejados)
    
    # False Negatives (detecciones de referencia no emparejadas)
    fn_idx=np.where(~qrs_det_emparejados)[0]
    FN = np.sum(~qrs_det_emparejados)
    
    # Construir matriz de confusión
    matriz = np.array([
        [TP, FP],
        [FN, 0]  # TN generalmente no aplica en detección de eventos
    ])
    
    print("Matriz de Confusión:")
    print(f"           Predicho")
    print(f"           Sí    No")
    print(f"Real Sí:  [{TP:2d}   {FN:2d}]")
    print(f"Real No:  [{FP:2d}    - ]")
    print(f"\nTP: {TP}, FP: {FP}, FN: {FN}")

    # Calcular métricas de performance
    if TP + FP > 0:
        precision = TP / (TP + FP) #valor predictivo positivo
    else:
        precision = 0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    print(f"\nMétricas:")
    print(f"Precisión: {precision:.3f}")
    print(f"Sensibilidad: {recall:.3f}")
    print(f"F1-score: {f1_score:.3f}")
    
    return matriz, TP, FP, FN, tp_idx, fp_idx, fn_idx

#%% Parámetros y funciones
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
qrs_detections=mat_struct['qrs_detections'].flatten()
qrs_pattern1=mat_struct['qrs_pattern1'].flatten()
qrs_pattern1=qrs_pattern1.astype(float)
cant_muestras=N
fs=1000

# Spline cúbico
n=80
m=qrs_detections-n
valor=ecg_one_lead[m]
cubico=CubicSpline(y=valor, x=m) #interpolante
N_vector=np.arange(N)
bs=cubico(N_vector) #ruido estimado 
ecg_filt_bs=ecg_one_lead-bs # sustracción

# Mediana
size=199 #muestras, y en mitad toma el valor enterno en lugar de una cant muestras par
filt_med_200=sig.medfilt(volume=ecg_one_lead,kernel_size=size)
bm=sig.medfilt(volume=filt_med_200,kernel_size=size*3) #ruido estimado
ecg_filt_med=ecg_one_lead-bm # sustracción

# Filtro adaptado
patron=(qrs_pattern1-np.mean(qrs_pattern1)).flatten()
ecg_filt_adp=sig.lfilter(b=patron,a=1,x=ecg_one_lead)
ecg_filt_adp_abs=np.abs(ecg_filt_adp)
#distinta escala a patrón, entonces normalizo por desvio
ecg_filt_adp_abs_norm=ecg_filt_adp_abs/np.std(ecg_filt_adp_abs)
ecg_one_lead_norm=ecg_one_lead/np.std(ecg_one_lead)
#se ven en misma escala pero hay retardo pq es inherente de sist lineal no simétrico (FIR no simetrico), y lo compenso tomando desde otra muestra
ecg_filt_adp_abs_norm_demo=ecg_filt_adp_abs_norm[50:] #ver con 56
ecg_filt_adp_norm=ecg_filt_adp/np.std(ecg_filt_adp)
ecg_filt_adp_norm_demo=ecg_filt_adp_norm[50:]

mis_qrs_detections,prominencias=sig.find_peaks(x=ecg_filt_adp_abs_norm_demo,height=1,distance=302)
# plt.figure()
# plt.plot(ecg_one_lead_norm, color='blue')
# plt.plot(ecg_filt_adp_abs_norm_demo, color='green') #veo mínimo pico a detectar por diferencia SNR y encontramos dos picos más

matriz, tp, fp, fn, tp_idx, fp_idx, fn_idx = matriz_confusion_qrs(mis_qrs_detections, qrs_detections)

# Gráficos
plt.figure()
plt.title('Estimación de ruido con Mediana')
plt.plot(ecg_one_lead, color='blue', label='ECG')
plt.plot(bm, color='red', label='Estimación del ruido')
plt.legend()

plt.figure()
plt.title('ECG filtrado con Mediana')
plt.plot(ecg_one_lead, color='blue', label='ECG')
plt.plot(ecg_filt_med, color='green', label='ECG filtrado')
plt.legend()

plt.figure()
plt.title('Estimación de ruido con Spline')
plt.plot(ecg_one_lead, color='blue', label='ECG')
#plt.plot(ecg_filt_bs, color='green', label='ECG filtrado')
plt.plot(bs, color='red',  label='Estimación del ruido')
plt.scatter(m, valor, color='green', s=50, label="Puntos usados por spline")
#plt.vlines(qrs_detections, ymin=min(ecg_one_lead), ymax=max(ecg_one_lead), colors='m', linestyles= 'dashed', linewidth=0.1)
plt.legend()

plt.figure()
plt.title('ECG filtrado con Spline')
plt.plot(ecg_one_lead, color='blue', label='ECG')
plt.plot(ecg_filt_bs, color='green', label='ECG filtrado')
#plt.vlines(qrs_detections, ymin=min(ecg_one_lead), ymax=max(ecg_one_lead), colors='m', linestyles= 'dashed', linewidth=0.1)
plt.legend()

plt.figure()
plt.title('Estimación de ruido con filtro adaptado')
plt.plot(ecg_one_lead_norm, color='blue', label='ECG')
# plt.plot(ecg_filt_adp_abs_norm_demo, label='filt_adp', color='orange')
plt.plot(mis_qrs_detections[tp_idx],ecg_one_lead_norm[mis_qrs_detections[tp_idx]],'oy',label='TP', markersize=6)
plt.plot(mis_qrs_detections[fp_idx],ecg_one_lead_norm[mis_qrs_detections[fp_idx]],'dr',label='FP',markersize=6)
plt.plot(qrs_detections[fn_idx],ecg_one_lead_norm[qrs_detections[fn_idx]],'sg',label='FN',markersize=6)
plt.legend()

plt.figure()
plt.title('ECG con filtro adaptado')
plt.plot(ecg_filt_adp_norm_demo, label='Filto adaptado', color='orange')
plt.plot(ecg_one_lead_norm, color='blue', label='ECG', alpha=0.6)
plt.legend()


# estimaciones de ruido
plt.figure()
plt.title('Estimación de ruido con Filtro adaptado')
plt.plot(bm, color='red', label='Estimación del ruido mediana')
plt.plot(bs, color='blue',  label='Estimación del ruido spline')

#%% Bonus
D = 10 # Factor de diezmado

# La función aplica el filtro anti aliasing y reduce fs de 1000Hz a 100Hz
ecg_down = sig.decimate(ecg_one_lead, D)
fs_diezmada = fs / D

# Aplicación del filtro de mediana ajustando las ventanas para menos muestras
# ventana de 200 ms: en fs/M son 200*(fs/D)/1000 = (200*1000/10)/1000 = 20 muestras pero necesito impar
w_200 = 21 
w_600 = 61
b_diezmado = sig.medfilt(sig.medfilt(ecg_down, w_200), w_600)

# Interpolación para volver a la fs original
# índices de las muestras originales correspondientes a los puntos de la señal diezmada
indice = np.arange(len(b_diezmado)) * D

f_interp = interp1d(indice, b_diezmado, kind='cubic', fill_value="extrapolate") # Función de interpolación

b_multirate = f_interp(np.arange(len(ecg_one_lead))) # Estimación del ruido
ecg_filt_med_multi=ecg_one_lead-b_multirate

plt.figure()
plt.title('Estimación del ruido con diezmado y mediana')
plt.plot(ecg_one_lead, color='blue', label='ECG')
plt.plot(b_multirate, color='red', label='Estimación del ruido')
plt.ylabel('Adimensional')
plt.xlabel('Muestras (#)')
plt.legend()

plt.figure()
plt.title('ECG filtrado con diezmado y mediana')
plt.plot(ecg_one_lead, color='blue', label='ECG')
plt.plot(ecg_filt_med_multi, color='green', label='ECG filtrado')
plt.ylabel('Adimensional')
plt.xlabel('Muestras (#)')
plt.legend()

#%% EXTRA
# pasa bajos
# ecg_one_lead_lp=sig.lfilter(b=np.ones(111), a=1, x=ecg_filt_adp_abs_norm_demo)
# ecg_one_lead_lp_norm_demo=(ecg_one_lead_lp/np.std(ecg_one_lead_lp))[50:]
# plt.figure()
# plt.plot(ecg_one_lead_norm, color='red') 
# plt.plot(ecg_filt_adp_abs_norm_demo, color='blue')
# plt.plot(ecg_one_lead_lp_norm_demo, color='green')

# algoritmo Woody
# qrs_mat=np.array([ecg_one_lead[ii-60:ii+60] for ii in mis_qrs_detections])
# qrs_mat_norm=qrs_mat-(np.mean(qrs_mat,axis=1).reshape(-1,1))
# qrs_mat_norm_trans=qrs_mat_norm.transpose()
# plt.figure()
# plt.plot(qrs_mat_norm_trans)
# plt.plot(np.mean(qrs_mat,axis=0),'-w',lw=2) #latido medio
# mis_qrs_detections,prominencias=sig.find_peaks(x=ecg_filt_adp_abs_norm_demo,height=1,distance=300, prominence=(None,6), width=(19, 23))
# plt.figure()
# plt.hist(prominencias['widths'],bins=20) #distribución de anchuras 
# plt.figure()
# plt.plot(np.mean(qrs_mat,axis=0),'--w',lw=2) #latido medio