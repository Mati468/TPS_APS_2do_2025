# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:39:59 2025

@author: Matías Cassino

CONSIGNA: 
"""
#%% Módulos
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from matplotlib import patches
from pytc2.sistemas_lineales import plot_plantilla

#%% Funciones
plt.close('all')

def rta_frec_fase_y_retardo_fir(b,worN,fs,filter_type,wp,ws,alpha_p,alpha_s,f_aprox):
    # worN define puntos de evaluación logarítmicamente espaciados delde 10^(-2) a 10^(2) de RTA frec hasta Nyquist
    w,h=sig.freqz(b=b,worN=worN, fs=fs) 
    phase=np.unwrap(np.angle(h)) 
    w_rad=w/(fs/2)*np.pi # 
    gd=-np.diff(phase)/np.diff(w_rad)

    plt.figure(figsize=(12,10))

    #Magnitud
    plt.subplot(3,1,1)
    plt.plot(w, 20*np.log10(abs(h)),label=f_aprox)
    plot_plantilla(filter_type=filter_type,fpass=wp,ripple=alpha_p,fstop=ws,attenuation=alpha_s,fs=fs) 
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # Fase
    plt.subplot(3,1,2)
    plt.plot(w, phase,label=f_aprox)
    plt.title('Fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    plt.legend()


    # Retardo de grupo
    plt.subplot(3,1,3)
    plt.plot(w[:-1], gd,label=f_aprox)
    plt.title('Retardo de Grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [# muestras]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    plt.tight_layout()

    return w,h,phase,gd

def rta_frec_fase_y_retardo_iir(sos,worN,fs,filter_type,wp,ws,alpha_p,alpha_s,f_aprox):
    # worN define puntos de evaluación logarítmicamente espaciados delde 10^(-2) a 10^(2) de RTA frec hasta Nyquist
    w,h=sig.freqz_sos(sos=sos,worN=worN, fs=fs) 
    phase=np.unwrap(np.angle(h)) 
    w_rad=w/(fs/2)*np.pi # 
    gd=-np.diff(phase)/np.diff(w_rad)

    plt.figure(figsize=(12,10))

    #Magnitud
    plt.subplot(3,1,1)
    plt.plot(w, 2*(20*np.log10(abs(h))),label=f_aprox)
    plot_plantilla(filter_type=filter_type,fpass=wp,ripple=alpha_p,fstop=ws,attenuation=alpha_s,fs=fs) 
    plt.title('Respuesta bidireccional en Magnitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # Fase
    plt.subplot(3,1,2)
    plt.plot(w, phase,label=f_aprox)
    plt.title('Fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    plt.legend()


    # Retardo de grupo
    plt.subplot(3,1,3)
    plt.plot(w[:-1], gd,label=f_aprox)
    plt.title('Retardo de Grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [# muestras]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    plt.tight_layout()

    return w,h,phase,gd

#%% Plantilla de diseño
fs=1000 # Hz
nyq_frec=fs/2
wp= (0.9, 35) #  frecuencia de corte/paso [Hz]
ws= (0.1, 37.5) #  frecuencia de stop [Hz]

ripple=1 #[dB]
atenuacion=40 #[dB]
alpha_p_FIR= ripple
alpha_s_FIR=atenuacion
# Divido por dos porque lo aplico bidireccional
alpha_p_IIR= ripple/2 #atenuacion maxima a la w_corte = alfa_max (perdidas en banda paso [db])
alpha_s_IIR= atenuacion/2 #atenuacion minima a la w_paso = alfa_min (minima atenuacion requerida en bamda de paso[db])

# plantilla normalizada a Nyquist en dB
frecs = np.array([0.0, ws[0], wp[0], wp[1], ws[1], nyq_frec])/nyq_frec

filter_type=['lowpass', 'highpass', 'bandpass', 'bandstop']
f_aprox_FIR=['Win','LS','PM']
f_aprox_IIR=['butter','cheby1','cheby1','cauer']

#%% Filtros FIR
worN=np.logspace(-2,2,1000)

# pesos_ls=[95,40,5]
pesos_ls=[70,0.3,20]
# pesos_ls=[90,10,60]

deseado_ls=[0,0, 1,1, 0,0]
cant_coef_ls=1129 #impar pq es tipo 2
retardo_ls=(cant_coef_ls-1)//2
fir_ls=sig.firls(numtaps=cant_coef_ls, bands=frecs, desired=deseado_ls,weight=pesos_ls)

pesos_pm=[88,7,90]
# pesos_pm=[0.5,0.005,80]
# pesos_pm=[20,10,90]
cant_coef_pm=1061 #impar pq es tipo 2
retardo_pm=(cant_coef_pm-1)//2
deseado_pm=[0,1,0]
fir_pm=sig.remez(numtaps=cant_coef_pm, bands=frecs, desired=deseado_pm, fs=2,weight=pesos_pm)
#nfreq define cant puntos de frecuencia para interpolar (mejor resolución)

#%% Filtros IIR para aproximaciones de módulo
mi_sos_butter=sig.iirdesign(wp=wp,ws=ws,gpass=alpha_p_IIR,gstop=alpha_s_IIR,analog=False,ftype=f_aprox_IIR[0],output='sos',fs=fs)

mi_sos_cauer=sig.iirdesign(wp=wp,ws=ws,gpass=alpha_p_IIR,gstop=alpha_s_IIR,analog=False,ftype=f_aprox_IIR[3],output='sos',fs=fs)

#%% Gráficos
w_ls,h_ls,phase_ls,gd_ls=rta_frec_fase_y_retardo_fir(b=fir_ls,worN=worN,fs=fs,filter_type=filter_type[2],wp=wp,ws=ws,alpha_p=alpha_p_FIR,alpha_s=alpha_s_FIR,f_aprox=f_aprox_FIR[1])

w_pm,h_pm,phase_pm,gd_pm=rta_frec_fase_y_retardo_fir(b=fir_pm,worN=worN,fs=fs,filter_type=filter_type[2],wp=wp,ws=ws,alpha_p=alpha_p_FIR,alpha_s=alpha_s_FIR,f_aprox=f_aprox_FIR[2])

w_butter,h_butter,phase_butter,gd_butter=rta_frec_fase_y_retardo_iir(sos=mi_sos_butter,worN=worN,fs=fs,filter_type=filter_type[2],wp=wp,ws=ws,alpha_p=alpha_p_FIR,alpha_s=alpha_s_IIR,f_aprox=f_aprox_FIR[0])

w_cauer,h_cauer,phase_cauer,gd_cauer=rta_frec_fase_y_retardo_iir(sos=mi_sos_cauer,worN=worN,fs=fs,filter_type=filter_type[2],wp=wp,ws=ws,alpha_p=alpha_p_FIR,alpha_s=alpha_s_IIR,f_aprox=f_aprox_FIR[0])

#################
# Lectura de ECG sin ruido
#################

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten() #registro con fs=1khz
heartbeat_pattern1=mat_struct['heartbeat_pattern1'].flatten() # latido normal
heartbeat_pattern2=mat_struct['heartbeat_pattern2'].flatten()# latido ventricular

N = len(ecg_one_lead)
cant_muestras=N

ecg_filt_butter=sig.sosfiltfilt(mi_sos_butter,ecg_one_lead)
ecg_filt_cauer=sig.sosfiltfilt(mi_sos_cauer,ecg_one_lead)

ecg_filt_ls=sig.lfilter(b=fir_ls, a=1 ,x=ecg_one_lead)
ecg_filt_pm=sig.lfilter(b=fir_pm, a=1 ,x=ecg_one_lead)


# plt.figure()
# plt.plot(heartbeat_pattern1, color='green',label='Normal')
# plt.plot(heartbeat_pattern2, color='blue',label='Ventricular')
# plt.xlabel('Tiempo [ms]')
# plt.ylabel('Amplitud normalizada')

##################################
# Regiones de interés sin ruido #
##################################
 
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='Cauer')
    plt.plot(zoom_region, ecg_filt_pm[zoom_region + retardo_pm], label='FIR PM')
    plt.plot(zoom_region, ecg_filt_ls[zoom_region + retardo_ls], label='FIR LS')
   
    plt.title('ECG sin ruido example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
###################################
# Regiones de interés con ruido #
###################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='Cauer')    
    plt.plot(zoom_region, ecg_filt_ls[zoom_region + retardo_ls], label='FIR LS')
    plt.plot(zoom_region, ecg_filt_pm[zoom_region + retardo_pm], label='FIR PM')
   
    plt.title('ECG con ruido example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
