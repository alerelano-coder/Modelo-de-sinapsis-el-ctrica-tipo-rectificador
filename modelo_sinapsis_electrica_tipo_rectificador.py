#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELO CONEXINA-36 - ANALISIS TFG COMPLETO (v4)
Sinapsis electricas con rectificacion debil

Autora: Alejandra Relano
Fecha: 2026-02-11

Estructura:
  Bloque A: Caracterizacion de neurona aislada
    A1. Duracion minima de pulso para generar 1 PA
    A2. Curva I-V (corriente vs voltaje pico)
    A3. Caracterizacion del PA (pulso corto, corriente continua, trenes)

  Bloque B: Neuronas acopladas (sinapsis electrica)
    Simulaciones para N = [0, 30, 38, 45, 70, 100, 1000] x [rect, no_rect]
    Metricas: latencia pico-pico, latencia onset-onset, amplitud, CC%, duracion, eficacia

  Bloque C: Filtro paso bajo
    CC% vs frecuencia de estimulacion para cada N

Salida: CSV en resultados/ para exportar a Excel
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.signal import find_peaks
import os
from datetime import datetime

# ==============================================================================
# CONFIGURACION
# ==============================================================================

DIR_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resultados')
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Parametros Hodgkin-Huxley
g_Na_max = 120.0  # mS/cm2
g_K_max = 36.0    # mS/cm2
g_L = 0.3         # mS/cm2
E_Na = 45.0       # mV
E_K = -82.0       # mV
C_m = 1.0         # uF/cm2
V_rest = -70.0    # mV

# Gap junction
Ej = 0.0          # mV (no selectividad ionica)
gamma_0 = 15.0    # pS (conductancia unitaria canal Cx36)
A_n = 1000.0      # um2 (area neuronal)

# Rectificacion Cx36 (coeficientes ecuacion cubica)
COEF = {
    'a': 4.2e-8,
    'b': 1.48e-5,
    'c': -1.51e-4,
    'd': 0.9984
}

# Frecuencias de analisis para filtro paso bajo
FREQ_COMPLETO = [1, 8, 40, 55, 70, 80, 100]  # Hz

# Niveles de canales
N_VALORES = [0, 30, 38, 45, 70, 100, 1000]

def gj_from_N(N):
    """Calcula conductancia macroscopica gj (nS) desde numero de canales N."""
    return N * gamma_0 / 1000.0
# N=0->0.0, N=30->0.45, N=38->0.57, N=45->0.675, N=70->1.05, N=100->1.5, N=1000->15.0
# Estimulo
AMP_SUPRAUMBRAL = 35.0  # uA/cm2

# Busqueda binaria duracion minima
DURACION_MIN_BUSQUEDA = 0.05  # ms
DURACION_MAX_BUSQUEDA = 10.0  # ms
TOLERANCIA_BUSQUEDA = 0.01    # ms

# Barrido I-V
IV_AMP_MIN = 0.0       # uA/cm2
IV_AMP_MAX = 45.0      # uA/cm2
IV_AMP_PASO = 1.0      # uA/cm2

# Corriente continua larga
DURACION_DC_LARGA = 300.0  # ms


# ==============================================================================
# FUNCIONES DE GATING (Hodgkin-Huxley)
# ==============================================================================

def alpha_m(V):
    """Tasa de apertura de la compuerta m (activacion Na+). Maneja singularidad en V=-45 mV."""
    x = V + 45.0
    if isinstance(V, np.ndarray):
        result = np.zeros_like(V, dtype=float)
        mask = np.abs(x) < 1e-6
        result[mask] = 1.0
        result[~mask] = 0.1 * x[~mask] / (1.0 - np.exp(-x[~mask] / 10.0))
        return result
    return 1.0 if abs(x) < 1e-6 else 0.1 * x / (1.0 - np.exp(-x / 10.0))

def beta_m(V): return 4.0 * np.exp(-(V + 70.0) / 18.0)
def alpha_h(V): return 0.07 * np.exp(-(V + 70.0) / 20.0)
def beta_h(V): return 1.0 / (1.0 + np.exp(-(V + 40.0) / 10.0))

def alpha_n(V):
    """Tasa de apertura de la compuerta n (activacion K+). Maneja singularidad en V=-60 mV."""
    x = V + 60.0
    if isinstance(V, np.ndarray):
        result = np.zeros_like(V, dtype=float)
        mask = np.abs(x) < 1e-6
        result[mask] = 0.1
        result[~mask] = 0.01 * x[~mask] / (1.0 - np.exp(-x[~mask] / 10.0))
        return result
    return 0.1 if abs(x) < 1e-6 else 0.01 * x / (1.0 - np.exp(-x / 10.0))

def beta_n(V): return 0.125 * np.exp(-(V + 70.0) / 80.0)

m0 = alpha_m(V_rest) / (alpha_m(V_rest) + beta_m(V_rest))
h0 = alpha_h(V_rest) / (alpha_h(V_rest) + beta_h(V_rest))
n0 = alpha_n(V_rest) / (alpha_n(V_rest) + beta_n(V_rest))

# ==============================================================================
# E_L - COMPROMISO ESTABILIDAD/EXCITABILIDAD
# ==============================================================================

E_L = -65.0  # mV

I_Na_rest = g_Na_max * m0**3 * h0 * (V_rest - E_Na)
I_K_rest = g_K_max * n0**4 * (V_rest - E_K)
I_L_rest = g_L * (V_rest - E_L)
I_total_rest = I_Na_rest + I_K_rest + I_L_rest

print(f"PARAMETROS:")
print(f"  E_L = {E_L:.3f} mV")
print(f"  Corrientes en reposo (V={V_rest} mV):")
print(f"    I_Na = {I_Na_rest:.6f} uA/cm2")
print(f"    I_K  = {I_K_rest:.6f} uA/cm2")
print(f"    I_L  = {I_L_rest:.6f} uA/cm2")
print(f"    I_total = {I_total_rest:.6f} uA/cm2")
print(f"  {'[OK] Estable' if abs(I_total_rest) < 5.0 else '[AVISO] Puede disparar'}")
print()

# ==============================================================================
# VERIFICACION PROPIEDADES Cx36
# ==============================================================================

def _G_raw_verif(Vj):
    c = COEF
    return c['a']*Vj**3 + c['b']*Vj**2 + c['c']*Vj + c['d']

print("VERIFICACION Cx36:")
print("  1) Canales Cx36 SIEMPRE ABIERTOS (sin gating m,h,n)")
print()
print("  2) Factor de rectificacion G(Vj) NORMALIZADO:")
_G0_norm = _G_raw_verif(0) / COEF['d']
_G100_norm = _G_raw_verif(100) / COEF['d']
_G110_norm = _G_raw_verif(110) / COEF['d']
print(f"     G(0 mV)   = {_G0_norm:.6f}  (debe ser 1.000000)")
print(f"     G(+100 mV)= {_G100_norm:.6f}  (+{(_G100_norm-1)*100:.1f}%)")
print(f"     G(+110 mV)= {_G110_norm:.6f}  (+{(_G110_norm-1)*100:.1f}%)")
print()
print(f"  3) N_VALORES = {N_VALORES}")
print(f"     gj correspondientes: {[gj_from_N(N) for N in N_VALORES]} nS")
print()
del _G_raw_verif, _G0_norm, _G100_norm, _G110_norm

# ==============================================================================
# FUNCIONES DE CONDUCTANCIA
# ==============================================================================

def G_rect(Vj):
    """
    Factor de rectificacion G(Vj) - ecuacion cubica NORMALIZADA.
    G(0) = 1.0 exactamente, para que a Vj=0 rect = no_rect.
    """
    c = COEF
    G_raw = c['a']*Vj**3 + c['b']*Vj**2 + c['c']*Vj + c['d']
    G = G_raw / c['d']
    return max(0.1, G)

def calcular_gj(Vj, N, modelo='rect'):
    """
    Conductancia juncional macroscopica (nS).
    gj = N * gamma_0 * G(Vj) / 1000
    """
    G = G_rect(Vj) if modelo == 'rect' else 1.0
    gj = N * gamma_0 * G / 1000.0
    return max(0.0, gj)

# ==============================================================================
# SISTEMAS DE ECUACIONES DIFERENCIALES
# ==============================================================================

def sistema_no_rectificador(y, t, gj_nS, Ej_param, I_func):
    """
    Sistema HH con gap junction NO RECTIFICADORA (tipo artificial).
    Conductancia constante G=1.0. Canales Cx36 siempre abiertos.
    """
    V1, m1, h1, n1, V2, m2, h2, n2 = y
    I_ext = I_func(t)

    Vj = V1 - V2
    Ij = gj_nS * Vj  # pA
    Ij_dens = Ij * 1e-6 / (A_n * 1e-8)  # uA/cm2

    I_Na1 = g_Na_max * m1**3 * h1 * (V1 - E_Na)
    I_K1 = g_K_max * n1**4 * (V1 - E_K)
    I_L1 = g_L * (V1 - E_L)

    dV1_dt = (I_ext - I_Na1 - I_K1 - I_L1 - Ij_dens) / C_m
    dm1_dt = alpha_m(V1) * (1 - m1) - beta_m(V1) * m1
    dh1_dt = alpha_h(V1) * (1 - h1) - beta_h(V1) * h1
    dn1_dt = alpha_n(V1) * (1 - n1) - beta_n(V1) * n1

    I_Na2 = g_Na_max * m2**3 * h2 * (V2 - E_Na)
    I_K2 = g_K_max * n2**4 * (V2 - E_K)
    I_L2 = g_L * (V2 - E_L)

    dV2_dt = (-I_Na2 - I_K2 - I_L2 + Ij_dens) / C_m
    dm2_dt = alpha_m(V2) * (1 - m2) - beta_m(V2) * m2
    dh2_dt = alpha_h(V2) * (1 - h2) - beta_h(V2) * h2
    dn2_dt = alpha_n(V2) * (1 - n2) - beta_n(V2) * n2

    return [dV1_dt, dm1_dt, dh1_dt, dn1_dt, dV2_dt, dm2_dt, dh2_dt, dn2_dt]

def sistema_rectificador(y, t, N, Ej_param, I_func):
    """
    Sistema HH con gap junction RECTIFICADORA (Cx36).
    gj(Vj) = N * gamma_0 * G(Vj) / 1000. Canales siempre abiertos.
    """
    V1, m1, h1, n1, V2, m2, h2, n2 = y
    I_ext = I_func(t)

    Vj = V1 - V2
    gj_efectiva = calcular_gj(Vj, N, 'rect')
    Ij = gj_efectiva * Vj  # pA
    Ij_dens = Ij * 1e-6 / (A_n * 1e-8)  # uA/cm2

    I_Na1 = g_Na_max * m1**3 * h1 * (V1 - E_Na)
    I_K1 = g_K_max * n1**4 * (V1 - E_K)
    I_L1 = g_L * (V1 - E_L)

    dV1_dt = (I_ext - I_Na1 - I_K1 - I_L1 - Ij_dens) / C_m
    dm1_dt = alpha_m(V1) * (1 - m1) - beta_m(V1) * m1
    dh1_dt = alpha_h(V1) * (1 - h1) - beta_h(V1) * h1
    dn1_dt = alpha_n(V1) * (1 - n1) - beta_n(V1) * n1

    I_Na2 = g_Na_max * m2**3 * h2 * (V2 - E_Na)
    I_K2 = g_K_max * n2**4 * (V2 - E_K)
    I_L2 = g_L * (V2 - E_L)

    dV2_dt = (-I_Na2 - I_K2 - I_L2 + Ij_dens) / C_m
    dm2_dt = alpha_m(V2) * (1 - m2) - beta_m(V2) * m2
    dh2_dt = alpha_h(V2) * (1 - h2) - beta_h(V2) * h2
    dn2_dt = alpha_n(V2) * (1 - n2) - beta_n(V2) * n2

    return [dV1_dt, dm1_dt, dh1_dt, dn1_dt, dV2_dt, dm2_dt, dh2_dt, dn2_dt]

# ==============================================================================
# FUNCIONES DE ESTIMULO
# ==============================================================================

def tren_pulsos(freq_Hz, n_pulsos, amp_uA_cm2, t_dur=2.0, t_start=50.0):
    """Tren de pulsos rectangulares (escalones de corriente)"""
    periodo = 1000.0 / freq_Hz  # ms
    def I_func(t):
        for i in range(n_pulsos):
            t_ini = t_start + i * periodo
            t_fin = t_ini + t_dur
            if t_ini <= t < t_fin:
                return amp_uA_cm2
        return 0.0
    return I_func

def corriente_continua(amp_uA_cm2, t_start=50.0, duracion=300.0):
    """Escalon de corriente continua (DC step)."""
    t_fin = t_start + duracion
    def I_func(t):
        return amp_uA_cm2 if t_start <= t < t_fin else 0.0
    return I_func

def pulso_unico(amp_uA_cm2, t_dur, t_start=50.0):
    """Pulso de corriente unico de duracion especifica."""
    return tren_pulsos(100.0, 1, amp_uA_cm2, t_dur=t_dur, t_start=t_start)

# ==============================================================================
# FUNCIONES DE DETECCION
# ==============================================================================

def detectar_PAs_riguroso(V, m, h, t):
    """Deteccion rigurosa de PAs (no spikelets)"""
    dV_dt = np.gradient(V, t)
    candidatos, _ = find_peaks(V, height=0, distance=50)

    PAs_verdaderos = []
    for idx in candidatos:
        if V[idx] <= 0:
            continue
        idx_ini = max(0, idx - 20)
        idx_fin = min(len(V), idx + 20)
        if np.max(dV_dt[idx_ini:idx_fin]) <= 100:
            continue
        if m[idx]**3 * h[idx] <= 0.1:
            continue
        PAs_verdaderos.append(idx)

    return len(PAs_verdaderos), PAs_verdaderos

def detectar_spikelets(V, m, h, t, umbral_mV=2.0):
    """Detecta picos subumbrales (spikelets): superan V_rest+umbral pero no cumplen criterios de PA.
    Devuelve (n_spikelets, indices_spikelets)."""
    _, idx_PAs = detectar_PAs_riguroso(V, m, h, t)
    set_PAs = set(idx_PAs)

    # Buscar todos los picos por encima de V_rest + umbral
    candidatos, _ = find_peaks(V, height=V_rest + umbral_mV, distance=50)

    spikelets = []
    for idx in candidatos:
        if idx not in set_PAs:
            spikelets.append(idx)

    return len(spikelets), spikelets


def medir_deflexion_subumbral(t, V1, m1, h1, V2, m2, h2,
                               ventana_max_ms=100.0, umbral_mV=1.0):
    """Detecta spikelets reales en V2 tras cada PA presinaptico, con mayor sensibilidad
    que detectar_spikelets (umbral 1 mV vs 2 mV). Mismo criterio de pico (find_peaks,
    distance=50) para garantizar que son eventos fisiologicos reales y no artefactos.
    La busqueda dentro de la ventana de cada PA pre evita contar residuos de ciclos
    anteriores en trenes de alta frecuencia. Excluye ventanas con PA completo post.
    Devuelve (n_spikelets_extendidos, lista_amplitudes_mV)."""
    _, idx_pre = detectar_PAs_riguroso(V1, m1, h1, t)
    _, idx_post_PA = detectar_PAs_riguroso(V2, m2, h2, t)
    set_post_PA = set(idx_post_PA)

    amplitudes = []
    for k, i_pre in enumerate(idx_pre):
        # Fin de ventana: inicio del siguiente PA pre o ventana_max_ms
        if k + 1 < len(idx_pre):
            i_fin = idx_pre[k + 1]
        else:
            t_fin = t[i_pre] + ventana_max_ms
            i_fin = min(int(np.searchsorted(t, t_fin)), len(t) - 1)

        if i_fin <= i_pre:
            continue

        # Excluir ventanas con PA completo en postsinaptico
        if any(idx in set_post_PA for idx in range(i_pre, i_fin)):
            continue

        # Mismos criterios que detectar_spikelets pero umbral 1 mV en lugar de 2 mV
        V2_ventana = V2[i_pre:i_fin]
        picos, _ = find_peaks(V2_ventana, height=V_rest + umbral_mV, distance=50)
        for p in picos:
            amplitudes.append(V2_ventana[p] - V_rest)

    return len(amplitudes), amplitudes


def detectar_inicio_PA(V, t, dV_umbral=20.0):
    """Localiza el inicio (onset) de cada PA buscando hacia atras donde dV/dt cae bajo el umbral.
    Devuelve (onset_indices, onset_times)."""
    dV_dt = np.gradient(V, t)
    picos, _ = find_peaks(V, height=0, distance=50)

    onset_indices = []
    onset_times = []

    for idx_pico in picos:
        if V[idx_pico] <= 0:
            continue
        # Buscar hacia atras desde el pico hasta que dV/dt < umbral
        idx_busq = idx_pico
        while idx_busq > 0 and dV_dt[idx_busq] > dV_umbral:
            idx_busq -= 1
        idx_onset = min(idx_busq + 1, idx_pico)
        onset_indices.append(idx_onset)
        onset_times.append(t[idx_onset])

    return onset_indices, onset_times


def detectar_inicios_respuesta(V, m, h, t, dV_umbral_PA=20.0, dV_umbral_spk=0.5):
    """Detecta el instante de inicio de PAs y spikelets usando umbrales de dV/dt diferenciados.
    Devuelve (onset_indices, onset_times) para todas las respuestas en V."""
    dV_dt = np.gradient(V, t)
    _, idx_PAs = detectar_PAs_riguroso(V, m, h, t)
    _, idx_spk = detectar_spikelets(V, m, h, t)

    set_PAs = set(idx_PAs)
    all_peaks = sorted(list(idx_PAs) + list(idx_spk))

    onset_indices = []
    onset_times = []

    for pk in all_peaks:
        if pk in set_PAs:
            # PA: buscar onset via dV/dt
            idx_busq = pk
            while idx_busq > 0 and dV_dt[idx_busq] > dV_umbral_PA:
                idx_busq -= 1
            idx_onset = min(idx_busq + 1, pk)
        else:
            # Spikelet: buscar onset via dV/dt (inicio del cambio de potencial)
            idx_busq = pk
            while idx_busq > 0 and dV_dt[idx_busq] > dV_umbral_spk:
                idx_busq -= 1
            idx_onset = min(idx_busq + 1, pk)

        onset_indices.append(idx_onset)
        onset_times.append(t[idx_onset])

    return onset_indices, onset_times


# ==============================================================================
# FUNCIONES DE ANALISIS
# ==============================================================================

def calcular_CC(V1, V2, t=None, t_start=100):
    """CC = (DeltaV_post / DeltaV_pre) * 100%"""
    if t is not None:
        idx = t >= t_start
        V1_ss = V1[idx]
        V2_ss = V2[idx]
    else:
        V1_ss = V1
        V2_ss = V2

    V1_amp = np.max(V1_ss) - np.min(V1_ss)
    V2_amp = np.max(V2_ss) - np.min(V2_ss)

    if V1_amp < 1e-6:
        return 0.0

    return (V2_amp / V1_amp) * 100.0

def _picos_respuesta_post(V2, m2, h2, t):
    """Indices de TODOS los picos de respuesta en V2: PAs + spikelets, ordenados."""
    _, idx_PAs = detectar_PAs_riguroso(V2, m2, h2, t)
    _, idx_spk = detectar_spikelets(V2, m2, h2, t)
    idx_all = sorted(list(idx_PAs) + list(idx_spk))
    return np.array(idx_all) if idx_all else np.array([], dtype=int)


def calcular_latencia_pico(t, V1, m1, h1, V2, m2, h2):
    """Latencia promedio pico-a-pico entre PA pre y respuesta post (PA o spikelet) (ms)"""
    _, idx_PA1 = detectar_PAs_riguroso(V1, m1, h1, t)
    idx_post = _picos_respuesta_post(V2, m2, h2, t)

    if len(idx_PA1) == 0 or len(idx_post) == 0:
        return np.nan

    latencias = []
    ventana_max_ms = 10.0
    t_post = t[idx_post]

    for idx1 in idx_PA1:
        t1 = t[idx1]
        ventana = (t_post > t1) & (t_post < t1 + ventana_max_ms)
        if np.any(ventana):
            latencias.append(t_post[ventana][0] - t1)

    return np.mean(latencias) if len(latencias) > 0 else np.nan

def calcular_latencias_pico_individuales(t, V1, m1, h1, V2, m2, h2):
    """Retorna lista de latencias pico-a-pico individuales (PA o spikelet en V2)"""
    _, idx_PA1 = detectar_PAs_riguroso(V1, m1, h1, t)
    idx_post = _picos_respuesta_post(V2, m2, h2, t)

    if len(idx_PA1) == 0 or len(idx_post) == 0:
        return []

    latencias = []
    ventana_max_ms = 10.0
    t_post = t[idx_post]
    for idx1 in idx_PA1:
        t1 = t[idx1]
        ventana = (t_post > t1) & (t_post < t1 + ventana_max_ms)
        if np.any(ventana):
            latencias.append(t_post[ventana][0] - t1)
    return latencias

def calcular_latencia_onset(t, V1, V2, m1, h1, m2, h2,
                             dV_umbral=10.0):
    """Latencia onset-to-onset (Tipo 2) entre inicio del PA pre y inicio de la respuesta post.
    Busqueda en 2 fases via dV/dt; restriccion fisica onset_V2 >= onset_V1. Devuelve (media, lista)."""
    _, idx_PA1 = detectar_PAs_riguroso(V1, m1, h1, t)
    idx_post = _picos_respuesta_post(V2, m2, h2, t)

    if len(idx_PA1) == 0 or len(idx_post) == 0:
        return np.nan, []

    dV1 = np.gradient(V1, t)
    dV2 = np.gradient(V2, t)
    dt = t[1] - t[0]
    max_back = int(10.0 / dt)

    # Umbral para considerar dV/dt como "reposo" en V2
    dV_reposo = 0.5  # mV/ms

    latencias = []
    t_post = t[idx_post]

    for idx1 in idx_PA1:
        t1_pk = t[idx1]
        ventana = (t_post > t1_pk) & (t_post < t1_pk + 10.0)
        if not np.any(ventana):
            continue
        idx2 = idx_post[ventana][0]

        # --- V1 onset: inicio del upstroke rapido ---
        # Busqueda en 2 fases hacia atras desde el pico:
        #   Fase 1: saltar zona del pico (dV/dt < umbral)
        #   Fase 2: atravesar upstroke rapido (dV/dt >= umbral)
        #   Parar cuando dV/dt < umbral = inicio del upstroke
        lim1 = max(0, idx1 - max_back)
        ib = idx1 - 1
        while ib > lim1 and dV1[ib] < dV_umbral:
            ib -= 1
        while ib > lim1 and dV1[ib] >= dV_umbral:
            ib -= 1
        t1_on = t[ib + 1]

        # --- V2 onset: inicio de la desviacion por gap junction ---
        # Misma busqueda en 2 fases, pero con umbral mas bajo (dV_reposo)
        # para captar tanto PA como spikelets pequenos:
        #   Fase 1: saltar zona del pico (dV/dt < dV_reposo)
        #   Fase 2: atravesar la subida (dV/dt >= dV_reposo)
        #   Parar cuando dV/dt < dV_reposo = inicio de la desviacion
        lim2 = max(0, idx2 - max_back)
        ib = idx2 - 1
        # Fase 1: saltar zona del pico donde dV/dt es pequeno
        while ib > lim2 and dV2[ib] < dV_reposo:
            ib -= 1
        # Fase 2: retroceder por la subida (dV/dt >= dV_reposo)
        while ib > lim2 and dV2[ib] >= dV_reposo:
            ib -= 1
        t2_on = t[ib + 1]

        # Restriccion fisica: onset_V1 <= onset_V2 < peak_V2
        if t2_on < t1_on:
            t2_on = t1_on
        if t2_on >= t[idx2]:
            continue

        latencias.append(t2_on - t1_on)

    return (np.mean(latencias) if latencias else np.nan), latencias


def calcular_frecuencia_natural(V, m, h, t):
    """Calcula frecuencia media de disparo y periodo refractario minimo desde un tren de PAs DC.
    Devuelve (freq_Hz, intervalos_ms, periodo_refractario_ms). Requiere al menos 2 PAs."""
    n_PA, idx_PA = detectar_PAs_riguroso(V, m, h, t)

    if n_PA < 2:
        return 0.0, [], np.nan

    t_picos = t[idx_PA]
    intervalos = np.diff(t_picos).tolist()

    freq_Hz = 1000.0 / np.mean(intervalos)
    periodo_refractario = np.min(intervalos)

    return freq_Hz, intervalos, periodo_refractario

def extraer_parametros_AP(V, m, h, t, V_umbral=-20.0):
    """Extrae amplitud, duracion (sobre V_umbral=-20 mV) y tiempo de pico de cada PA detectado.
    Devuelve lista de dicts con campos t_pico_ms, V_pico_mV, V_base_mV, amplitud_mV, duracion_ms."""
    n_PA, idx_PA = detectar_PAs_riguroso(V, m, h, t)
    resultados = []

    for idx in idx_PA:
        V_pico = V[idx]
        t_pico = t[idx]

        idx_base = max(0, idx - 200)
        V_base = np.min(V[idx_base:idx])
        amplitud = V_pico - V_base

        idx_ini_busq = max(0, idx - 300)
        idx_fin_busq = min(len(V), idx + 300)
        segmento = V[idx_ini_busq:idx_fin_busq]
        t_segmento = t[idx_ini_busq:idx_fin_busq]

        sobre_umbral = segmento > V_umbral
        if np.any(sobre_umbral):
            indices_sobre = np.where(sobre_umbral)[0]
            duracion = t_segmento[indices_sobre[-1]] - t_segmento[indices_sobre[0]]
        else:
            duracion = np.nan

        resultados.append({
            't_pico_ms': t_pico,
            'V_pico_mV': V_pico,
            'V_base_mV': V_base,
            'amplitud_mV': amplitud,
            'duracion_ms': duracion
        })

    return resultados

# ==============================================================================
# EXTRACCION COMPLETA DE METRICAS
# ==============================================================================

def extraer_metricas_completas(t, sol, N, modelo_nombre):
    """Calcula CC, eficacia, latencias (pico y onset), amplitudes, spikelets y tipo de respuesta.
    Devuelve dict con todas las metricas y los arrays temporales V1, V2."""
    V1, m1, h1, n1 = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    V2, m2, h2, n2 = sol[:, 4], sol[:, 5], sol[:, 6], sol[:, 7]

    n_PA_pre, _ = detectar_PAs_riguroso(V1, m1, h1, t)
    n_PA_post, _ = detectar_PAs_riguroso(V2, m2, h2, t)

    # Spikelets en post (picos que no son PAs completos)
    n_spikelets, idx_spikelets = detectar_spikelets(V2, m2, h2, t)
    amp_spikelets = [V2[i] - V_rest for i in idx_spikelets] if idx_spikelets else []

    # Deflexion subumbral: cualquier respuesta de V2 tras un PA pre sin PA post completo
    n_subumbrales, amp_subumbrales = medir_deflexion_subumbral(t, V1, m1, h1, V2, m2, h2)

    params_pre = extraer_parametros_AP(V1, m1, h1, t)
    params_post = extraer_parametros_AP(V2, m2, h2, t)

    # Latencia tipo 1 (pico-a-pico): PA_pre -> respuesta_post (PA o spikelet)
    lats_pico = calcular_latencias_pico_individuales(t, V1, m1, h1, V2, m2, h2)
    lat_pico_media = np.mean(lats_pico) if lats_pico else np.nan

    # Latencia tipo 2 (onset-to-onset): inicio PA_pre -> inicio respuesta_post
    lat_onset_media, lats_onset = calcular_latencia_onset(t, V1, V2, m1, h1, m2, h2)

    # CC%
    CC = calcular_CC(V1, V2, t=t, t_start=50.0)
    if n_PA_pre > 0 and n_PA_post > 0:
        CC = min(CC, 100.0)

    # Eficacia
    eficacia = (n_PA_post / n_PA_pre * 100) if n_PA_pre > 0 else 0.0

    # Tipo respuesta
    if n_PA_pre > 0 and n_PA_post > 0:
        tipo = 'PA_transmitido'
    elif n_PA_pre > 0 and n_spikelets > 0:
        tipo = 'spikelet'
    elif n_PA_pre > 0:
        tipo = 'sin_respuesta'
    else:
        tipo = 'sin_PA_pre'

    # Amplitudes y duraciones medias
    amp_pre = np.mean([p['amplitud_mV'] for p in params_pre]) if params_pre else np.nan
    amp_post = np.mean([p['amplitud_mV'] for p in params_post]) if params_post else np.nan
    dur_pre = np.mean([p['duracion_ms'] for p in params_pre]) if params_pre else np.nan
    dur_post = np.mean([p['duracion_ms'] for p in params_post]) if params_post else np.nan

    return {
        'N': N,
        'gj_nS': gj_from_N(N),
        'modelo': modelo_nombre,
        'n_PA_pre': n_PA_pre,
        'n_PA_post': n_PA_post,
        'n_spikelets': n_spikelets,
        'eficacia_pct': eficacia,
        'CC_pct': CC,
        'tipo': tipo,
        'latencia_pico_media_ms': lat_pico_media,
        'latencia_pico_sd_ms': np.std(lats_pico) if len(lats_pico) > 1 else 0.0,
        'latencia_onset_media_ms': lat_onset_media,
        'latencia_onset_sd_ms': np.std(lats_onset) if len(lats_onset) > 1 else 0.0,
        'amplitud_spikelet_media_mV': np.mean(amp_spikelets) if amp_spikelets else np.nan,
        'n_subumbrales': n_subumbrales,
        'amp_subumbral_media_mV': np.mean(amp_subumbrales) if amp_subumbrales else np.nan,
        'amplitud_pre_media_mV': amp_pre,
        'amplitud_post_media_mV': amp_post,
        'duracion_pre_media_ms': dur_pre,
        'duracion_post_media_ms': dur_post,
        'params_pre': params_pre,
        'params_post': params_post,
        'lats_pico': lats_pico,
        'lats_onset': lats_onset,
        't': t, 'V1': V1, 'V2': V2,
        'm1': m1, 'h1': h1, 'm2': m2, 'h2': h2,
    }

# ==============================================================================
# SIMULADOR UNIFICADO
# ==============================================================================

def simular(modelo, N_or_gj, I_func, t_max=300, dt=0.01):
    """Integra el sistema HH de dos neuronas con paso fijo dt=0.01 ms via odeint/LSODA.
    modelo='no_rect' (N_or_gj=gj en nS) o 'rect' (N_or_gj=N canales Cx36)."""
    t = np.arange(0, t_max, dt)
    OPTS = {'hmax': dt}  # paso fijo: hmax = dt = 0.01 ms
    y0 = [V_rest, m0, h0, n0, V_rest, m0, h0, n0]

    if modelo == 'no_rect':
        sol = odeint(sistema_no_rectificador, y0, t, args=(N_or_gj, Ej, I_func), **OPTS)
    else:
        sol = odeint(sistema_rectificador, y0, t, args=(N_or_gj, Ej, I_func), **OPTS)

    return t, sol

def simular_neurona_aislada(I_func, t_max=300, dt=0.01):
    """Simula neurona aislada (gj=0). Paso fijo dt=0.01 ms. Solo V1 es relevante."""
    return simular('no_rect', 0.0, I_func, t_max=t_max, dt=dt)

# ==============================================================================
# FUNCION AUXILIAR CSV
# ==============================================================================

def guardar_csv(df, nombre):
    """Guarda DataFrame como CSV con formato para Excel."""
    archivo = os.path.join(DIR_RESULTADOS, nombre)
    try:
        df.to_csv(archivo, index=False, sep=';', decimal=',',
                  encoding='utf-8-sig', float_format='%.6f')
        print(f"  [OK] Guardado: {archivo}")
    except PermissionError:
        base, ext = os.path.splitext(nombre)
        archivo_alt = os.path.join(DIR_RESULTADOS, f"{base}_nuevo{ext}")
        df.to_csv(archivo_alt, index=False, sep=';', decimal=',',
                  encoding='utf-8-sig', float_format='%.6f')
        print(f"  [AVISO] {nombre} bloqueado, guardado como: {archivo_alt}")
        archivo = archivo_alt
    return archivo

# ==============================================================================
# HELPERS PARA BLOQUES DE ANALISIS
# ==============================================================================

def _metricas_a_fila(m):
    """Convierte dict de metricas a fila plana para DataFrame."""
    return {
        'N': m['N'],
        'gj_nS': m['gj_nS'],
        'modelo': m['modelo'],
        'estimulo': m.get('estimulo', ''),
        'n_PA_pre': m['n_PA_pre'],
        'n_PA_post': m['n_PA_post'],
        'n_spikelets': m['n_spikelets'],
        'eficacia_pct': m['eficacia_pct'],
        'CC_pct': m['CC_pct'],
        'tipo': m['tipo'],
        'latencia_pico_media_ms': m['latencia_pico_media_ms'],
        'latencia_pico_sd_ms': m['latencia_pico_sd_ms'],
        'latencia_onset_media_ms': m['latencia_onset_media_ms'],
        'latencia_onset_sd_ms': m['latencia_onset_sd_ms'],
        'amplitud_spikelet_media_mV': m['amplitud_spikelet_media_mV'],
        'amplitud_pre_media_mV': m['amplitud_pre_media_mV'],
        'amplitud_post_media_mV': m['amplitud_post_media_mV'],
        'duracion_pre_media_ms': m['duracion_pre_media_ms'],
        'duracion_post_media_ms': m['duracion_post_media_ms'],
    }

def _guardar_temporal(metricas, rows_list, N, modelo, estimulo,
                      decimacion=10, I_func=None):
    """Agrega datos temporales decimados a la lista."""
    t = metricas['t']
    V1 = metricas['V1']
    V2 = metricas['V2']
    for j in range(0, len(t), decimacion):
        row = {
            'N': N, 'modelo': modelo, 'estimulo': estimulo,
            't_ms': t[j], 'V1_pre_mV': V1[j], 'V2_post_mV': V2[j]
        }
        if I_func is not None:
            row['I_stim_uA_cm2'] = I_func(t[j])
        rows_list.append(row)

def _guardar_AP_detalle(metricas, rows_list, N, modelo, estimulo):
    """Agrega detalle por AP a la lista."""
    for neurona, params in [('pre', metricas['params_pre']),
                            ('post', metricas['params_post'])]:
        for i, p in enumerate(params):
            row = {'N': N, 'modelo': modelo, 'estimulo': estimulo,
                   'neurona': neurona, 'AP_num': i+1}
            row.update(p)
            rows_list.append(row)

def _generar_csvs_barras(df):
    """Genera CSVs separados optimizados para graficas de barras."""
    cols_base = ['N', 'gj_nS', 'modelo', 'estimulo']

    # Latencia pico-a-pico
    guardar_csv(df[cols_base + ['latencia_pico_media_ms', 'latencia_pico_sd_ms']].dropna(
        subset=['latencia_pico_media_ms']), 'bloque_B_latencia_pico.csv')

    # Latencia onset-to-onset
    guardar_csv(df[cols_base + ['latencia_onset_media_ms', 'latencia_onset_sd_ms']].dropna(
        subset=['latencia_onset_media_ms']), 'bloque_B_latencia_onset.csv')

    # Amplitud
    guardar_csv(df[cols_base + ['amplitud_pre_media_mV', 'amplitud_post_media_mV']],
                'bloque_B_amplitud.csv')

    # CC%
    guardar_csv(df[cols_base + ['CC_pct', 'tipo']], 'bloque_B_CC.csv')

    # Duracion PA
    guardar_csv(df[cols_base + ['duracion_pre_media_ms', 'duracion_post_media_ms']],
                'bloque_B_duracion_PA.csv')

    # Eficacia
    guardar_csv(df[cols_base + ['n_PA_pre', 'n_PA_post', 'eficacia_pct']],
                'bloque_B_eficacia.csv')

    # Spikelets: conteo y amplitud
    guardar_csv(df[cols_base + ['n_spikelets', 'amplitud_spikelet_media_mV']],
                'bloque_B_spikelets.csv')


# ==============================================================================
# BLOQUE A1: DURACION MINIMA DE PULSO PARA 1 PA
# ==============================================================================

def bloque_A1_duracion_minima():
    """
    Busqueda binaria: duracion minima de pulso a AMP_SUPRAUMBRAL para generar 1 PA.
    CSV: bloque_A1_busqueda_duracion.csv, bloque_A1_temporal_umbral.csv
    """
    print("\n" + "="*80)
    print("BLOQUE A1: DURACION MINIMA DE PULSO PARA 1 PA")
    print("="*80)

    t_start = 50.0
    amp = AMP_SUPRAUMBRAL

    d_min = DURACION_MIN_BUSQUEDA
    d_max = DURACION_MAX_BUSQUEDA
    tol = TOLERANCIA_BUSQUEDA

    rows_busqueda = []

    while (d_max - d_min) > tol:
        d_mid = (d_min + d_max) / 2.0
        I_func = pulso_unico(amp, t_dur=d_mid, t_start=t_start)
        t_sim, sol = simular_neurona_aislada(I_func, t_max=t_start + 50)
        V1, m1, h1 = sol[:, 0], sol[:, 1], sol[:, 2]
        n_PA, _ = detectar_PAs_riguroso(V1, m1, h1, t_sim)

        rows_busqueda.append({
            'd_min_ms': d_min, 'd_max_ms': d_max,
            'd_test_ms': d_mid, 'n_PA': n_PA,
            'amplitud_uA_cm2': amp
        })

        print(f"  d={d_mid:.4f} ms -> {n_PA} PA")

        if n_PA >= 1:
            d_max = d_mid
        else:
            d_min = d_mid

    duracion_umbral = d_max  # menor valor testado que genera 1 PA
    print(f"\n  RESULTADO: Duracion minima = {duracion_umbral:.4f} ms (a {amp} uA/cm2)")

    guardar_csv(pd.DataFrame(rows_busqueda), 'bloque_A1_busqueda_duracion.csv')

    # Sweep lineal de duraciones para grafica en Excel
    print("\n  Generando curva duracion vs V_peak...")
    dur_sweep = np.arange(0.025, 1.025, 0.025)  # 0.025 a 1.000 ms, paso 0.025 ms
    rows_sweep = []
    for dur in dur_sweep:
        I_func_s = pulso_unico(amp, t_dur=dur, t_start=t_start)
        t_s, sol_s = simular_neurona_aislada(I_func_s, t_max=t_start + 50)
        V_s, m_s, h_s = sol_s[:, 0], sol_s[:, 1], sol_s[:, 2]
        n_s, _ = detectar_PAs_riguroso(V_s, m_s, h_s, t_s)
        V_peak_s = np.max(V_s[t_s >= t_start])
        rows_sweep.append({
            'duracion_ms': dur,
            'V_peak_mV': V_peak_s,
            'n_PA': n_s,
            'amplitud_uA_cm2': amp
        })
    guardar_csv(pd.DataFrame(rows_sweep), 'bloque_A1_curva_duracion.csv')
    print(f"  Curva: {len(rows_sweep)} puntos (0.025 a 1.000 ms, paso 0.025 ms)")

    # Traza temporal con duracion_umbral: muestra el PA al minimo estimulo (objeto de A1)
    I_func_umbral = pulso_unico(amp, t_dur=duracion_umbral, t_start=t_start)
    t_sim, sol = simular_neurona_aislada(I_func_umbral, t_max=t_start + 50)
    V1 = sol[:, 0]
    rows_temp = []
    for j in range(0, len(t_sim), 5):
        rows_temp.append({
            't_ms': t_sim[j],
            'V1_mV': V1[j],
            'I_stim_uA_cm2': I_func_umbral(t_sim[j])
        })
    guardar_csv(pd.DataFrame(rows_temp), 'bloque_A1_temporal_umbral.csv')

    # Traza subumbral: multiples pulsos independientes de duracion creciente
    # Todos a la misma amplitud, ninguno genera PA (d_min es el ultimo valor
    # subumbral confirmado por busqueda binaria, garantizado < duracion_umbral)
    duraciones_sub = [d_min * f for f in [0.20, 0.40, 0.60, 0.80, 0.90, 0.97, 1.00]]
    separacion_sub = 30.0  # ms entre pulsos (relajacion completa)

    def I_func_multi_sub(t):
        for k, dur in enumerate(duraciones_sub):
            t_ini = t_start + k * separacion_sub
            if t_ini <= t < t_ini + dur:
                return amp
        return 0.0

    t_max_sub = t_start + len(duraciones_sub) * separacion_sub + 20.0
    t_sim2, sol2 = simular_neurona_aislada(
        I_func_multi_sub, t_max=t_max_sub)
    V1_sub = sol2[:, 0]
    rows_temp_sub = []
    for j in range(0, len(t_sim2), 5):
        rows_temp_sub.append({
            't_ms': t_sim2[j],
            'V1_mV': V1_sub[j],
            'I_stim_uA_cm2': I_func_multi_sub(t_sim2[j])
        })
    guardar_csv(pd.DataFrame(rows_temp_sub), 'bloque_A1_temporal_subumbral.csv')
    print(f"  Subumbral: {len(duraciones_sub)} pulsos independientes "
          f"(duraciones: {[f'{d:.4f}' for d in duraciones_sub]} ms)")

    return duracion_umbral


# ==============================================================================
# BLOQUE A2: CURVA I-V (NEURONA AISLADA)
# ==============================================================================

def bloque_A2_curva_IV(duracion_pulso):
    """Curva I-V: barre I de 0-45 uA/cm2 en pulsos independientes de duracion_pulso.
    Identifica umbral de disparo y guarda bloque_A2_curva_IV.csv y bloque_A2_temporal_IV.csv."""
    print("\n" + "="*80)
    print("BLOQUE A2: CURVA I-V (NEURONA AISLADA)")
    print("="*80)

    t_start = 50.0
    amplitudes = np.arange(IV_AMP_MIN, IV_AMP_MAX + IV_AMP_PASO, IV_AMP_PASO)

    rows_IV = []
    primer_PA = False

    # Curva I-V: simulaciones independientes por amplitud
    for amp in amplitudes:
        I_func = pulso_unico(amp, t_dur=duracion_pulso, t_start=t_start)
        t_sim, sol = simular_neurona_aislada(I_func, t_max=t_start + 50)
        V1, m1, h1 = sol[:, 0], sol[:, 1], sol[:, 2]

        V_peak = np.max(V1[t_sim >= t_start])
        n_PA, _ = detectar_PAs_riguroso(V1, m1, h1, t_sim)

        marker = ''
        if n_PA > 0 and not primer_PA:
            marker = ' <-- PRIMER PA'
            primer_PA = True

        rows_IV.append({
            'I_stim_uA_cm2': amp,
            'V_peak_mV': V_peak,
            'duracion_pulso_ms': duracion_pulso,
            'n_PA': n_PA,
            'tipo': 'PA' if n_PA > 0 else 'subumbral'
        })

        print(f"  I={amp:6.1f} uA/cm2 -> V_peak={V_peak:8.2f} mV "
              f"[{'PA' if n_PA > 0 else 'subumbral'}]{marker}")

    guardar_csv(pd.DataFrame(rows_IV), 'bloque_A2_curva_IV.csv')

    # Temporal: simulacion unica con todos los pulsos secuenciales
    separacion_iv = 30.0  # ms entre pulsos
    amps_no_cero = [a for a in amplitudes if a > 0]

    def I_func_secuencial(t):
        for k, a in enumerate(amps_no_cero):
            t_ini = t_start + k * separacion_iv
            if t_ini <= t < t_ini + duracion_pulso:
                return a
        return 0.0

    t_max_seq = t_start + len(amps_no_cero) * separacion_iv + 20.0
    t_sim_seq, sol_seq = simular_neurona_aislada(
        I_func_secuencial, t_max=t_max_seq)
    V1_seq = sol_seq[:, 0]
    rows_temporal = []
    for j in range(0, len(t_sim_seq), 5):
        rows_temporal.append({
            't_ms': t_sim_seq[j],
            'V1_mV': V1_seq[j],
            'I_stim_uA_cm2': I_func_secuencial(t_sim_seq[j])
        })
    guardar_csv(pd.DataFrame(rows_temporal), 'bloque_A2_temporal_IV.csv')
    print(f"  Temporal: {len(amps_no_cero)} pulsos secuenciales "
          f"({amps_no_cero[0]}-{amps_no_cero[-1]} uA/cm2)")

    return pd.DataFrame(rows_IV)


# ==============================================================================
# BLOQUE A3: CARACTERIZACION COMPLETA DEL PA
# ==============================================================================

def bloque_A3_caracterizacion_PA(duracion_pulso, duracion_trabajo=None):
    """A3a: PA unico con duracion_trabajo; A3b: frecuencia natural DC; A3c: trenes 1-200 Hz.
    Genera CSVs bloque_A3a/b/c_* con parametros y trazas temporales."""
    if duracion_trabajo is None:
        duracion_trabajo = duracion_pulso
    print("\n" + "="*80)
    print("BLOQUE A3: CARACTERIZACION COMPLETA DEL PA")
    print("="*80)

    t_start = 50.0
    amp = AMP_SUPRAUMBRAL

    # ---- A3a: 1 PA con pulso de trabajo ----
    print("\n  --- A3a: 1 PA (pulso de trabajo) ---")
    I_func_1 = pulso_unico(amp, t_dur=duracion_trabajo, t_start=t_start)
    t_sim, sol = simular_neurona_aislada(I_func_1, t_max=t_start + 50)
    V1, m1, h1 = sol[:, 0], sol[:, 1], sol[:, 2]

    params_1PA = extraer_parametros_AP(V1, m1, h1, t_sim)
    if params_1PA:
        p = params_1PA[0]
        print(f"    Amplitud: {p['amplitud_mV']:.2f} mV")
        print(f"    Duracion: {p['duracion_ms']:.3f} ms")
        print(f"    V_pico: {p['V_pico_mV']:.2f} mV")
        print(f"    V_base: {p['V_base_mV']:.2f} mV")

    guardar_csv(pd.DataFrame(params_1PA), 'bloque_A3a_1PA_parametros.csv')

    rows_temp_1 = []
    for j in range(0, len(t_sim), 5):
        rows_temp_1.append({
            't_ms': t_sim[j], 'V1_mV': V1[j],
            'I_stim_uA_cm2': I_func_1(t_sim[j])
        })
    guardar_csv(pd.DataFrame(rows_temp_1), 'bloque_A3a_1PA_temporal.csv')

    # ---- A3b: Corriente continua larga -> frecuencia natural maxima ----
    print("\n  --- A3b: Corriente continua (frecuencia natural maxima) ---")
    I_func_dc = corriente_continua(amp, t_start=t_start, duracion=DURACION_DC_LARGA)
    t_max_dc = t_start + DURACION_DC_LARGA + 50
    t_sim_dc, sol_dc = simular_neurona_aislada(I_func_dc, t_max=t_max_dc)
    V1_dc = sol_dc[:, 0]
    m1_dc, h1_dc = sol_dc[:, 1], sol_dc[:, 2]

    params_dc = extraer_parametros_AP(V1_dc, m1_dc, h1_dc, t_sim_dc)
    freq_natural, intervalos, refractario = calcular_frecuencia_natural(
        V1_dc, m1_dc, h1_dc, t_sim_dc)

    print(f"    APs generados: {len(params_dc)}")
    print(f"    Frecuencia natural maxima: {freq_natural:.1f} Hz")
    if not np.isnan(refractario):
        print(f"    Periodo refractario aprox: {refractario:.2f} ms")
    print(f"    Intervalo medio: {np.mean(intervalos):.2f} ms" if intervalos else "")

    rows_dc_params = []
    for i, p in enumerate(params_dc):
        row = {'AP_num': i+1}
        row.update(p)
        row['intervalo_previo_ms'] = intervalos[i-1] if i > 0 else np.nan
        rows_dc_params.append(row)
    guardar_csv(pd.DataFrame(rows_dc_params), 'bloque_A3b_DC_parametros.csv')

    df_freq = pd.DataFrame([{
        'frecuencia_natural_Hz': freq_natural,
        'periodo_refractario_ms': refractario,
        'n_APs': len(params_dc),
        'duracion_DC_ms': DURACION_DC_LARGA,
        'amplitud_uA_cm2': amp,
        'intervalo_medio_ms': np.mean(intervalos) if intervalos else np.nan,
        'intervalo_sd_ms': np.std(intervalos) if len(intervalos) > 1 else 0.0
    }])
    guardar_csv(df_freq, 'bloque_A3b_frecuencia_natural.csv')

    rows_temp_dc = []
    for j in range(0, len(t_sim_dc), 10):
        rows_temp_dc.append({
            't_ms': t_sim_dc[j], 'V1_mV': V1_dc[j],
            'I_stim_uA_cm2': I_func_dc(t_sim_dc[j])
        })
    guardar_csv(pd.DataFrame(rows_temp_dc), 'bloque_A3b_DC_temporal.csv')

    # ---- A3c: Respuesta a distintas frecuencias de estimulacion ----
    # Ventana fija de 1 segundo. Para cada frecuencia f_estimulo se entregan
    # exactamente n_pulsos = f_estimulo estimulos. El modelo responde libremente.
    # Muestra la curva de transferencia: freq_respuesta vs freq_estimulo.
    print("\n  --- A3c: Respuesta a distintas frecuencias de estimulacion ---")
    print(f"    (ventana = 1 s | freq natural = {freq_natural:.1f} Hz)")

    FREQS_A3C = [1, 2, 5, 10, 20, 40, 60, 80, 100, 110, 120, 150, 200]
    T_VENTANA_A3C = 1000.0  # ms (1 segundo)

    rows_trenes = []
    rows_trenes_temp = []

    for freq in FREQS_A3C:
        n_pulsos = int(round(freq))       # pulsos en 1 s = freq numericamente
        periodo = 1000.0 / freq           # ms entre pulsos
        t_max_tren = t_start + T_VENTANA_A3C + 50  # 1100 ms total

        I_func_tren = tren_pulsos(freq, n_pulsos, amp, t_dur=duracion_trabajo,
                                  t_start=t_start)
        t_sim_t, sol_t = simular_neurona_aislada(I_func_tren, t_max=t_max_tren)
        V1_t, m1_t, h1_t = sol_t[:, 0], sol_t[:, 1], sol_t[:, 2]
        n_PA, _ = detectar_PAs_riguroso(V1_t, m1_t, h1_t, t_sim_t)

        efic = n_PA / n_pulsos * 100 if n_pulsos > 0 else 0
        freq_respuesta = n_PA / (T_VENTANA_A3C / 1000.0)  # APs por segundo
        nota = 'OK' if n_PA == n_pulsos else f'Faltan {n_pulsos - n_PA} APs'

        rows_trenes.append({
            'freq_estimulo_Hz': freq,
            'periodo_estimulo_ms': round(periodo, 4),
            'n_pulsos': n_pulsos,
            'duracion_pulso_ms': round(duracion_trabajo, 4),
            'n_PA_generados': n_PA,
            'freq_respuesta_Hz': round(freq_respuesta, 1),
            'eficiencia_pct': round(efic, 1),
            'nota': nota
        })

        print(f"    {freq:4.0f} Hz -> {n_pulsos:4d} pulsos, {n_PA:4d} PAs "
              f"(efic={efic:.0f}%, f_resp={freq_respuesta:.1f} Hz)"
              f"{' *** ' + nota if n_PA != n_pulsos else ''}")

        # Temporal decimado: paso ~0.1 ms para ver forma de AP
        for j in range(0, len(t_sim_t), 10):
            rows_trenes_temp.append({
                'freq_estimulo_Hz': freq,
                'n_pulsos': n_pulsos,
                't_ms': t_sim_t[j],
                'V1_mV': V1_t[j],
                'I_stim_uA_cm2': I_func_tren(t_sim_t[j])
            })

    guardar_csv(pd.DataFrame(rows_trenes), 'bloque_A3c_trenes_validacion.csv')
    guardar_csv(pd.DataFrame(rows_trenes_temp), 'bloque_A3c_trenes_temporal.csv')

    return freq_natural, refractario, duracion_pulso


# ==============================================================================
# BLOQUE B: NEURONAS ACOPLADAS - ANALISIS COMPLETO
# ==============================================================================

def bloque_B_neuronas_acopladas(duracion_pulso):
    """Simula cada N x modelo con pulso corto (1 PA); extrae CC, eficacia, latencias y amplitudes.
    Guarda bloque_B_resumen_completo.csv y CSVs individuales para graficas de barras."""
    print("\n" + "="*80)
    print("BLOQUE B: NEURONAS ACOPLADAS - ANALISIS COMPLETO")
    print("="*80)

    t_start = 50.0
    amp = AMP_SUPRAUMBRAL

    rows_resumen = []
    rows_temporal_corto = []
    rows_temporal_dc = []
    rows_AP_detalle = []

    for N in N_VALORES:
        gj = gj_from_N(N)

        for modelo_nombre in ['no_rect', 'rect']:
            # N=0 con rect es identico a no_rect con gj=0, saltar
            if modelo_nombre == 'rect' and N == 0:
                continue

            param = N if modelo_nombre == 'rect' else gj

            print(f"\n  N={N} (gj={gj:.3f} nS), modelo={modelo_nombre}")

            # ---- Pulso corto: 1 PA ----
            I_func_corto = pulso_unico(amp, t_dur=duracion_pulso, t_start=t_start)
            t_sim, sol = simular(modelo_nombre, param, I_func_corto, t_max=t_start + 50)

            metricas_corto = extraer_metricas_completas(t_sim, sol, N, modelo_nombre)
            fila = _metricas_a_fila(metricas_corto)
            fila['estimulo'] = 'pulso_corto'
            rows_resumen.append(fila)

            _guardar_temporal(metricas_corto, rows_temporal_corto,
                              N, modelo_nombre, 'pulso_corto', decimacion=10,
                              I_func=I_func_corto)
            _guardar_AP_detalle(metricas_corto, rows_AP_detalle,
                                N, modelo_nombre, 'pulso_corto')

            print(f"    Pulso corto: {metricas_corto['n_PA_pre']} PA pre, "
                  f"{metricas_corto['n_PA_post']} PA post, "
                  f"CC={metricas_corto['CC_pct']:.1f}%")

            # ---- Corriente continua: multiples PA ----
            # (comentado: analisis DC no utilizado en la TFG)
            # I_func_dc = corriente_continua(amp, t_start=t_start,
            #                                 duracion=DURACION_DC_LARGA)
            # t_max_dc = t_start + DURACION_DC_LARGA + 50
            # t_sim_dc, sol_dc = simular(modelo_nombre, param, I_func_dc,
            #                             t_max=t_max_dc)
            #
            # metricas_dc = extraer_metricas_completas(t_sim_dc, sol_dc, N, modelo_nombre)
            # fila_dc = _metricas_a_fila(metricas_dc)
            # fila_dc['estimulo'] = 'DC_continua'
            # rows_resumen.append(fila_dc)
            #
            # _guardar_temporal(metricas_dc, rows_temporal_dc,
            #                   N, modelo_nombre, 'DC', decimacion=1,
            #                   I_func=I_func_dc)
            # _guardar_AP_detalle(metricas_dc, rows_AP_detalle,
            #                     N, modelo_nombre, 'DC')
            #
            # print(f"    DC: {metricas_dc['n_PA_pre']} PA pre, "
            #       f"{metricas_dc['n_PA_post']} PA post, "
            #       f"CC={metricas_dc['CC_pct']:.1f}%")

    # Guardar resumen completo
    df_resumen = pd.DataFrame(rows_resumen)
    guardar_csv(df_resumen, 'bloque_B_resumen_completo.csv')

    # Generar CSVs de barras
    print("\n  Generando CSVs para graficas de barras...")
    _generar_csvs_barras(df_resumen)

    # Guardar temporales
    if rows_temporal_corto:
        guardar_csv(pd.DataFrame(rows_temporal_corto),
                    'bloque_B_temporales_pulso_corto.csv')
    if rows_temporal_dc:
        guardar_csv(pd.DataFrame(rows_temporal_dc),
                    'bloque_B_temporales_DC.csv')
    if rows_AP_detalle:
        guardar_csv(pd.DataFrame(rows_AP_detalle),
                    'bloque_B_parametros_AP_detalle.csv')

    return df_resumen


# ==============================================================================
# BLOQUE C: FILTRO PASO BAJO (SUPRAUMBRAL)
# ==============================================================================

def bloque_C_filtro_paso_bajo(duracion_pulso):
    """Evalua transmision a 7 frecuencias (1-100 Hz) para cada N y modelo.
    Calcula CC_nueva (eficacia x fidelidad amplitud) y guarda bloque_C_resumen.csv."""
    print("\n" + "="*80)
    print("BLOQUE C: FILTRO PASO BAJO")
    print("="*80)

    t_start = 50.0
    amp = AMP_SUPRAUMBRAL

    rows_resumen = []
    rows_temporal = []

    # Extiende N_VALORES con 31-37 para mayor resolucion en la zona de transicion
    # spikelet->PA (entre N=30 y N=38), sin afectar al resto de bloques.
    N_VALORES_C = sorted(set(N_VALORES) | set(range(31, 38)))

    for freq in FREQ_COMPLETO:
        n_pulsos = 10 if freq <= 1 else 20

        for N in N_VALORES_C:
            gj = gj_from_N(N)
            if N == 0:
                continue  # Sin acoplamiento no hay filtro

            for modelo_nombre in ['no_rect', 'rect']:
                param = N if modelo_nombre == 'rect' else gj

                I_func = tren_pulsos(freq, n_pulsos, amp,
                                     t_dur=duracion_pulso, t_start=t_start)
                periodo = 1000.0 / freq
                t_max = t_start + n_pulsos * periodo + 100

                t_sim, sol = simular(modelo_nombre, param, I_func, t_max=t_max)
                metricas = extraer_metricas_completas(t_sim, sol, N, modelo_nombre)

                # CC_nueva = (n_PA_post/n_PA_pre) x (ampl_post/ampl_pre)
                # Combina eficacia de transmision y fidelidad de amplitud
                n_pre  = metricas['n_PA_pre']
                n_post = metricas['n_PA_post']
                amp_pre  = metricas['amplitud_pre_media_mV']
                amp_post = metricas['amplitud_post_media_mV']
                if (n_pre > 0 and n_post > 0
                        and not np.isnan(amp_pre) and not np.isnan(amp_post)
                        and amp_pre > 0):
                    CC_nueva = (n_post / n_pre) * (amp_post / amp_pre) * 100.0
                else:
                    CC_nueva = 0.0

                # CC_spikelet = (n_spikelets/n_PA_pre) x (amp_spikelet/amp_pre)
                # Misma formula que CC_nueva pero sobre potenciales subumbrales transmitidos.
                # Mide cuanta senal llega al postsinaptico cuando el acoplamiento no es
                # suficiente para generar un PA completo.
                n_spk    = metricas['n_spikelets']
                amp_spk  = metricas['amplitud_spikelet_media_mV']
                if (n_pre > 0 and n_spk > 0
                        and not np.isnan(amp_spk) and not np.isnan(amp_pre)
                        and amp_pre > 0):
                    CC_spikelet = (n_spk / n_pre) * (amp_spk / amp_pre) * 100.0
                else:
                    CC_spikelet = 0.0

                # CC_subumbral = (n_subumbrales/n_PA_pre) x (amp_subumbral/amp_pre)
                # Como CC_spikelet pero con umbral de deteccion de 0.5 mV (vs 2 mV),
                # capturando spikelets pequenos en regimenes de acoplamiento debil.
                n_sub   = metricas['n_subumbrales']
                amp_sub = metricas['amp_subumbral_media_mV']
                if (n_pre > 0 and n_sub > 0
                        and not np.isnan(amp_sub) and not np.isnan(amp_pre)
                        and amp_pre > 0):
                    CC_subumbral = (n_sub / n_pre) * (amp_sub / amp_pre) * 100.0
                else:
                    CC_subumbral = 0.0

                rows_resumen.append({
                    'frecuencia_Hz': freq,
                    'N': N,
                    'gj_nS': gj,
                    'modelo': modelo_nombre,
                    'n_pulsos': n_pulsos,
                    'n_PA_pre': metricas['n_PA_pre'],
                    'n_PA_post': metricas['n_PA_post'],
                    'eficacia_pct': metricas['eficacia_pct'],
                    'CC_pct': metricas['CC_pct'],
                    'CC_nueva_pct': round(CC_nueva, 2),
                    'CC_spikelet_pct': round(CC_spikelet, 2),
                    'CC_subumbral_pct': round(CC_subumbral, 2),
                    'tipo': metricas['tipo'],
                    'latencia_pico_media_ms': metricas['latencia_pico_media_ms'],
                    'latencia_pico_sd_ms': metricas['latencia_pico_sd_ms'],
                    'n_spikelets': metricas['n_spikelets'],
                    'amplitud_spikelet_media_mV': metricas['amplitud_spikelet_media_mV'],
                    'amplitud_pre_media_mV': metricas['amplitud_pre_media_mV'],
                    'amplitud_post_media_mV': metricas['amplitud_post_media_mV'],
                    'latencia_onset_media_ms': metricas['latencia_onset_media_ms'],
                    'latencia_onset_sd_ms': metricas['latencia_onset_sd_ms'],
                })

                print(f"  {freq:4d} Hz, N={N:4d}, {modelo_nombre:7s}: "
                      f"CC_nueva={CC_nueva:6.1f}%, "
                      f"CC_spikelet={CC_spikelet:6.1f}%, "
                      f"CC_subumbral={CC_subumbral:6.1f}%, "
                      f"Efic={metricas['eficacia_pct']:5.0f}%, "
                      f"PA_pre={metricas['n_PA_pre']}, PA_post={metricas['n_PA_post']}, "
                      f"spikelets={metricas['n_spikelets']}, sub={metricas['n_subumbrales']}")

                # Decimacion adaptativa: ~100 muestras por periodo (evita artefacto en 55/70 Hz)
                decim_temporal = max(5, round(periodo / (100 * 0.01)))
                for j in range(0, len(t_sim), decim_temporal):
                    rows_temporal.append({
                        'frecuencia_Hz': freq, 'N': N,
                        'modelo': modelo_nombre,
                        't_ms': t_sim[j],
                        'V1_pre_mV': metricas['V1'][j],
                        'V2_post_mV': metricas['V2'][j],
                        'I_stim_uA_cm2': I_func(t_sim[j])
                    })

    guardar_csv(pd.DataFrame(rows_resumen), 'bloque_C_resumen.csv')
    if rows_temporal:
        guardar_csv(pd.DataFrame(rows_temporal), 'bloque_C_temporales.csv')

    return pd.DataFrame(rows_resumen)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# MODELO CONEXINA-36 - ANALISIS COMPLETO TFG (v4)")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*80)

    t_inicio = datetime.now()

    # Bloque A: Neurona individual
    duracion_umbral = bloque_A1_duracion_minima()

    # Duracion de trabajo: factor de seguridad x5 respecto al umbral minimo.
    # Garantiza disparo robusto en todos los analisis (aislada y acoplada).
    # Tipicamente resulta en ~1.12 ms, dentro del rango patch-clamp estandar.
    duracion_trabajo = max(duracion_umbral * 5, 1.0)  # ms — factor x5 sobre umbral
    print(f"\n  Duracion umbral (aislada):     {duracion_umbral:.4f} ms")
    print(f"  Duracion trabajo (estandar):   {duracion_trabajo:.4f} ms")
    print(f"  Factor de seguridad:           x{duracion_trabajo/duracion_umbral:.1f} respecto umbral aislada")

    bloque_A2_curva_IV(duracion_umbral)
    freq_natural, refractario, _ = bloque_A3_caracterizacion_PA(
        duracion_umbral, duracion_trabajo)

    # Bloque B: Neuronas acopladas (todos los N, ambos modelos)
    bloque_B_neuronas_acopladas(duracion_trabajo)

    # Bloque C: Filtro paso bajo
    bloque_C_filtro_paso_bajo(duracion_trabajo)

    t_fin = datetime.now()
    duracion = (t_fin - t_inicio).total_seconds()

    print("\n" + "#"*80)
    print(f"# COMPLETADO en {duracion:.1f} s")
    print(f"# CSVs en: {os.path.abspath(DIR_RESULTADOS)}")
    print("#"*80)
