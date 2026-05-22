# Cx36 Electrical Synapse Model with Weak Rectification

Computational model of two Hodgkin-Huxley neurons electrically coupled through Connexin-36 (Cx36) gap junctions. The model characterizes the biophysical properties of weakly rectifying electrical synapses, including action potential transmission, coupling efficiency, and low-pass filtering behavior.

*Bachelor's Thesis (TFG) — Alejandra Relaño, 2026*

---

## Overview

Electrical synapses formed by Cx36 channels are the predominant form of gap junctions in the mammalian brain. Unlike purely ohmic (non-rectifying) gap junctions, Cx36 channels exhibit **weak rectification**: their macroscopic conductance depends on the transjunctional voltage (Vj) following a cubic polynomial, making transmission asymmetric depending on the direction of current flow.

This model simulates a pair of neurons in three analysis blocks:

| Block | Description |
|-------|-------------|
| **A** | Isolated single neuron: minimum pulse duration, I-V curve, AP characterization |
| **B** | Coupled neurons: CC%, latency, amplitude, and transmission efficacy vs. number of Cx36 channels |
| **C** | Low-pass filter: frequency-dependent transmission for each coupling level |

All results are exported as CSV files (Excel-compatible) to the `../resultados/` directory.

---

## Biophysical Model

### Hodgkin-Huxley neurons

Each neuron follows the standard HH formulation with resting potential V_rest = −70 mV:

| Parameter | Value | Units |
|-----------|-------|-------|
| g_Na_max | 120.0 | mS/cm² |
| g_K_max | 36.0 | mS/cm² |
| g_L | 0.3 | mS/cm² |
| E_Na | +45.0 | mV |
| E_K | −82.0 | mV |
| E_L | −65.0 | mV |
| C_m | 1.0 | µF/cm² |

Gating variables (m, h, n) follow their standard alpha/beta rate functions with singularity handling at V = −45 mV (m gate) and V = −60 mV (n gate).

### Cx36 gap junction

The macroscopic junctional conductance is:

```
gj(Vj, N) = N × γ₀ × G(Vj) / 1000   [nS]
```

where:
- **N** — number of Cx36 channels
- **γ₀ = 15 pS** — unitary channel conductance
- **G(Vj)** — normalized rectification factor (cubic polynomial, G(0) = 1.0)

```python
G(Vj) = (a·Vj³ + b·Vj² + c·Vj + d) / d
# Coefficients: a=4.2e-8, b=1.48e-5, c=-1.51e-4, d=0.9984
```

Cx36 channels are always open (no gating); only conductance magnitude varies with Vj. A minimum G = 0.1 prevents complete channel closure.

Two coupling models are compared throughout:
- **`rect`** — rectifying Cx36 (G depends on Vj)
- **`no_rect`** — ideal ohmic junction (G = 1.0 constant)

---

## Code Structure

```
modelo_sinapsis_electrica_tipo_rectificador.py
│
├── Configuration constants
│   ├── Hodgkin-Huxley parameters
│   ├── Cx36 gap junction parameters
│   └── Analysis sweep settings
│
├── Gating functions
│   ├── alpha_m / beta_m  (Na⁺ activation)
│   ├── alpha_h / beta_h  (Na⁺ inactivation)
│   └── alpha_n / beta_n  (K⁺ activation)
│
├── ODE systems
│   ├── sistema_no_rectificador()  — ohmic gap junction
│   └── sistema_rectificador()     — Cx36 rectifying junction
│
├── Stimulus generators
│   ├── pulso_unico()        — single current pulse
│   ├── tren_pulsos()        — rectangular pulse train
│   └── corriente_continua() — DC step current
│
├── Detection functions
│   ├── detectar_PAs_riguroso()     — true action potentials (dV/dt + m³h threshold)
│   ├── detectar_spikelets()        — subthreshold peaks transmitted via gap junction
│   ├── medir_deflexion_subumbral() — any postsynaptic deflection after a presynaptic AP
│   └── detectar_inicio_PA()        — AP onset via dV/dt backtracking
│
├── Analysis functions
│   ├── calcular_CC()                     — coupling coefficient (%)
│   ├── calcular_latencia_pico()          — peak-to-peak latency (ms)
│   ├── calcular_latencia_onset()         — onset-to-onset latency (ms)
│   ├── calcular_frecuencia_natural()     — firing rate under DC stimulation
│   ├── extraer_parametros_AP()           — amplitude, duration, peak time per AP
│   └── extraer_metricas_completas()      — full metric set from a simulation run
│
├── Simulator
│   ├── simular()               — unified ODE integrator (LSODA, fixed step dt=0.01 ms)
│   └── simular_neurona_aislada() — convenience wrapper for gj = 0
│
└── Analysis blocks
    ├── bloque_A1_duracion_minima()      — binary search for rheobase pulse duration
    ├── bloque_A2_curva_IV()             — I-V curve sweep (0–45 µA/cm²)
    ├── bloque_A3_caracterizacion_PA()   — single AP, DC train, frequency sweep
    ├── bloque_B_neuronas_acopladas()    — coupled pair, all N values × both models
    └── bloque_C_filtro_paso_bajo()      — CC% vs frequency (1–100 Hz)
```

---

## Analysis Blocks

### Block A — Isolated neuron

**A1 — Minimum pulse duration (rheobase)**
Binary search finds the shortest pulse at I = 35 µA/cm² that reliably triggers one AP. Outputs a binary search log and a duration–V_peak curve (0.025–1.000 ms in 0.025 ms steps).

**A2 — I-V curve**
Sweeps stimulus amplitude from 0 to 45 µA/cm² in 1 µA/cm² steps using the rheobase pulse duration. Records peak voltage and AP count per stimulus level.

**A3 — AP characterization**
- **A3a**: Single AP properties (amplitude, duration above −20 mV, peak voltage, base voltage).
- **A3b**: Natural firing frequency and refractory period under sustained DC stimulation.
- **A3c**: Transfer function — AP count and firing rate vs. stimulus frequency (1–200 Hz, 1-second window).

### Block B — Coupled neurons

Simulates every combination of **N ∈ {0, 30, 38, 45, 70, 100, 1000}** channels and both coupling models (`rect` / `no_rect`) under a single suprathreshold pulse.

Metrics computed per condition:

| Metric | Description |
|--------|-------------|
| CC% | Coupling coefficient: ΔV_post / ΔV_pre × 100 |
| Efficacy (%) | n_PA_post / n_PA_pre × 100 |
| Latency (peak-to-peak) | Time from presynaptic AP peak to postsynaptic response peak |
| Latency (onset-to-onset) | Time from AP upstroke start to postsynaptic deflection start |
| Spikelets | Subthreshold postsynaptic peaks (transmitted but not regenerated) |
| AP amplitude & duration | Mean across all detected APs |
| Response type | `PA_transmitido` / `spikelet` / `sin_respuesta` / `sin_PA_pre` |

### Block C — Low-pass filter

Repeats Block B across 7 stimulus frequencies: **1, 8, 40, 55, 70, 80, 100 Hz** (10 pulses at ≤1 Hz, 20 pulses otherwise). Block C also extends the N sweep with intermediate values 31–37 to improve resolution in the spikelet→AP transition zone.

Three transmission metrics are reported:

| Metric | Formula |
|--------|---------|
| CC_nueva | (n_post/n_pre) × (amp_post/amp_pre) × 100 |
| CC_spikelet | (n_spikelets/n_pre) × (amp_spikelet/amp_pre) × 100 |
| CC_subumbral | (n_subumbral/n_pre) × (amp_subumbral/amp_pre) × 100 |

---

## Action Potential Detection

Detection uses a three-criterion filter to distinguish true APs from spikelets:

1. **Voltage threshold**: peak V > 0 mV
2. **dV/dt threshold**: max slope in ±0.2 ms window > 100 mV/ms
3. **Sodium channel activation**: m³h > 0.1 at the peak

Peaks that do not pass all three criteria are classified as **spikelets**. A third category, **subthreshold deflections**, uses a lower voltage threshold (V_rest + 1 mV) and no slope filter, capturing the weakest coupling-mediated responses.

---

## Outputs

All CSV files are written to `../resultados/` with semicolon separators and comma decimals for direct Excel import.

| File | Contents |
|------|----------|
| `bloque_A1_busqueda_duracion.csv` | Binary search iterations |
| `bloque_A1_curva_duracion.csv` | Duration–V_peak sweep |
| `bloque_A1_temporal_umbral.csv` | Voltage trace at threshold duration |
| `bloque_A1_temporal_subumbral.csv` | Subthreshold voltage traces |
| `bloque_A2_curva_IV.csv` | I-V curve data |
| `bloque_A2_temporal_IV.csv` | Sequential pulse voltage traces |
| `bloque_A3a_1PA_*.csv` | Single AP parameters and trace |
| `bloque_A3b_DC_*.csv` | DC stimulation parameters and trace |
| `bloque_A3c_trenes_*.csv` | Frequency sweep summary and traces |
| `bloque_B_resumen_completo.csv` | All metrics for all N × model conditions |
| `bloque_B_latencia_pico.csv` | Peak-to-peak latency (bar chart ready) |
| `bloque_B_latencia_onset.csv` | Onset-to-onset latency (bar chart ready) |
| `bloque_B_amplitud.csv` | Pre/post AP amplitude |
| `bloque_B_CC.csv` | Coupling coefficient |
| `bloque_B_duracion_PA.csv` | AP duration pre/post |
| `bloque_B_eficacia.csv` | Transmission efficacy |
| `bloque_B_spikelets.csv` | Spikelet count and amplitude |
| `bloque_B_temporales_pulso_corto.csv` | Decimated voltage traces (Block B) |
| `bloque_C_resumen.csv` | All metrics across frequencies and N values |
| `bloque_C_temporales.csv` | Decimated voltage traces (Block C) |

---

## Requirements

```
python >= 3.8
numpy
pandas
scipy
```

Install with:

```bash
pip install numpy pandas scipy
```

---

## Usage

```bash
python modelo_sinapsis_electrica_tipo_rectificador.py
```

The script runs Blocks A → B → C sequentially and prints progress to stdout. Total runtime is approximately 5–15 minutes depending on hardware (dominated by Block C: 7 frequencies × 14 N values × 2 models = 196 ODE integrations).

Results appear in `../resultados/` relative to the script location.

---

## Key Parameters to Modify

| Variable | Location | Effect |
|----------|----------|--------|
| `N_VALORES` | line 66 | Cx36 channel counts to simulate |
| `FREQ_COMPLETO` | line 63 | Frequencies for Block C |
| `AMP_SUPRAUMBRAL` | line 73 | Stimulus current amplitude (µA/cm²) |
| `gamma_0` | line 51 | Unitary Cx36 conductance (pS) |
| `COEF` | lines 55-60 | Rectification cubic polynomial coefficients |
| `E_L` | line 129 | Leak reversal potential (stability vs. excitability trade-off) |

---

## References

- Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117, 500–544.
- Srinivas, M. et al. (1999). Voltage dependence of macroscopic and unitary currents of gap junction channels formed by mouse connexin30. *Journal of Physiology*, 517, 673–689.
- Connors, B.W. & Long, M.A. (2004). Electrical synapses in the mammalian brain. *Annual Review of Neuroscience*, 27, 393–418.
