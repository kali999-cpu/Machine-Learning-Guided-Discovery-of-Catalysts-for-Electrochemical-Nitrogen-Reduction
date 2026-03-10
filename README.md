# Machine Learning Guided Discovery of Catalysts for Electrochemical Nitrogen Reduction (eNRR)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Samples](https://img.shields.io/badge/Samples-100-8B5CF6?style=for-the-badge)
![Features](https://img.shields.io/badge/Features-43-06B6D4?style=for-the-badge)
![Models](https://img.shields.io/badge/Models-6-F97316?style=for-the-badge)

<br/>

**A full computational science pipeline combining quantum physics, density functional theory (DFT), and machine learning to discover high-performance catalysts for sustainable ammonia production — replacing 6 hours of supercomputer time per material with 0.001 seconds of ML inference.**

<br/>

*MSc Computer Science — Data Mining & Machine Learning Project*

<br/>

[🌍 Why It Matters](#-why-this-matters) •
[🧪 Chemistry](#-the-chemistry) •
[⚛️ Quantum Mechanics](#️-quantum-mechanics--dft) •
[🤖 ML Pipeline](#-machine-learning--data-analytics---complete-cs-guide) •
[📊 DA & EDA](#4-exploratory-data-analysis-eda) •
[📈 Results](#-results) •
[⚙️ Installation](#️-installation) •
[🚀 Usage](#-usage) •
[📚 References](#-references)

</div>

---

## 🌍 Why This Matters

The **Haber-Bosch process**, invented in 1909, synthesises nearly **200 million tonnes of ammonia (NH₃)** every year. Ammonia is the foundation of synthetic fertilisers that feed roughly half of Earth's 8 billion people. Yet it operates under brutally harsh conditions and carries an enormous environmental price:

| Global Metric | Value |
|---|---|
| Energy consumption | ~2% of total world energy |
| CO₂ emissions | ~1.4% of global emissions (~450 Mt CO₂/yr) |
| Operating temperature | 400–500 °C |
| Operating pressure | 150–300 atm |
| Annual H₂ feedstock | ~55 Mt (mostly from fossil fuels) |
| Industry value | ~$150 billion/year |

```
Haber-Bosch:   N₂  +  3H₂   →   2NH₃    (400–500°C, 150–300 atm, iron catalyst)
eNRR:          N₂  +  6H⁺  +  6e⁻   →   2NH₃    (25°C, 1 atm, renewable electricity)
```

**Electrochemical nitrogen reduction (eNRR)** is the clean alternative. It runs at room temperature and atmospheric pressure using only water, nitrogen from the air, and renewable electricity. The bottleneck is finding a catalyst that is simultaneously **active**, **selective**, **stable**, and **scalable**.

The challenge: millions of possible materials exist. Experimentally testing each one takes weeks. DFT simulation of a single candidate takes 6–24 hours. Screening the full search space would take **decades** of compute time.

This project uses **machine learning trained on quantum-mechanical DFT data** to predict catalyst performance in **0.001 seconds per material** — a speedup of over **1,000,000×** — enabling the screening of unlimited candidates.

---

## 🧪 The Chemistry

### 1. The Nitrogen Problem

Molecular nitrogen (N₂) is extraordinarily chemically inert. Its triple bond is one of the strongest in all of chemistry:

```
N≡N     Bond dissociation energy  =  945 kJ mol⁻¹
        Bond length               =  1.098 Å
        HOMO–LUMO gap             =  10.8 eV
```

Activating this bond — the prerequisite for any nitrogen chemistry — requires either:
- **Extreme thermal energy** (Haber-Bosch: 400–500 °C, 300 atm), or
- An electrochemical catalyst that **donates electron density into the N₂ π\* antibonding orbital**, progressively weakening and eventually breaking the triple bond at room temperature.

### 2. The eNRR Reaction Mechanism

Two mechanistic pathways operate in eNRR:

**Dissociative mechanism** (requires very strong N binding — rare in electrochemistry):
```
N₂  →  2N*
N*  +  H⁺ + e⁻  →  NH*  →  NH₂*  →  NH₃ + *
```

**Associative mechanism** (dominant in eNRR — works at room temperature):
```
N₂  +  *   →  N₂*
N₂* +  H⁺ + e⁻  →  *NNH
*NNH + H⁺ + e⁻  →  *NNH₂   or   *NHNH
...  continuing 6 proton-electron transfer steps  ...
→  NH₃  +  NH₃ + *
```

The full free energy pathway includes 8 distinct intermediates, each with a Gibbs free energy change (ΔG). The **largest uphill ΔG step** is the **potential-determining step** — the kinetic bottleneck that sets the minimum voltage needed to drive the reaction:

```
N₂  →  *N₂  →  *NNH  →  *NNH₂  →  *N + NH₃  →  *NH  →  *NH₂  →  NH₃

Free energy profile for Mo (excellent catalyst):
  +0.00 → +0.26 → −0.15 → −0.35 → −0.62 → −0.85 → −1.12 → −1.60 eV

Free energy profile for Cu (terrible catalyst):
  +0.00 → +1.65 → +0.90 → ...    ← First step alone costs 1.65 eV
```

### 3. The Sabatier Principle and the Volcano Plot

The **Sabatier principle** (Paul Sabatier, Nobel Prize 1912) is the central organising idea of heterogeneous catalysis: the ideal catalyst binds its substrate *not too strongly and not too weakly* — the so-called Goldilocks condition.

For eNRR, this translates to an **optimal nitrogen adsorption free energy window**:

```
ΔG_N* = −0.50  to  −0.20 eV    →   ⭐ OPTIMAL (Sabatier window)

ΔG_N* << −0.50 eV              →   ❌ Too strong: N* poisons the surface.
                                     N≡N is broken but NH₃ is never released.
                                     (Mn at −1.20 eV, W at −0.85 eV)

ΔG_N* >> −0.20 eV              →   ❌ Too weak: N₂ does not adsorb at all.
                                     HER dominates — only H₂ is produced.
                                     (Cu at +0.67 eV)
```

Plotting catalyst activity (limiting potential) against ΔG_N\* gives the **volcano curve** — the key visualisation of this project:

```
   Activity
      ↑         Ru(−0.31)  Mo(−0.28)
      │             /‾‾‾‾\
      │            /       \
      │     Fe    /         \    Ni
      │    (−0.44)/           \(−0.15)
      │          /             \
      │    W    /               \    Cu
      │ (−0.85)                  \ (+0.67)
  ────┼─────────────────────────────────→  ΔG_N* (eV)
     −1.2   −0.8   −0.5  −0.35  −0.2   0   +0.7
                    ↑──── Optimal ────↑
```

### 4. The Limiting Potential and Overpotential

The **limiting potential** (U_L) is the most negative electrode potential at which every electrochemical step becomes thermodynamically downhill. It sets the required **overpotential**:

```
η = U_L − U_equilibrium       where U_eq = −0.16 V vs. RHE for N₂ → 2NH₃

Approximate volcano model:
U_L ≈ −( |ΔG_N* + 0.35| × 0.65 + 0.09 )   [V vs. RHE]

Examples:
  Mo (ΔG_N* = −0.28 eV):  U_L ≈ −0.44 V   ← Near-optimal
  W  (ΔG_N* = −0.85 eV):  U_L ≈ −0.91 V   ← Large overpotential
  Cu (ΔG_N* = +0.67 eV):  U_L ≈ −1.42 V   ← Terrible
```

A less negative U_L = lower applied voltage = lower energy cost per mole of NH₃.

### 5. The HER Competition Problem

The biggest practical challenge of eNRR is selectivity. The **hydrogen evolution reaction (HER)** competes directly:

```
eNRR:  N₂  +  6H⁺  +  6e⁻  →  2NH₃    E° = −0.16 V vs. RHE
HER:   2H⁺  +  2e⁻  →  H₂            E° =  0.00 V vs. RHE
```

At most metallic surfaces, HER kinetics are 2–4 orders of magnitude faster than NRR. The **Faradaic Efficiency (FE%)** — the fraction of electrons flowing that actually produce NH₃ — is therefore the critical experimental figure of merit:

```
FE (%) = (moles NH₃ × 3 × F) / (total charge passed) × 100

Typical state-of-the-art FE values:
  Mo bulk:         8.1%     (best single-metal)
  Ru bulk:         7.5%
  Fe bulk:         4.5%
  SAC on N-C:     15–35%   (single-atom advantage: no adjacent H sites)
  This project peak: ~30%  (predicted for optimal ΔG_N* candidates)
```

The empirical FE model used in this pipeline:

```python
FE(ΔG_N*) = 0.30 × exp(−0.5 × ((ΔG_N* + 0.35) / 0.18)²)
# Gaussian centred at optimal −0.35 eV, peak FE = 30%, σ = 0.18 eV
```

### 6. Single-Atom Catalysts (SAC) — Why They Are Special

The most promising emerging catalyst class is **single-atom catalysts (SAC)**: isolated transition metal atoms anchored on a 2D support substrate (N-doped graphene, MXene, MoS₂, hexagonal BN). Their advantages:

| Property | SAC Advantage | Physical Reason |
|---|---|---|
| Atom utilisation | 100% | Every atom is an active site |
| Selectivity | High FE (15–35%) | N-coordination blocks adjacent H* adsorption |
| Tunability | Support-dependent | N-ligands shift d-band via charge transfer |
| Cost | Earth-abundant metals | Fe, Co, Ni, Mn, Cr — not Rh, Ir, Pt |
| ΔG_N\* control | ±0.5 eV vs. bulk | Support choice moves the binding energy |

The support is not chemically innocent — it modulates the metal's d-band centre through ligand field effects and charge transfer, shifting ΔG_N\* by 0.1–0.5 eV relative to the same metal in bulk form.

### 7. Catalyst Material Classes in This Project

| Class | Count | Examples | ΔG_N\* Range |
|---|---|---|---|
| Bulk transition metals | 20 | Fe, Mo, Ru, Co, W, V, Mn | −1.20 to +0.67 eV |
| Bimetallic alloys | 38 | FeMo, RuCoMo, FeMo2Ru | −0.85 to −0.22 eV |
| SAC / graphene & N-C | 12 | Mo-SAC/N-C, Fe-SAC/gr | −0.38 to −0.14 eV |
| SAC / MXene | 9 | Mo-SAC/Ti3C2, Fe-SAC/V4C3 | −0.35 to −0.19 eV |
| SAC / MoS₂ & BN | 6 | Co-SAC/MoS2, Rh-SAC/BN | −0.37 to −0.28 eV |
| Nitrides | 6 | MoN, Fe₂N, VN, W₂N | −0.58 to −0.35 eV |
| Carbides | 6 | Mo₂C, WC, VC, TiC | −0.80 to −0.54 eV |
| Phosphides | 6 | MoP, FeP, CoP, Ni₂P | −0.62 to −0.30 eV |
| **Total** | **103** | — | — |

---

## ⚛️ Quantum Mechanics & DFT

### 1. The Electronic Schrödinger Equation

Every material property ultimately originates from solving the **time-independent Schrödinger equation**:

```
Ĥ Ψ = E Ψ

where:
  Ĥ  = Hamiltonian operator (total energy of the quantum system)
  Ψ  = many-body wavefunction (describes ALL electrons simultaneously)
  E  = ground-state energy (what we want to compute)
```

The full Hamiltonian for a system of N electrons and M nuclei expands as:

```
Ĥ = T̂_e + T̂_n + V̂_ee + V̂_en + V̂_nn

  T̂_e   = −(ℏ²/2mₑ) Σᵢ ∇ᵢ²          electron kinetic energy
  T̂_n   ≈  0                           nuclear KE (Born-Oppenheimer approximation)
  V̂_ee  = Σᵢ<ⱼ e²/|rᵢ−rⱼ|            electron-electron repulsion
  V̂_en  = −Σᵢ,α Zα e²/|rᵢ−Rα|        electron-nuclear attraction
  V̂_nn  = Σα<β ZαZβ e²/|Rα−Rβ|        nuclear-nuclear repulsion
```

For a realistic catalyst surface slab of 200 atoms (~1,600 electrons), the wavefunction Ψ(r₁, r₂, ..., r₁₆₀₀) lives in a **4,800-dimensional space** — completely impossible to solve exactly. This is where DFT steps in.

### 2. Density Functional Theory (DFT)

**DFT** (Hohenberg-Kohn 1964; Kohn-Sham 1965) is the computational workhorse of modern materials science. The revolutionary insight: **the electron density ρ(r)** — a function of just 3 spatial coordinates — contains *all* the information of the full many-body wavefunction.

#### The Two Hohenberg-Kohn Theorems

**HK Theorem 1 (Existence):** The external potential V_ext(r) — and therefore all ground-state properties — is a unique functional of the electron density ρ(r). If you know ρ(r), you know everything about the system.

**HK Theorem 2 (Variational principle):** The exact ground-state energy is the global minimum of the energy functional E[ρ]. For any trial density ρ̃: E[ρ₀] ≤ E[ρ̃].

#### Kohn-Sham Equations — The Practical Implementation

Kohn and Sham reformulated the many-body problem as a set of single-particle equations solved self-consistently:

```
[ −(ℏ²/2mₑ) ∇² + V_eff(r) ] φᵢ(r) = εᵢ φᵢ(r)    (Kohn-Sham orbitals)

Effective potential:
  V_eff(r) = V_ext(r) + V_Hartree(r) + V_xc(r)

           = V_ext(r) + ∫ ρ(r')/(|r−r'|) dr' + δE_xc/δρ(r)
             ─────────   ──────────────────────   ───────────────
            nuclear pot.  classical Coulomb        exchange-correlation
                          (Hartree term)           (all quantum effects)

Electron density reconstructed from orbitals:
  ρ(r) = Σᵢ |φᵢ(r)|²

Self-consistency loop:
  ρ_in → V_eff → {φᵢ} → ρ_out → (if ρ_out ≠ ρ_in) → update → repeat
```

The **exchange-correlation functional** V_xc(r) contains all the many-body quantum effects that cannot be computed exactly. Common approximations used in catalysis:

| Functional | Level | Typical Error | Used For |
|---|---|---|---|
| LDA | Local Density | ~0.5 eV | Quick estimates |
| GGA-PBE | Semi-local | ~0.2 eV | Standard surfaces |
| DFT+U | GGA + Hubbard U | ~0.1 eV | Fe, Co, Ni (strong correlation) |
| HSE06 | Hybrid | ~0.05 eV | Band gaps, semiconductors |

#### Computational Cost Comparison

| Method | Physical Accuracy | Formal Scaling | Time (200-atom surface) |
|---|---|---|---|
| Full CI | Exact | O(N!) | ~Centuries |
| CCSD(T) | ~meV | O(N⁷) | ~Decades |
| DFT (GGA-PBE) | ~0.1–0.2 eV | O(N³) | **6–24 hours** |
| **ML (this project)** | **~0.08 eV** | **O(1)** | **0.001 seconds** |

The ML model achieves accuracy approaching DFT chemical accuracy while providing a **speedup of 10⁶–10⁷×**.

### 3. The Hammer-Nørskov d-Band Model

The most powerful theoretical insight connecting quantum electronic structure to catalytic reactivity is the **d-band model** (Hammer and Nørskov, 1995). Its central prediction:

> *The reactivity of a transition metal surface is primarily controlled by the mean energy of its d-band (ε_d, the d-band centre) relative to the Fermi level.*

The physical basis comes from the Newns-Anderson chemisorption Hamiltonian:

```
Interaction energy of adsorbate orbital (energy εₐ) with the metal surface:

E_ads ≈  [sp-band contribution, ~constant]
       + [d-band contribution, strongly variable]

d-band contribution ∝ − V²_ad / (εₐ − εd)  ×  f(d-band filling)

where:
  V_ad   = coupling matrix element (adsorbate-d orbital hybridisation)
  εd     = d-band centre (the Hammer-Nørskov descriptor)
  f(...)  = Pauli repulsion correction from d-band filling
```

**Key predictions validated in this dataset:**
- A higher (less negative) d-band centre → stronger adsorbate binding → more negative ΔG_N\*
- Shifting ε_d upward by alloying → tunes ΔG_N\* predictably
- The correlation d_band_center ↔ ΔG_N\* has r = −0.52 in our data — strong but imperfect

The imperfect correlation (r² = 0.27, not 1.0) is a critical scientific finding: **d-band centre alone explains only 27% of variance in ΔG_N\***. This quantitatively justifies using a 43-feature ML model rather than the single-descriptor d-band model — the core scientific motivation for this entire project.

### 4. Computing ΔG_N* from DFT

The target variable ΔG_N\* is computed in DFT as the Gibbs free energy of nitrogen adsorption:

```
ΔG_N* = E_DFT(slab+N*) − E_DFT(slab) − ½ E_DFT(N₂) + ΔZPE − TΔS

where:
  E_DFT(slab+N*)  = DFT energy: surface with one N atom adsorbed
  E_DFT(slab)     = DFT energy: clean catalyst surface
  ½ E_DFT(N₂)    = DFT energy of gas-phase N₂, per N atom
  ΔZPE            = zero-point energy correction   ≈ +0.03 eV
  TΔS             = entropic correction at 300 K   ≈ −0.02 eV

Net thermal correction ≈ +0.01 eV (small compared to variation in ΔG_N*)
```

This single number encapsulates the thermodynamic driving force for the first and rate-limiting step of eNRR. Our ML model predicts it **from atomic features alone**, bypassing the DFT slab calculation entirely.

---

## 🤖 Machine Learning & Data Analytics — Complete CS Guide

> This is the core Computer Science and Data Science contribution. The following sections document the complete ML and DA pipeline from problem formulation through deployment — written for an MSc Computer Science audience.

---

### 1. ML Task Definition & Problem Formulation

Before writing a single line of code, the ML problem must be formally defined:

| ML Property | Definition | In This Project |
|---|---|---|
| Learning paradigm | Supervised Learning | Every sample has a known DFT label (ΔG_N\*) |
| Task type | **Regression** | Target is a continuous real number in eV |
| Input X | Feature matrix | 100 materials × 43 atomic/electronic descriptors |
| Output y | Target vector | ΔG_N\* — nitrogen adsorption free energy (eV) |
| Goal | Generalisation | Predict ΔG_N\* for **new, unseen** catalyst materials |

#### Why Supervised and Not Unsupervised?

This is **100% supervised learning** because every training sample has a known correct output (ΔG_N\* from DFT literature). Unsupervised methods (K-Means, PCA, hierarchical clustering) were used only as supplementary EDA tools — to group similar metals and visualise the feature space — not as the primary prediction engine.

#### Why Regression and Not Classification?

The target variable ΔG_N\* is a **continuous real number** ranging from −1.20 to +0.67 eV. Regression predicts any real number. Classification would only output discrete labels (e.g. "Optimal / Too Strong / Too Weak"), which would lose the quantitative ranking needed for the volcano plot and production time calculations.

#### Is Logistic Regression Used? — Exact Explanation

**No — and here is the precise reason:**

Logistic Regression applies a sigmoid function σ(z) = 1/(1+e⁻ᶻ) to output **class probabilities** — it is fundamentally a **classification algorithm**. Since this project predicts a continuous eV value, it is not applicable.

| Property | Linear Regression ✅ (USED) | Logistic Regression ❌ (NOT used) |
|---|---|---|
| Task | Regression → continuous output | Classification → discrete class |
| Output | Real number (e.g. −0.28 eV) | Probability 0–1, thresholded to class |
| Output function | ŷ = wᵀx + b | P(y=1) = σ(wᵀx + b) |
| Loss function | MSE | Binary Cross-Entropy |
| Evaluation | R², MAE, RMSE | Accuracy, Precision, Recall, F1, ROC-AUC |
| When to use | Predicting energy in eV ✅ | Binary "Good / Bad catalyst?" |

*If the task were reformulated as "Is ΔG_N\* between −0.50 and −0.20 eV? (Yes/No)", that becomes binary classification and Logistic Regression would be the correct tool.*

---

### 2. Data Sources — Where the Data Comes From

The dataset was assembled from four complementary sources:

| # | Source | Type | Access Method | What It Provides |
|---|---|---|---|---|
| 1 | `pymatgen` Python library | Atomic & elemental properties | `from pymatgen.core import Element` | Electronegativity, atomic radius, ionisation energy, d-electrons, melting point |
| 2 | `mendeleev` Python library | Additional element properties | `import mendeleev` | Electron affinity, bulk modulus, van der Waals radius, lattice constants |
| 3 | Materials Project REST API | Electronic structure + crystal data | HTTP GET → JSON → DataFrame | Formation energies, band gaps, magnetic moments, crystal structures |
| 4 | DFT scientific literature | Adsorption energies — training labels | Manual extraction from papers | ΔG_N\*, limiting potential U_L, Faradaic Efficiency |

#### Key Literature Sources (Training Labels)

| Paper | Year | Data Obtained |
|---|---|---|
| Skúlason et al., *Phys. Chem. Chem. Phys.* | 2012 | N adsorption energies on 14 transition metals |
| Vojvodic et al., *Chem. Phys. Lett.* | 2014 | NRR energetics on alloy surfaces |
| Andersen et al., *Nature* | 2019 | Rigorous electrochemical benchmarks |
| Wang et al., *ACS Catalysis* | 2023 | SAC on MXene and N-doped carbon |
| Liu et al., *Nat. Commun.* | 2022 | Single-atom catalyst DFT dataset |
| Qing et al., *Chem. Rev.* | 2020 | Faradaic Efficiency experimental values |

#### Data Collection Code — Source 1 (pymatgen)

```python
from pymatgen.core import Element
import pandas as pd

metals = ["Fe","Mo","Ru","Co","Ni","W","V","Mn","Cr","Cu","Rh","Re","Os","Ir"]
rows = []
for sym in metals:
    el = Element(sym)
    rows.append({
        "element":           sym,
        "electronegativity": el.X,
        "atomic_radius":     float(el.atomic_radius),
        "ionization_energy": el.ionization_energies[0],
        "electron_affinity": el.electron_affinity,
        "d_electrons":       el.get_electronic_structure_dict().get("d", 0),
        "group":             el.group,
        "period":            el.row,
        "melting_point":     el.melting_point,
    })
df_atomic = pd.DataFrame(rows)
df_atomic.to_csv("data/atomic_properties.csv", index=False)
```

#### Data Collection Code — Source 3 (Materials Project REST API)

```python
from mp_api.client import MPRester

# REST API: HTTP GET request → JSON response → pandas DataFrame
with MPRester("YOUR_FREE_API_KEY") as mpr:
    docs = mpr.materials.summary.search(
        elements=["Mo", "N"],
        is_stable=True,
        fields=["material_id","formula_pretty",
                "formation_energy_per_atom","band_gap","total_magnetization"]
    )
df_mp = pd.DataFrame([d.dict() for d in docs])
# This is a standard programmatic REST API call — authenticate → GET → parse JSON → DataFrame
```

---

### 3. Dataset Structure & Feature Table

**Final dataset (v3): 100 samples × 43 features + 1 target**

#### The 43 Feature Descriptors — 8 Physics-Motivated Groups

| # | Feature | Group | Physical Meaning | Importance |
|---|---|---|---|---|
| 1 | `atomic_number` | Atomic identity | Nuclear charge Z; determines electron count | Medium |
| 2 | `atomic_mass` | Atomic identity | Relativistic effects proxy | Low |
| 3 | `electronegativity` | Electrochemical | Electron-attracting power (Pauling scale) | High |
| 4 | `ionization_energy` | Electrochemical | Energy to remove outermost electron | Medium |
| 5 | `second_ionization_energy` | Electrochemical | Core electron accessibility | Medium |
| 6 | `electron_affinity` | Electrochemical | Energy gained by accepting an electron | High |
| 7 | `atomic_radius` | Size/geometry | Orbital overlap extent with N adsorbate | Medium |
| 8 | `covalent_radius` | Size/geometry | Half covalent bond length | Medium |
| 9 | `ionic_radius` | Size/geometry | Radius in ionic bonding state | Low |
| 10 | `van_der_waals_radius` | Size/geometry | Non-bonded contact distance | Low |
| 11 | `coordination_number` | Size/geometry | Nearest-neighbour atom count | Medium |
| 12 | `lattice_constant` | Size/geometry | Unit cell parameter | Medium |
| 13 | `valence_electrons` | Electronic config | Total electrons available for bonding | Medium |
| 14 | `d_electrons` | Electronic config | d-orbital occupancy — core d-band descriptor | **High** |
| 15 | `s_electrons` | Electronic config | s-orbital occupancy | Low |
| 16 | `p_electrons` | Electronic config | p-orbital occupancy | Low |
| 17 | `f_electrons` | Electronic config | f-orbital occupancy (rare earth/actinide) | Low |
| 18 | `group` | Electronic config | Periodic table column | Medium |
| 19 | `period` | Electronic config | Periodic table row (3d, 4d, 5d metals) | Low |
| 20 | `melting_point` | Thermodynamic/bulk | Thermal stability; strong bond → high melt | **High** |
| 21 | `boiling_point` | Thermodynamic/bulk | Vapour pressure at operating temperature | Medium |
| 22 | `density` | Thermodynamic/bulk | Crystal packing efficiency | Medium |
| 23 | `heat_of_fusion` | Thermodynamic/bulk | Solid→liquid transition enthalpy | Medium |
| 24 | `heat_of_vaporization` | Thermodynamic/bulk | Cohesion energy proxy | Medium |
| 25 | `specific_heat` | Thermodynamic/bulk | Thermal mass | Low |
| 26 | `cohesive_energy` | Thermodynamic/bulk | Total binding energy per atom | High |
| 27 | `formation_energy` | Thermodynamic/bulk | Compound thermodynamic stability | Medium |
| 28 | `nitride_formation_energy` | Chemical affinity | Drive to form bulk nitride — N-binding proxy | **#1** |
| 29 | `oxide_formation_energy` | Chemical affinity | Oxidation resistance under operation | Medium |
| 30 | `surface_energy` | Surface/mechanical | Energy cost of creating a surface | Medium |
| 31 | `work_function` | Surface/mechanical | Fermi level proxy; electron emission barrier | **High** |
| 32 | `bulk_modulus` | Surface/mechanical | Resistance to uniform compression | Medium |
| 33 | `shear_modulus` | Surface/mechanical | Resistance to shear | Medium |
| 34 | `youngs_modulus` | Surface/mechanical | Tensile stiffness | Medium |
| 35 | `poisson_ratio` | Surface/mechanical | Lateral strain ratio | Low |
| 36 | `thermal_conductivity` | Surface/mechanical | Heat dissipation (Joule heating) | Medium |
| 37 | `electrical_conductivity` | Surface/mechanical | Electron transport to active site | Medium |
| 38 | `magnetic_moment` | Electronic structure | Net electron spin (Fe, Co, Ni) | Low |
| 39 | `d_band_center` | Electronic structure | ε_d — Hammer-Nørskov descriptor | **#2** |
| 40 | `d_band_width` | Electronic structure | d-orbital delocalisation width | High |
| 41 | `d_band_filling` | Electronic structure | Fraction of d-band occupied | High |
| 42 | `fermi_energy` | Electronic structure | Electron chemical potential | Medium |
| 43 | `band_gap` | Electronic structure | Conductor vs. semiconductor | Low |

> **Why 43?** This is the minimal complete set capturing: atomic identity, bulk thermodynamics, surface reactivity, electronic structure, and mechanical stability — the five physical pillars of catalyst performance. No single feature explains >27% of ΔG_N\* variance — all 43 are needed.

#### ML Terminology Applied to This Dataset

| ML Term | Also Called | In This Project | Example |
|---|---|---|---|
| Features (X) | Predictors, inputs | 43 atomic/electronic properties | `electronegativity=2.16`, `d_band_center=−1.30` |
| Target (y) | Label, output | ΔG_N\* in eV | Mo → ΔG_N\* = −0.28 eV |
| Sample | Instance, row | One material | Row: Mo, 2.16, −1.30, ..., −0.28 |
| Feature matrix | Design matrix | X.shape = (100, 43) | 100 materials × 43 features |
| Target vector | Label array | y.shape = (100,) | [−0.44, −0.28, ..., +0.67] |

#### Data Types of the Features

| Feature | ML Data Type | Why It Matters for Preprocessing |
|---|---|---|
| `electronegativity` | Continuous / Ratio | Can be scaled; linear operations valid |
| `d_band_center` | Continuous / Ratio | Negative values fine — RobustScaler handles |
| `d_electrons` | Discrete / Integer | Count of electrons (integers 0–10) |
| `group`, `period` | Ordinal / Integer | Has natural order — no one-hot encoding needed |
| ΔG_N\* (TARGET) | Continuous / Ratio | Regression target — real number |

#### Reference: All 14 Bulk Transition Metals (Benchmark)

| Metal | χ | d-band (eV) | Work Fn (eV) | ΔG_N\* (eV) | U_L (V) | FE% | Quality |
|---|---|---|---|---|---|---|---|
| **Mo** | 2.16 | −1.30 | 4.36 | −0.28 | −0.44 | 8.1% | ⭐ BEST |
| **Ru** | 2.20 | −1.41 | 4.71 | −0.31 | −0.48 | 7.5% | ⭐ Excellent |
| **Fe** | 1.83 | −1.29 | 4.67 | −0.44 | −0.52 | 4.5% | ✅ Near Optimal |
| **Rh** | 2.28 | −1.73 | 4.98 | −0.40 | −0.55 | 5.6% | ✅ Near Optimal |
| Ir | 2.20 | −2.11 | 5.27 | −0.50 | −0.65 | 4.1% | Borderline |
| Co | 1.88 | −1.17 | 5.00 | −0.52 | −0.61 | 3.2% | Borderline |
| Ni | 1.91 | −1.29 | 5.15 | −0.15 | −0.72 | 1.8% | Too Weak |
| Cr | 1.66 | −1.49 | 4.50 | −0.60 | −0.82 | 2.9% | Too Strong |
| V | 1.63 | −1.09 | 4.30 | −0.68 | −0.78 | 3.5% | Too Strong |
| W | 2.36 | −1.78 | 4.55 | −0.85 | −0.91 | N/A | Too Strong |
| Os | 2.20 | −1.78 | 5.20 | −0.72 | −0.88 | N/A | Too Strong |
| Re | 1.90 | −1.83 | 4.96 | −0.95 | −1.05 | N/A | Too Strong |
| Mn | 1.55 | −1.70 | 4.10 | −1.20 | −1.20 | N/A | Way Too Strong |
| Cu | 1.90 | −2.67 | 4.65 | **+0.67** | −1.42 | 0.5% | ❌ WORST |

χ = Pauling electronegativity | U_L = limiting potential | FE = Faradaic Efficiency

---

### 4. Exploratory Data Analysis (EDA)

EDA is performed **before any modelling** to understand distributions, detect outliers, identify correlations, and decide preprocessing steps.

#### 4.1 Statistical Summary — Always First

```python
import pandas as pd

df = pd.read_csv("data/master_dataset.csv")

print(df.shape)                               # (100, 44) — 100 samples, 43 features + target
print(df.describe())                          # count, mean, std, min, 25%, 50%, 75%, max
print(df.isnull().sum())                      # missing values per column
print(df.dtypes)                              # data type of each column
print(df[["d_band_center","dG_N"]].corr())   # quick Pearson correlation check
```

**Target ΔG_N\* statistics:**

| Statistic | Value |
|---|---|
| Count | 100 (no missing) |
| Mean | −0.44 eV |
| Standard deviation | 0.31 eV |
| Min | −1.20 eV (Mn — too strong) |
| Max | +0.67 eV (Cu — too weak) |
| Optimal window | −0.50 to −0.20 eV |
| Samples in optimal window | ~35 (35%) |

#### 4.2 Feature Distribution Histograms

```python
import matplotlib.pyplot as plt

features_to_plot = [
    "electronegativity","d_band_center","work_function",
    "dG_N","nitride_formation_energy","melting_point"
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feat in zip(axes.flat, features_to_plot):
    df[feat].hist(bins=12, ax=ax, color="#00C9A7", edgecolor="white")
    ax.axvline(df[feat].mean(), color="navy", linestyle="--", label="Mean")
    ax.set_title(feat.replace("_"," ").title())
    ax.set_xlabel("Value"); ax.set_ylabel("Count")
plt.suptitle("Feature Distributions — 100 Catalyst Materials", fontsize=14)
plt.tight_layout()
plt.savefig("plots/plot_distributions.png", dpi=150)
```

**DA findings from histograms:**

| Feature | Shape | Key Insight | ML Implication |
|---|---|---|---|
| Electronegativity | Bimodal | Two natural groups: early vs. late TMs | group/period features capture this |
| d-band centre | Unimodal + outlier | Cu at −2.67 eV is >2σ from mean | Prefer MAE over RMSE as primary metric |
| ΔG_N\* target | Right-skewed | Only ~35% in optimal window | Class imbalance — LOO-CV essential |
| Nitride formation energy | Left-skewed | Strong nitride formers cluster at −1.0 eV | Top predictor — confirmed by importance |
| Melting point | Uniform spread | Wide range 1357–3695 K | High variance → informative feature |

#### 4.3 Pearson Correlation Heatmap

```python
import seaborn as sns

cols = ["electronegativity","d_electrons","melting_point","d_band_center",
        "d_band_filling","work_function","nitride_formation_energy",
        "cohesive_energy","surface_energy","dG_N"]

corr_matrix = df[cols].corr(method="pearson")

plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap (Pearson r)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/plot_correlation_heatmap.png", dpi=150)
```

> Pearson r ranges from −1 to +1. Dark RED = strong positive | Dark BLUE = strong negative | White = no linear correlation. The diagonal is always r = 1.00.

**Key correlations and their physical meaning:**

| Feature Pair | r Value | Interpretation |
|---|---|---|
| ΔG_N ↔ nitride_formation_energy | −0.48 | Drive to form nitride → stronger N* binding |
| ΔG_N ↔ d_band_center | −0.52 | Higher ε_d → stronger N binding (Hammer-Nørskov) |
| ΔG_N ↔ work_function | +0.39 | Higher Φ → weaker N binding |
| ΔG_N ↔ cohesive_energy | −0.35 | Strongly bound metals bind N strongly |
| d_electrons ↔ d_band_filling | +0.85 | Direct physics — highly correlated |
| melting_point ↔ surface_energy | +0.80 | Both measure metal bond strength — partially redundant |
| electronegativity ↔ atomic_radius | −0.55 | Classic periodic table anti-correlation |

**Critical EDA insight:** No feature achieves |r| > 0.55 with ΔG_N\*. The maximum single-feature R² is 0.27 (d-band centre). This **quantitatively proves** that a multi-feature ML model is scientifically necessary — you cannot predict catalyst activity from one physical descriptor alone.

#### 4.4 Hammer-Nørskov Scaling Scatter Plot

```python
from scipy.stats import pearsonr
import numpy as np

x = df["d_band_center"].values
y_target = df["dG_N"].values
r, p_value = pearsonr(x, y_target)
coeffs = np.polyfit(x, y_target, 1)
x_line = np.linspace(x.min()-0.2, x.max()+0.2, 200)

plt.figure(figsize=(9, 6))
plt.scatter(x, y_target, s=150, zorder=5, c="#00C9A7", edgecolors="white")
plt.plot(x_line, np.polyval(coeffs, x_line), "--",
         color="#8B5CF6", linewidth=2,
         label=f"Linear fit  r = {r:.3f},  p = {p_value:.4f}")
for i, el in enumerate(df["material"]):
    plt.annotate(el, (x[i], y_target[i]),
                 textcoords="offset points", xytext=(5,3), fontsize=8)
plt.xlabel("d-band Centre εd (eV)", fontsize=13)
plt.ylabel("ΔG_N* (eV)", fontsize=13)
plt.title("Hammer-Nørskov Scaling: d-band Centre vs. N Adsorption Energy")
plt.legend(); plt.savefig("plots/plot_dband_scaling.png", dpi=150)
```

Result: **r = −0.52**, p < 0.001 — statistically significant but explains only 27% of variance. The scatter around the trend line is the scientific justification for adding 42 more features.

#### 4.5 Volcano Plot

```python
dG_range  = np.linspace(-2.2, 1.0, 500)
left_leg  = 0.44 - 0.65 * (dG_range + 0.35)   # NNH formation barrier
right_leg = 0.44 + 0.65 * (dG_range + 0.35)   # NH₃ desorption barrier
volcano   = -np.maximum(left_leg, right_leg)   # theoretical upper bound activity

plt.figure(figsize=(11, 7))
plt.plot(dG_range, volcano, color="navy", lw=2.5, label="Theoretical volcano")
plt.axvspan(-0.50, -0.20, alpha=0.12, color="#00C9A7", label="Optimal zone")
plt.axvline(-0.35, ls="--", color="#00C9A7", lw=1.5, label="Ideal −0.35 eV")
for _, row in df[df["type"]=="bulk"].iterrows():
    plt.scatter(row["dG_N"], -(abs(row["dG_N"]+0.35)*0.65+0.09),
                s=200, zorder=5)
    plt.annotate(row["material"], ...)
plt.xlabel("ΔG_N* (eV)"); plt.ylabel("Limiting Potential UL (V)")
plt.title("Volcano Plot — eNRR Catalyst Activity")
plt.savefig("plots/plot_volcano.png", dpi=150)
```

**Volcano plot reading guide:**

| Region | ΔG_N\* Range | Examples | Problem |
|---|---|---|---|
| Too Weak (right) | > −0.20 eV | Cu (+0.67 eV) | N₂ never adsorbs — no reaction |
| ⭐ Optimal (peak) | −0.50 to −0.20 eV | Mo, Ru, Fe, Rh | Best catalysts at volcano peak |
| Too Strong (left) | < −0.50 eV | Mn, Re, W | Surface poisoned — NH₃ never desorbs |

#### 4.6 EDA Summary

| Plot | Tool | Purpose | Key Finding |
|---|---|---|---|
| Histograms | `df.hist()`, matplotlib | Distribution analysis | Cu d-band extreme outlier; target right-skewed |
| Heatmap | `df.corr()`, seaborn | Correlation structure | Best single r = −0.52; multi-feature model justified |
| Scatter + fit | `scipy.stats.pearsonr` | Scaling relation | r = −0.52 — d-band alone insufficient |
| Volcano plot | matplotlib | Activity vs binding | Mo/Ru at peak; Cu and Mn far outside |
| Feature importance | Random Forest Gini | Predictor ranking | nitride_formation_energy #1, d_band_center #2 |
| Parity plot | matplotlib | Prediction quality | LOO MAE = 0.079 eV — within DFT accuracy |

---

### 5. Data Preprocessing Pipeline

All preprocessing steps are applied **strictly before model training** to ensure valid, unbiased evaluation.

#### 5.1 Feature and Target Selection

```python
FEATURES = [
    # Atomic identity (2)
    "atomic_number", "atomic_mass",
    # Electrochemical (4)
    "electronegativity", "ionization_energy",
    "second_ionization_energy", "electron_affinity",
    # Size / geometry (6)
    "atomic_radius", "covalent_radius", "ionic_radius",
    "van_der_waals_radius", "coordination_number", "lattice_constant",
    # Electronic config (7)
    "valence_electrons", "d_electrons", "s_electrons",
    "p_electrons", "f_electrons", "group", "period",
    # Thermodynamic / bulk (8)
    "melting_point", "boiling_point", "density",
    "heat_of_fusion", "heat_of_vaporization", "specific_heat",
    "cohesive_energy", "formation_energy",
    # Chemical affinity (2)
    "nitride_formation_energy", "oxide_formation_energy",
    # Surface / mechanical (8)
    "surface_energy", "work_function",
    "bulk_modulus", "shear_modulus", "youngs_modulus", "poisson_ratio",
    "thermal_conductivity", "electrical_conductivity",
    # Electronic structure (6)
    "magnetic_moment", "d_band_center", "d_band_width",
    "d_band_filling", "fermi_energy", "band_gap",
]
TARGET = "dG_N"

X = df[FEATURES].values   # shape: (100, 43)
y = df[TARGET].values     # shape: (100,)
```

#### 5.2 Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,    # 80% training, 20% test
    random_state=42    # reproducibility — same split every run
)
# X_train.shape = (80, 43)
# X_test.shape  = (20, 43)
```

| Split | Size | Count | Purpose |
|---|---|---|---|
| Training set | 80% | 80 materials | Model learns weights from this data |
| Test set | 20% | 20 materials | Unseen — final generalisation check |

*Why `random_state=42`?* Sets the pseudo-random number seed so the same split is produced on every run — essential for reproducibility. Any integer works; 42 is conventional.

#### 5.3 Feature Scaling — RobustScaler (v3) vs StandardScaler (v2)

**v3 uses RobustScaler** — superior to StandardScaler for this dataset because several features (Cu d-band at −2.67 eV, extreme melting points) are statistical outliers that inflate the mean and standard deviation:

```python
from sklearn.preprocessing import RobustScaler

# RobustScaler formula:
# x_scaled = (x − median) / IQR
# IQR = interquartile range (Q75 − Q25) — outlier-robust
# cf. StandardScaler: x_scaled = (x − mean) / std   ← pulled by outliers

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ON training data ONLY
X_test_scaled  = scaler.transform(X_test)        # transform only — never fit

# ⚠️ CRITICAL: fit_transform on X_test = DATA LEAKAGE
# Test set statistics would "leak" into training → falsely optimistic results
```

**Why scaling is essential:**

| Feature | Raw Range | Without Scaling Effect |
|---|---|---|
| `melting_point` | 1357–3695 K | Dominates all distance computations |
| `electronegativity` | 1.55–2.36 | Dwarfed by melting_point |
| `d_band_center` | −2.67 to −1.09 eV | Artificially small influence |
| `electron_affinity` | 0.00–2.13 eV | Correct relative weight lost |

After RobustScaler: every feature is centred at its median with unit IQR — all features contribute proportionally based on their actual predictive value, not their measurement units.

#### 5.4 Handling Missing Values

```python
print(df.isnull().sum())
# All 43 ML features: 0 missing  ← dataset is clean
# Experimental FE%: 4 missing    ← W, Mn, Re, Os — kept as NaN, excluded from X

df_ml = df[FEATURES + [TARGET]].dropna()   # 100 clean samples
```

---

### 6. Feature Engineering & Selection

Feature engineering is the process of transforming raw material identifiers into numerical descriptors that ML algorithms can learn from. Without this step, the model receives only strings like "Mo" or "FeMo alloy" — meaningless to any regression algorithm.

#### 6.1 Columns Excluded and Why

| Column | Reason for Exclusion |
|---|---|
| `material`, `formula`, `material_id` | String identifiers — not numeric |
| `crystal_structure` | Categorical (FCC/BCC/HCP) — would need one-hot encoding; d-band already captures this |
| `dG_NNH`, `dG_NH`, `dG_NH2` | Target-related — using them as features causes **target leakage** |
| `experimental_FE_percent` | 4 missing values + risk of contaminating the target |
| `limiting_potential_UL` | **Derived** from ΔG_N\* — using it as a feature leaks the target entirely |

#### 6.2 Why These 43 Features — Not More, Not Fewer

Including every available elemental property would add noise features that reduce model accuracy on small datasets. Each of the 43 selected features satisfies at least one criterion:
- **Physical theory predicts** a direct link to ΔG_N\* (e.g. d_band_center via Hammer-Nørskov)
- **Statistical correlation** |r| > 0.15 with ΔG_N\* (confirmed in EDA)
- **Information not already captured** by other features in the set (low redundancy)

---

### 7. All ML Models — Theory, Mathematics & Code

Six regression models were trained using the identical preprocessing pipeline, evaluated with Leave-One-Out CV, and compared on identical metrics.

#### 7.1 Ridge Regression — 🥇 Best Model (LOO R² = 0.783)

Ridge adds an **L2 penalty** to the ordinary least squares objective, preventing overfitting when features are correlated:

```
Objective:
  Minimise:  Σᵢ (yᵢ − ŷᵢ)²  +  α × Σⱼ wⱼ²
             ─────────────────   ─────────────
             Prediction error    L2 regularisation (penalty)

Closed-form solution:
  W* = (XᵀX + αI)⁻¹ Xᵀy        ← numerically stable inversion

Compared to OLS (α = 0):
  W*_OLS  = (XᵀX)⁻¹ Xᵀy        ← singular if features are collinear!
  Ridge αI ensures invertibility even when features are perfectly correlated.
```

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

model_ridge = Ridge(alpha=1.0)   # alpha=0 → identical to LinearRegression
model_ridge.fit(X_train_scaled, y_train)
y_pred = model_ridge.predict(X_test_scaled)

print(f"Ridge coefficients: {model_ridge.coef_}")  # 43 weights
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} eV")

# alpha=0.0   → no regularisation = Linear Regression (risky with correlated features)
# alpha=1.0   → moderate shrinkage (our best performing setting)
# alpha=10.0  → strong shrinkage (reduces variance but increases bias)
# alpha=100.0 → extreme shrinkage (all weights ~0; high bias, underfitting)
```

| Property | Value |
|---|---|
| Parameters learned | 44 (43 weights + 1 intercept) |
| Key hyperparameter | α = 1.0 |
| Training method | Analytic (closed-form) |
| Loss function | MSE + L2 penalty |
| **LOO R²** | **0.783 ✅** |
| LOO MAE | 0.079 eV ✅ |
| Rank | **🥇 1st** |

**Why Ridge wins with 43 features and 100 samples:** Many features are correlated (e.g. d_electrons and d_band_filling, r = +0.85; melting_point and surface_energy, r = +0.80). Pure Linear Regression becomes numerically unstable in this multicollinear setting. Ridge's L2 penalty distributes coefficient mass across correlated predictors rather than arbitrarily amplifying one — exactly what the physics of our problem requires.

#### 7.2 Ridge (α = 10) — 🥈 2nd

```python
model_ridge10 = Ridge(alpha=10.0)
# Stronger penalty → more shrinkage → LOO R² = 0.752 (slightly worse than α=1)
# Suggests α=1 is near-optimal for this dataset size
```

#### 7.3 Lasso Regression — Sparse Feature Selection

Lasso uses an **L1 penalty** that pushes some coefficients to exactly zero — performing automatic feature selection:

```
Objective:
  Minimise:  Σᵢ (yᵢ − ŷᵢ)²  +  α × Σⱼ |wⱼ|
                                  ────────────
                              L1 penalty (diamond constraint)

No closed-form solution → solved iteratively via coordinate descent
```

```python
from sklearn.linear_model import Lasso

model_lasso = Lasso(alpha=0.01, max_iter=10000)
model_lasso.fit(X_train_scaled, y_train)

# Check which features were zeroed out
zero_features = [f for f, w in zip(FEATURES, model_lasso.coef_) if w == 0]
print(f"Lasso zeroed out {len(zero_features)} features: {zero_features}")
# LOO R² = 0.620  — worse than Ridge because L1 is too aggressive on correlated features
```

**L1 vs L2 geometric intuition:** The L2 (Ridge) penalty has a circular constraint region — the optimum lies where the loss ellipsoid touches the circle, rarely at exactly zero. The L1 (Lasso) penalty has a diamond constraint — the optimum often lies at a corner of the diamond where many weights = 0.

#### 7.4 ElasticNet — Hybrid Regularisation

```
Objective:
  Minimise: MSE  +  α × [ρ × Σⱼ|wⱼ|  +  (1−ρ)/2 × Σⱼwⱼ²]
                         ──────────────  ──────────────────────
                              L1                  L2
l1_ratio ρ = 0.5  →  equal weighting of L1 and L2
```

```python
from sklearn.linear_model import ElasticNet

model_en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
model_en.fit(X_train_scaled, y_train)
# LOO R² = 0.643 — better than Lasso alone; handles groups of correlated features better
# Lasso arbitrarily picks one feature from each correlated group; ElasticNet retains all
```

#### 7.5 Random Forest — Ensemble Bagging

Trains `n_estimators` independent Decision Trees, each on a random **bootstrap sample** of the data with a random **subset of features** at each split. Final prediction = average across all trees:

```
ŷ_RF = (1/T) × Σₜ fₜ(x)      [average of T trees]

Bootstrap sampling:  each tree trained on N samples drawn WITH replacement
Feature sampling:    each split considers m = round(0.6 × 43) = 26 features
Aggregation:         ŷ = mean(tree₁, tree₂, ..., treeT)   → reduces variance
```

```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(
    n_estimators=50,        # number of independent trees
    max_depth=4,            # maximum tree depth — limits overfitting
    min_samples_leaf=2,     # minimum samples at a leaf
    max_features=0.6,       # 60% of features considered per split
    random_state=42,
    n_jobs=-1               # use all available CPU cores
)
model_rf.fit(X_train_scaled, y_train)

# Feature importance from Gini impurity reduction (averaged over all trees)
importances = model_rf.feature_importances_   # shape: (43,)
```

| Property | Value |
|---|---|
| Algorithm type | Ensemble — Bootstrap Aggregating (Bagging) |
| Trees | 50 independent, parallelised |
| LOO R² | 0.593 |
| Test R² | 0.792 (highest test R²) |
| Rank | 🥉 5th by LOO |

**Bagging vs. Boosting — key difference:** Random Forest trees are **independent** — all trained in parallel, then averaged (reduces variance). Gradient Boosting trees are **sequential** — each corrects the residuals of all previous trees (reduces bias). Fundamentally different ensemble strategies for fundamentally different error sources.

#### 7.6 Extra Trees — Extremely Randomised Trees

Like Random Forest but **additionally randomises the split threshold** at each node, not just the feature selection:

```python
from sklearn.ensemble import ExtraTreesRegressor

model_et = ExtraTreesRegressor(
    n_estimators=50, max_depth=4,
    min_samples_leaf=2, max_features=0.6,
    random_state=42, n_jobs=-1
)
# At each node:
#   1. Draw m random features (same as RF)
#   2. For each feature, draw a RANDOM split threshold (vs. searching optimal in RF)
#   3. Select the best among these random splits
# Effect: higher bias, lower variance → faster and sometimes better on small n
```

#### 7.7 Complete Model Comparison

| Model | Algorithm Type | LOO R² | LOO MAE (eV) | Test R² | Hyperparameters | Rank |
|---|---|---|---|---|---|---|
| **Ridge (α=1)** | Parametric, Linear + L2 | **0.783** | **0.079 ✅** | 0.716 | α=1.0 | **🥇 1st** |
| Ridge (α=10) | Parametric, Linear + L2 | 0.752 | 0.086 | 0.668 | α=10.0 | 🥈 2nd |
| ElasticNet | Parametric, L1+L2 | 0.643 | 0.100 | 0.639 | α=0.01, ρ=0.5 | 🥉 3rd |
| Lasso | Parametric, Linear + L1 | 0.620 | 0.101 | 0.623 | α=0.01 | 4th |
| Random Forest | Ensemble, Bagging | 0.593 | 0.098 | **0.792** | n=50, d=4 | 5th |
| Extra Trees | Ensemble, Bagging | 0.579 | 0.099 | 0.531 | n=50, d=4 | 6th |

**Interesting divergence — RF: LOO R² = 0.593 but Test R² = 0.792.** This suggests RF overfits moderately (memorises individual training samples in LOO), but still generalises well on the held-out 20%. Ridge is more consistent across both evaluations — confirming it as the production model.

---

### 8. Model Evaluation & Metrics

#### 8.1 The Three Regression Metrics

| Metric | Formula | This Project (Best) | Physical Interpretation |
|---|---|---|---|
| **R²** | 1 − SS_res/SS_tot | 0.783 (LOO) | 78% of variance in ΔG_N\* explained by features |
| **MAE** | mean(\|y − ŷ\|) | 0.079 eV ✅ | Average absolute prediction error |
| **RMSE** | √(mean((y−ŷ)²)) | 0.110 eV | Penalises large errors (Cu outlier at +0.405 eV inflates this) |

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

y_pred = best_model.predict(X_test_scaled)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.4f} eV")
print(f"RMSE: {rmse:.4f} eV")
```

**When to use each metric:**
- **R²**: Model comparison (scale-independent)
- **MAE**: Primary metric when outliers exist (Cu's +0.405 eV error would inflate RMSE unfairly)
- **RMSE**: When large errors are especially costly (safety-critical applications)

#### 8.2 Leave-One-Out Cross-Validation (LOO-CV) — Primary Metric

LOO-CV is the gold standard for small datasets:

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

loo = LeaveOneOut()
loo_predictions = np.zeros(N)

for train_idx, test_idx in loo.split(X_scaled):
    model_tmp = Ridge(alpha=1.0)
    model_tmp.fit(X_scaled[train_idx], y[train_idx])
    loo_predictions[test_idx] = model_tmp.predict(X_scaled[test_idx])

loo_r2  = r2_score(y, loo_predictions)
loo_mae = mean_absolute_error(y, loo_predictions)
print(f"LOO R²:  {loo_r2:.4f}")
print(f"LOO MAE: {loo_mae:.4f} eV")
```

**Why LOO and not just a single train/test split?**

With 100 samples, a random 80/20 split gives only 20 test materials. Which 20 are selected (easy Mo/Ru vs. hard Cu/Mn) dramatically changes R² by ±0.15 purely by chance. LOO tests every single material exactly once — providing a reliable, seed-independent performance estimate.

```
LOO-CV structure (N = 100 iterations):
  Round 1:   Test [sample 1]      Train [samples 2–100]
  Round 2:   Test [sample 2]      Train [samples 1, 3–100]
  ...
  Round 100: Test [sample 100]    Train [samples 1–99]
  ─────────────────────────────────────────────────────
  Final LOO R² = r2_score(y_all, predictions_all)
  Every sample is tested exactly once → unbiased estimate
```

#### 8.3 5-Fold Cross-Validation — Secondary Check

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2  = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
cv_mae = -cross_val_score(model, X_scaled, y, cv=kf,
                           scoring="neg_mean_absolute_error")

print(f"5F CV R²:  {cv_r2.mean():.3f}  ±  {cv_r2.std():.3f}")
print(f"5F CV MAE: {cv_mae.mean():.3f} ± {cv_mae.std():.3f} eV")
```

#### 8.4 Parity Plot — Visual Evaluation

```python
y_pred_full = best_model.predict(X_all_scaled)

plt.figure(figsize=(8, 7))
# Colour by material type
for mat_type, colour in type_colours.items():
    mask = types == mat_type
    plt.scatter(y[mask], y_pred_full[mask], s=60, color=colour,
                edgecolors="white", lw=0.8, label=mat_type, zorder=5)

lims = [y.min()-0.05, y.max()+0.05]
plt.plot(lims, lims, "k--", lw=2, label="Perfect prediction (y = ŷ)")
plt.fill_between(lims,[l-0.10 for l in lims],[l+0.10 for l in lims],
                 alpha=0.07, color="gray", label="±0.10 eV (DFT accuracy)")
plt.xlabel("DFT ΔG_N* Actual (eV)"); plt.ylabel("ML Predicted (eV)")
plt.title(f"Parity Plot — Ridge (α=1)  |  LOO R² = {loo_r2:.3f}")
plt.legend(); plt.savefig("plots/v3_plotB_parity.png", dpi=150)
```

**Reading the parity plot:**
- Points **on** the dashed diagonal = perfect prediction
- Points **above** the line = model overpredicted (too weak binding)
- Points **below** the line = model underpredicted (too strong binding)
- Distance from diagonal = absolute error for that material

---

### 9. Feature Importance & Interpretability

#### 9.1 Random Forest Gini Importance

```python
fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": model_rf.feature_importances_   # Gini impurity reduction
}).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fi["feature"].head(20), fi["importance"].head(20), color="#00C9A7")
ax.set_xlabel("Gini Importance")
ax.set_title("Top-20 Feature Importances — Random Forest")
plt.tight_layout()
plt.savefig("plots/v3_plotC_feature_importance.png", dpi=150)
```

**Gini importance:** At each decision tree node, the model chooses a feature-threshold split. The total reduction in Gini impurity (or MSE for regression) attributed to each feature, averaged across all trees and all nodes, gives the importance score. Higher = more discriminating for predicting ΔG_N\*.

#### 9.2 Ridge Absolute Coefficients

```python
ridge_importance = pd.DataFrame({
    "feature":    FEATURES,
    "abs_coeff":  np.abs(model_ridge.coef_)
}).sort_values("abs_coeff", ascending=False)

print("Top-10 Ridge coefficients (absolute):")
print(ridge_importance.head(10).to_string())
```

**Consistent top predictors across both methods:**

| Feature | Physical Reason |
|---|---|
| `nitride_formation_energy` | Direct thermodynamic measure of N affinity |
| `d_band_center` | Hammer-Nørskov descriptor — 30 years of theory validated |
| `cohesive_energy` | Strongly bound metals bind N strongly |
| `work_function` | Fermi level position controls electron transfer |
| `d_band_filling` | Fuller d-band → more Pauli repulsion with N\* |

---

### 10. Overfitting, Underfitting & Bias-Variance Tradeoff

The most important concept in applied ML is **generalisation** — a model that performs well on *new, unseen* data, not one that merely memorised training data.

#### Diagnosis Table

| Condition | Train R² | CV/LOO R² | Gap | Example in This Project |
|---|---|---|---|---|
| Underfitting (high bias) | Low ~0.50 | Low ~0.50 | Small | Not observed — Ridge fits adequately |
| ✅ Good generalisation | High ~0.92 | High ~0.78 | < 0.15 | **Ridge (α=1) — target** |
| Overfitting (high variance) | Very high ~0.99 | Low ~0.59 | Large | Random Forest on 100 samples |

#### The Bias-Variance Decomposition

```
Total Prediction Error = Bias²  +  Variance  +  Irreducible Noise

Bias     = error from wrong model assumptions (underfitting)
           → Ridge with large α; Lasso with too much sparsity
Variance = error from sensitivity to training set (overfitting)
           → Random Forest memorising individual samples
Noise    = inherent randomness in DFT labels (~0.05 eV)
```

The **bias-variance tradeoff** is why increasing model complexity (more trees, deeper networks) helps up to a point, then hurts: complexity reduces bias but increases variance. The optimal model complexity for 100 samples and 43 features sits in the linear/regularised-linear region — exactly where Ridge wins.

#### Overfitting Control Techniques Used

| Technique | Where Applied | Effect |
|---|---|---|
| LOO-CV | All 6 models | Large train/LOO gap reveals overfitting |
| L2 Regularisation (α=1.0) | Ridge | Shrinks weights, reduces variance |
| L1 Regularisation (α=0.01) | Lasso, ElasticNet | Sparsity, automatic feature selection |
| Max depth limit (depth=4) | RF, ExtraTrees | Prevents trees from memorising training data |
| Min samples per leaf (=2) | RF, ExtraTrees | Leaf nodes need ≥2 training points |
| Model simplicity | Final selection | Occam's Razor: simplest model that fits the data |

---

### 11. Catalyst Screening & New Predictions

After training, the best model screens **29 new candidate materials** that were never in the training set:

```python
import joblib

# Load saved model and scaler — no retraining needed
best_model = joblib.load("best_model_v3.pkl")
scaler     = joblib.load("scaler_v3.pkl")

# Define new candidates with all 43 features
# (see future_prediction_v3.py for the full feature list)
new_candidates_X = scaler.transform(new_candidates[FEATURES].values)
predicted_dG     = best_model.predict(new_candidates_X)

new_candidates["predicted_dG_N"]    = predicted_dG
new_candidates["dist_from_optimal"] = abs(predicted_dG - (-0.35))
results = new_candidates.sort_values("dist_from_optimal")
results.to_csv("screening_results_v3.csv", index=False)
```

#### Production Time Model

For a lab-scale electrolyser at 10 mA cm⁻² over 4 cm²:

```python
def faradaic_efficiency(dG):
    """Empirical FE model — peaks at optimal ΔG_N* = −0.35 eV."""
    return 0.30 * np.exp(-0.5 * ((dG + 0.35) / 0.18)**2)

def nh3_production_time_hours(dG, current_mA=10, area_cm2=4, target_mg=1.0):
    """
    N₂ + 6H⁺ + 6e⁻ → 2NH₃  →  3 electrons per NH₃ molecule
    rate [mol s⁻¹] = FE × I_A / (3 × F)
    """
    fe     = faradaic_efficiency(dG)
    I_A    = current_mA * area_cm2 * 1e-3        # convert mA to A
    rate   = fe * I_A / (3.0 * 96485)            # mol NH₃ s⁻¹
    mg_per_s = rate * 17031                       # mg NH₃ s⁻¹  (MW = 17.031 g/mol)
    return (target_mg / mg_per_s) / 3600         # hours to produce 1 mg NH₃
```

---

### 12. ML & DA Results Summary

| Category | Result |
|---|---|
| ML task type | Supervised Learning → Regression |
| Dataset size | 100 materials × 43 features |
| Best model | Ridge Regression (α=1.0) |
| LOO-CV R² | **0.783** (was 0.707 in v2: +10.7% improvement) |
| LOO-CV MAE | **0.079 eV ✅** (within DFT chemical accuracy ~0.10 eV) |
| Predictions within ±0.10 eV | **75%** of training samples |
| Predictions within ±0.05 eV | **43%** of training samples |
| Top training metal | Mo (ΔG_N\* = −0.28 eV, FE = 8.1%) |
| Top screening candidate | RuCoMo alloy (−0.370 eV, FE = 29.8%, stable) |
| Most important feature | `nitride_formation_energy` |
| ML speedup vs. DFT | **~1,000,000×** (0.001 s vs. 6–24 hours) |
| Logistic Regression used? | ❌ No — regression task, not classification |
| Supervised or unsupervised? | ✅ Supervised — all samples have known DFT labels |
| Plots generated | 8 publication-quality plots |

---

## 📈 Results

### Model Performance

| Model | LOO R² | LOO MAE (eV) | Test R² | Test MAE (eV) | Rank |
|---|---|---|---|---|---|
| **Ridge (α=1)** | **0.783** | **0.079 ✅** | 0.716 | 0.065 | **🥇** |
| Ridge (α=10) | 0.752 | 0.086 | 0.668 | 0.071 | 🥈 |
| ElasticNet | 0.643 | 0.100 | 0.639 | 0.074 | 🥉 |
| Lasso | 0.620 | 0.101 | 0.623 | 0.075 | 4th |
| Random Forest | 0.593 | 0.098 | 0.792 | 0.057 | 5th |
| Extra Trees | 0.579 | 0.099 | 0.531 | 0.085 | 6th |

### Improvement: v2 → v3

| Metric | v2 (16 features, 43 samples) | v3 (43 features, 100 samples) | Change |
|---|---|---|---|
| Best LOO R² | 0.707 | **0.783** | **+10.7%** |
| Best LOO MAE | 0.076 eV | 0.079 eV | −3.9% |
| Features | 16 | **43** | +169% |
| Samples | 43 | **100** | +133% |

### Residual Statistics (Best Model)

```
RMSE                     = 0.110 eV
Mean |error|  (LOO MAE)  = 0.079 eV
Within ±0.10 eV          = 75.0% of samples
Within ±0.05 eV          = 43.0% of samples
Maximum |error|          = 0.405 eV  (Cu — anomalous full d-band)
```

Hardest-to-predict materials: Cu (+0.405), Pd (+0.379), Pt (+0.315). All three are **noble metals** whose electronic structure is dominated by relativistic effects (5d orbital contraction) and completely filled d-bands (d¹⁰) — a regime poorly represented in a training set dominated by partially-filled d-band catalytic metals.

### Top Screening Results

| Rank | Material | Type | ΔG_N\* | FE% | Time/1mg NH₃ | Confidence | Stability |
|---|---|---|---|---|---|---|---|
| 1 | **RuCoMo** | Alloy | −0.370 eV | 29.8% | 0.40 h | HIGH | ✅ Stable |
| 2 | **FeMo2Ru** | Alloy | −0.383 eV | 29.5% | 0.40 h | HIGH | ✅ Stable |
| 3 | **Cr-SAC/N-C** | SAC | −0.371 eV | 29.8% | 0.40 h | HIGH | ✅ Stable |
| 4 | **FeCoRu** | Alloy | −0.405 eV | 28.6% | 0.41 h | HIGH | ✅ Stable |
| 5 | **Co-SAC/BN** | SAC_BN | −0.409 eV | 28.4% | 0.42 h | HIGH | ✅ Stable |
| 6 | **NiMoRu** | Alloy | −0.291 eV | 28.4% | 0.42 h | HIGH | ✅ Stable |
| 7 | **Fe-SAC/V4C3** | MXene | −0.365 eV | 29.9% | 0.39 h | HIGH | ⚠️ Delamination |
| 8 | **Fe-SAC/Ti2N** | MXene | −0.368 eV | 29.8% | 0.40 h | HIGH | ⚠️ Delamination |

> **Priority for synthesis:** RuCoMo, FeMo2Ru, Cr-SAC/N-C, and FeCoRu — HIGH confidence, stable, and within 0.05 eV of the ideal −0.35 eV target.

---

## 🗂️ Repository Structure

```
eNRR-ML-Catalyst-Discovery/
│
├── 📄 README.md
├── 📄 LICENSE
│
├── 🐍 future_prediction_v3.py          ← Main pipeline (43 features, 100 samples)
├── 🐍 future_prediction_v2.py          ← Previous pipeline (16 features, 43 samples)
│
├── 📦 models/
│   ├── best_model_v3.pkl               ← Trained Ridge model (joblib)
│   └── scaler_v3.pkl                   ← RobustScaler (joblib)
│
├── 📊 data/
│   ├── screening_results_v3.csv        ← Predictions for 29 new candidates
│   └── master_dataset.csv              ← Full 100-sample training dataset
│
└── 🖼️ plots/
    ├── v3_plotA_model_comparison.png   ← LOO R², MAE, Test R² — all 6 models
    ├── v3_plotB_parity.png             ← DFT vs. ML parity scatter
    ├── v3_plotC_feature_importance.png ← Top-20 feature importances
    ├── v3_plotD_screening_volcano.png  ← Activity volcano with 29 candidates
    ├── v3_plotE_top_candidates.png     ← Ranked bar chart with ±MAE error bars
    ├── v3_plotF_nh3_time.png           ← FE% and production time vs. ΔG_N*
    ├── v3_plotG_v2_vs_v3.png           ← Version comparison (improvement shown)
    └── v3_plotH_residuals.png          ← Residual distribution + scatter
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/eNRR-ML-Catalyst-Discovery.git
cd eNRR-ML-Catalyst-Discovery

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.11.0
joblib>=1.3.0
seaborn>=0.12.0
pymatgen>=2024.1.0
mendeleev>=0.14.0
mp-api>=0.41.0
```

---

## 🚀 Usage

### Run the Full v3 Pipeline

```bash
python future_prediction_v3.py
```

Expected runtime: **~60 seconds** on any modern laptop (LOO-CV on 100 samples × 6 models).

### Predict a New Material

```python
import joblib, numpy as np

model  = joblib.load("models/best_model_v3.pkl")
scaler = joblib.load("models/scaler_v3.pkl")

# Provide all 43 feature values for your new material
# (same order as FEATURES list in future_prediction_v3.py)
new_material_features = np.array([[
    42, 95.96,          # atomic_number, atomic_mass  (Mo example)
    2.16, 7.09, 16.16, 0.746,   # electronegativity, IE, IE2, EA
    1.39, 1.54, 0.69, 2.17, 8, 3.15,  # radii, CN, lattice
    6, 5, 1, 0, 0, 6, 5,        # valence/d/s/p/f electrons, group, period
    2896, 4912, 10.28, 37.48, 598, 0.251, 6.82, 0.00,  # thermal/bulk
    -0.88, -5.90,       # nitride_formation_energy, oxide_formation_energy
    2.91, 4.36, 261, 126, 329, 0.31, 138.0, 18.7,  # surface/mechanical
    0.00, -1.30, 4.20, 0.70, -0.12, 0.0  # electronic structure
]])

X_scaled = scaler.transform(new_material_features)
dG = model.predict(X_scaled)[0]

print(f"Predicted ΔG_N* = {dG:.3f} eV")
if   -0.50 <= dG <= -0.20: print("⭐ OPTIMAL — Excellent eNRR candidate!")
elif -0.60 <= dG < -0.50:  print("✅ Borderline — Worth investigating")
elif dG > 0:               print("❌ Too Weak — N₂ will not adsorb")
else:                      print("❌ Too Strong — Surface will be poisoned")
```

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Dataset size** — 100 samples with 43 features (ratio 2.3:1). Standard practice suggests ≥10× samples per feature. Noble metals (Cu, Pd, Pt) are systematically mispredicted due to relativistic effects underrepresented in training data.

2. **Alloy feature averaging** — alloy descriptors are composition-weighted averages. This misses synergistic effects: surface strain, ensemble effects, ligand effects at heteroatom interfaces.

3. **Static atomic properties** — the model uses ground-state elemental data. It does not capture: surface reconstruction under applied bias, adsorbate-induced segregation, or electrochemical double-layer effects.

4. **No explicit solvation** — DFT labels include implicit solvent corrections, but not explicit water structure, which shifts ΔG_N\* by 0.05–0.15 eV in acidic media.

### Future Directions

| Direction | Description | Expected Benefit |
|---|---|---|
| Expand dataset to 500+ | High-throughput DFT (VASP/QE) for alloys and MXenes | Better generalisation, LOO R² > 0.90 |
| Graph Neural Networks | Represent catalysts as atomic graphs (MACE, CHGNet) | Learns from geometry directly — no manual features |
| Active Learning | ML selects next DFT calculation; iterate to convergence | Minimise compute cost; maximise accuracy |
| Multi-target prediction | Simultaneously predict ΔG_N\*, ΔG_NH\*, ΔG_NH₂\*, HER | Full free-energy pathway — more reliable screening |
| SHAP values | Model-agnostic feature attribution | Directional, interaction-aware interpretability |
| Experimental validation | Test top candidates in electrochemical cell | Close the loop: computation → ML → experiment |
| Bayesian optimisation | Uncertainty-guided candidate selection | Probabilistically optimal next-experiment choice |

---

## 📚 References

### Foundational Theory

1. **Skulason, E. et al.** (2012). A theoretical evaluation of possible transition metal electro-catalysts for N₂ reduction. *Phys. Chem. Chem. Phys.*, 14(3), 1235–1245.
2. **Nørskov, J.K. et al.** (2004). Origin of the overpotential for oxygen reduction at a fuel-cell cathode. *J. Phys. Chem. B*, 108(46), 17886–17892.
3. **Hammer, B. & Nørskov, J.K.** (1995). Why gold is the noblest of all the metals. *Nature*, 376, 238–240.
4. **Hohenberg, P. & Kohn, W.** (1964). Inhomogeneous electron gas. *Physical Review*, 136(3B), B864.
5. **Kohn, W. & Sham, L.J.** (1965). Self-consistent equations including exchange and correlation effects. *Physical Review*, 140(4A), A1133.

### eNRR Catalyst Studies

6. **Vojvodic, A. et al.** (2014). Exploring the limits: A low-pressure, low-temperature Haber-Bosch process. *Chem. Phys. Lett.*, 598, 108–112.
7. **Andersen, S.Z. et al.** (2019). A rigorous electrochemical ammonia synthesis protocol with quantitative isotope measurements. *Nature*, 570, 504–508.
8. **Wang, S. et al.** (2023). Single-atom catalysts for electrochemical nitrogen reduction. *ACS Catalysis*, 13(2), 1234–1256.
9. **Liu, C. et al.** (2022). Rational design of single-atom catalysts for nitrogen reduction. *Nat. Commun.*, 13, 4108.
10. **Qing, G. et al.** (2020). Recent advances and challenges of electrocatalytic N₂ reduction to ammonia. *Chem. Rev.*, 120(12), 5437–5516.
11. **Zhao, J. & Chen, Z.** (2017). Single Mo atom supported on defective boron nitride monolayer as an efficient electrocatalyst for nitrogen fixation. *J. Am. Chem. Soc.*, 139(36), 12480–12487.

### Machine Learning for Materials

12. **Kitchin, J.R.** (2018). Machine learning in catalysis. *Nature Catalysis*, 1, 230–232.
13. **Behler, J.** (2016). Perspective: Machine learning potentials for atomistic simulations. *J. Chem. Phys.*, 145(17), 170901.
14. **Freeze, J.G. et al.** (2019). Search for catalysts by inverse design. *Chem. Rev.*, 119(11), 6595–6612.

---

## 📋 Citation

```bibtex
@software{enrr_ml_2025,
  author    = {Your Name},
  title     = {Machine Learning Guided Discovery of Catalysts for
               Electrochemical Nitrogen Reduction (eNRR)},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/YOUR_USERNAME/eNRR-ML-Catalyst-Discovery},
  note      = {MSc CS project — 43-feature ML pipeline, 100 DFT-labelled samples,
               6 regression models, LOO R² = 0.783, MAE = 0.079 eV}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ⚗️ Quantum Chemistry · ⚛️ Density Functional Theory · 🤖 Machine Learning**

*MSc Computer Science — Data Mining & Machine Learning — 2025*

*"The goal of theoretical chemistry is to understand nature well enough to predict it."*

<br/>

⭐ **Star this repository if it was useful to your research!**

</div>
