# Project Conventions and Technical Standards
**Project:** Physics-Informed Neural Networks for Parameter Estimation and Hidden-State Recovery in an SEIRD Model with Time-Varying Transmission  
**Case Study:** 2014--2016 West African Ebola Outbreak (Sierra Leone, $N = 7{,}092{,}113$)  
**Module 1 Technical Deliverables (Task T02 & Extensions)**

---

## 1. Mathematical Formulation & Parameter Conventions

The population $N$ is normalized to unity ($s = S/N, e = E/N, i = I/N, r = R/N, d = D/N$), satisfying the physical conservation invariant:
$$s(t) + e(t) + i(t) + r(t) + d(t) = 1 \quad \forall t \ge 0$$

The system of ordinary differential equations (ODEs) is:
$$\frac{ds}{dt} = -\beta(t) \, s(t) \, i(t)$$
$$\frac{de}{dt} = \beta(t) \, s(t) \, i(t) - \sigma \, e(t)$$
$$\frac{di}{dt} = \sigma \, e(t) - (\gamma + \mu) \, i(t)$$
$$\frac{dr}{dt} = \gamma \, i(t)$$
$$\frac{dd}{dt} = \mu \, i(t)$$

### Standardized Ebola Epidemiological Parameters & Units:
- **$t$**: Time in days.
- **$\beta(t)$**: Dynamic contact and transmission rate per day ($\text{day}^{-1}$). Continuous function learned via neural network with $\text{Softplus}$ activation.
- **$\sigma$**: Incubation transition rate from Exposed to Infectious ($\text{day}^{-1}$):
  $$\sigma = \frac{1}{9.4} \approx 0.1064\text{ day}^{-1}$$
  (Mean incubation period $1/\sigma \approx 9.4\text{ days}$, fixed from WHO Ebola Response Team, *NEJM* 2014).
- **$\gamma$**: Recovery rate ($\text{day}^{-1}$):
  $$\gamma = \frac{1}{7.0} \approx 0.1429\text{ day}^{-1}$$
  (Mean infectious duration among survivors $1/\gamma \approx 7.0\text{ days}$).
- **$\mu$**: Disease-induced mortality rate ($\text{day}^{-1}$):
  $$\mu \approx 0.1000\text{ day}^{-1}$$
  yielding an overall untreated case-fatality ratio $\text{CFR} = \frac{\mu}{\gamma + \mu} \approx \frac{0.1000}{0.2429} \approx 41.2\%$ (and up to $65-70\%$ in hospitalised subsets), characteristic of the West African Zaire ebolavirus strain.
- **$R_0$**: Basic reproduction number:
  $$R_0 = \frac{\beta}{\gamma + \mu}$$
  For baseline transmission $\beta \approx 0.4858\text{ day}^{-1}$, $R_0 \approx 2.00$ (consistent with consensus estimates $R_0 \in [1.5, 2.5]$ from Althaus 2014 and Rivers et al. 2014).
- **$R_t$**: Instantaneous effective reproduction number:
  $$R_t = \frac{\beta(t) \cdot s(t)}{\gamma + \mu}$$
- **$p_c$**: Critical herd immunity / vaccination coverage threshold:
  $$p_c = 1 - \frac{1}{R_0}$$
- **$\delta_c$**: Critical contact reduction factor required to halt transmission:
  $$\delta_c = 1 - \frac{1}{R_0}$$

---

## 2. Module 1 Analytical Extensions (Own Mathematical Analysis on Published Models)

In accordance with project instructions ("Published models may be used as a starting point, but each group must demonstrate its own mathematical analysis, implementation, interpretation, or meaningful extension"):
1. **Comparative Synthesis of Published Models**: Module 1 synthesizes and critiques 3 seminal published models: Legrand et al. (2007), Rivers et al. (2014), and Althaus (2014).
2. **Phase-Plane & Nullcline Analysis**: Derivation of total active infection trajectory $(e+i)(s)$:
   $$\frac{d(e+i)}{ds} = -1 + \frac{1}{R_0 s} \implies (e+i)(s) = (e_0 + i_0) + s_0 - s + \frac{1}{R_0}\ln\left(\frac{s}{s_0}\right)$$
   Yielding theoretical maximum active peak at nullcline $s^* = 1/R_0$:
   $$(e+i)_{\max} = (e_0 + i_0) + s_0 - \frac{1}{R_0}\left(1 + \ln(R_0 s_0)\right)$$
3. **Threshold & Intervention Trade-Off Curves**: Analytical computation of $p_c(R_0)$ and 2D parameter space contour analysis of $R_0(\beta, \gamma + \mu)$.

---

## 3. Real-World Outbreak Dataset Specification

- **Outbreak Target**: Republic of Sierra Leone 2014 Ebola Outbreak ($N = 7{,}092{,}113$).
- **Source**: WHO Ebola situation reports as published on the Humanitarian Data Exchange (HDX, dataset `ebola-cases-2014`, file `ebola_data_db_format.csv`): laboratory-confirmed cumulative cases and deaths for Sierra Leone, 29 Aug 2014 - 6 Nov 2015, real report dates only (no interpolation).
- **Partition (E5)**: 29 Aug 2014 - 25 Apr 2015 (days 0-239); first 75% (days 0-179) for model calibration, remaining 25% (days 180-239) strictly held out for testing. E7 uses rolling 8-week fits and 4-week forecasts on the full series.
- **Notice**: No synthetic datasets are to be used for the final calibration task; real data must be fetched using instructions in `data/README.md`.

---

## 4. Shared Computational Conventions

1. **Random Seeds**: Locked to `42` across NumPy, PyTorch, and random generators:
   ```python
   import torch, numpy as np, random
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(42)
   ```
2. **Directory & File Hierarchy**:
   - `notebook.ipynb`: Master Jupyter notebook running top-to-bottom.
   - `report/main.tex`: Editable master LaTeX document.
   - `report/refs.bib`: BibTeX bibliography database.
   - `figs/`: Vector EPS figure exports without alpha channel.
   - `data/`: Outbreak dataset documentation and processing scripts (`data/README.md`).

3. **Figure Export Standard**:
   - EPS format only (`plt.savefig("figs/fig_name.eps", format="eps", bbox_inches="tight")`).
   - Clean, publication-quality palettes, labeled axes with units, distinct line styles.
