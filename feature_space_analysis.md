# Feature Space Multicollinearity & Statistical Separability Report
**Author:** Shravani (Machine Learning Specialist)  
**Date:** 2024-11-12  
**Dataset:** Tomato Bacterial Wilt Hyperspectral Vectors (180 Bands)  

---

## 1. Pearson Correlation Structure
Spectral reflectance measured by continuous narrowband spectrometers creates smooth, highly autocorrelated spectral trajectories. 
- Contiguous band correlation: **$\bar{r} = 0.9984$**
- Over $98\%$ of non-adjacent band pairs within the same physiological window (e.g., $750-900\text{ nm}$) have Pearson $r > 0.95$.

### Practical Implication for Machine Learning:
- Inverting the covariance matrix $\mathbf{X}^T \mathbf{X}$ is prone to numerical instability (condition number $\kappa > 10^7$).
- Weight vectors in linear models will oscillate wildly without L1/L2 penalties.
- PCA and Savitzky-Golay derivative filtering are strongly indicated.

---

## 2. Univariate Class Separability (ANOVA $F$-tests)
To benchmark feature utility before non-linear modeling, one-way ANOVA $F$-tests were computed across all 5 severity levels for each wavelength:

$$\mathcal{F} = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$$

### Distinct Separability Hotspots:
1. **Primary Chlorophyll $a$ Peak ($664.8\text{ nm} - 668.2\text{ nm}$):**  
   $\mathcal{F} = 133.79$, $p = 1.99 \times 10^{-64}$.  
   Directly reflects the loss of light-harvesting complex integrity.
2. **Orange-Red Carotenoid Region ($607.8\text{ nm} - 617.9\text{ nm}$):**  
   $\mathcal{F} = 128.62$, $p = 7.65 \times 10^{-63}$.  
   Captures foliar yellowing and chlorosis.
3. **NIR Parenchyma Turgor Peak ($936.3\text{ nm} - 949.7\text{ nm}$):**  
   $\mathcal{F} = 126.47$, $p = 3.59 \times 10^{-62}$.  
   Reflects tissue dehydration and anatomical structural collapse.

