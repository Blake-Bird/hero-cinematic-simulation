![Banner](./im.png)
*Image credit: [Sarah Kane](https://www.ast.cam.ac.uk/people/sarah.kane)*

# A JAX-Accelerated Stellar Stream Generator

StreaMAX is a lightweight, high-performance simulator for stellar streams built with JAX.  
It’s designed to make modeling and inference on stellar streams fast enough for Bayesian analysis and modern data-intensive workflows.

---

## 🚀 Features

- Particle-spray modeling of stellar streams  
- Fast 2D track extraction  
- JAX-accelerated integration for GPU/TPU compatibility  
- Automatic differentiation, can easily compute gradients of the models directly via JAX  
- Modular architecture

---

## 🧩 Models

StreaMAX supports:
- Rapid generation of stellar streams using the particle-spray method  
- Efficient extraction of stream tracks for analysis and inference  

Host potentials include point mass, Plummer, Hernquist, NFW, Miyamoto-Nagai,
logarithmic, exponential disk, and a rotating bar.

**Flattened NFW haloes come in two flavours.** `'NFW'` applies the axis ratio to
the *potential*, which is cheap but produces unphysical (negative) densities for
strong flattening. `'NFWFlatDensity'` applies it to the *density* and recovers
the potential via Chandrasekhar's homoeoidal integral, guaranteeing `rho > 0`
for any `q`. Validated against analytic spherical NFW, Miyamoto-Nagai via a
direct 3D Poisson integral, and `galpy`'s `TriaxialNFWPotential` across
`q` in [0.5, 1.5]. Use `'NFWFlatDensity'` for halo-shape inference.

---

## 🧪 Quick Start

See the `quick_start.ipynb` notebook for an example on how to:

- Define potential and stream parameters  
- Generate a stream using the particle-spray method  
- Extract and visualize the resulting track  

---

## ⚙️ Installation

### From PyPI

```bash
pip install StreaMAX
```

### From GitHub (development version)

```bash
git clone https://github.com/David-Chemaly/StreaMAX.git
cd StreaMAX
pip install -e .
```

### Manual dependency installation

```bash
pip install -r requirements.txt
```

---

## 📚 Citation & License

This project is released under the MIT License.  
If you use StreaMAX in your research, please cite and reference the following paper:

[Chemaly, D. et al. 2026, Hierarchical bayesian inference: constraining population distribution of dark matter halo shapes via stellar streams](https://ui.adsabs.harvard.edu/abs/2026arXiv260115373C/abstract)

The inference pipeline built on StreaMAX — fitting halo shapes to observed
stream tracks, individually and hierarchically across a population — lives at
[**StreaMAX-fit**](https://github.com/David-Chemaly/StreaMAX-fit).

---

## 🪐 Contributing

Pull requests and feature suggestions are welcome.  
If you encounter bugs or wish to contribute new potentials or integrators, please open an issue on GitHub.

---

**Fast. Differentiable. Modular. JAX-powered.**