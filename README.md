<div align="center">

# VoRTeC: Taming Foundation Flow for One-step Real-time Video Compression

[![arXiv](https://img.shields.io/badge/arXiv-2026.02291-b31b1b.svg)](http://arxiv.org/abs/2609.02291)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Framework](https://img.shields.io/badge/Built%20on-Wan2.1%201.3B-FF6F00.svg)](https://github.com/Wan-Video/Wan2.1)
[![Code Status](https://img.shields.io/badge/Code-Partial%20Open--Source-yellow.svg)](#-release-status)
[![Stars](https://img.shields.io/github/stars/Darc8-sun/VoTec-video-compression?style=social)](https://github.com/Darc8-sun/VoTec-video-compression/)

<p align="center">
  <a href="http://arxiv.org/abs/2609.02291"><b>Paper</b></a> ·
  <a href="https://darc8-sun.github.io/VoRTec_compress/"><b>Project Page</b></a> ·
  <a href="#-method-overview">Method</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-citation">Citation</a>
</p>


</div>

---

> **TL;DR** &nbsp; VoRTeC is the **first one-step generative video codec** built on top of a **video-native flow-matching foundation model** (Wan2.1). It tames a billion-parameter DiT into a single decoding step without touching its weights or gradients, achieving **58–73% bitrate savings** over prior diffusion-based codecs **and** decoding speeds up to **3–197× faster**, reaching **32 FPS at 480P** and **3.93 FPS at 1080P**.

<p align="center">
  <img src="assets/teaser.png" alt="VoRTeC teaser — rate–perception curves and qualitative comparison" width="96%">
</p>

## 📢 News

- **[2026-09-02]** &nbsp; 🎉 Initial repository created with the **partial open-source release** accompanying the arXiv preprint.
- **[2026-09-02]** &nbsp; 📝 Paper released on arXiv (link TBD).
- **[Coming]** &nbsp; 🔓 Full inference code and pretrained weights to be released upon paper acceptance.

## ✨ Highlights

| | |
|---|---|
| ⚡ **Real-time decoding** | **32 FPS @ 480P** · **13 FPS @ 720P** · **3.93 FPS @ 1080P** on a single A800 |
| 📉 **Bitrate savings** | **58.12–73.25%** LPIPS BD-Rate over prior diffusion-based video codecs |
| 🔁 **Speedup over diffusion-based baselines** | **3×–197×** faster decoding |
| 🚫 **Non-invasive to backbone** | Wan2.1 weights & gradients are **never accessed** during training |
| 🧩 **Single-step decoding** | No iterative flow-matching sampling — one forward pass of the foundation model |

## 🧠 Method Overview

<p align="center">
  <img src="assets/pipeline.png" alt="VoRTeC pipeline overview" width="95%">
  <br>
  <em>Overall pipeline of VoRTeC. A frozen 3D encoder/decoder ℰ/𝒟 from Wan2.1 produces a spatio-temporal latent; a lightweight <strong>Latent Codec</strong> compresses it; a <strong>Flow State Estimator</strong> projects the compressed latent onto the flow path; the frozen <strong>Flow DiT</strong> performs a single-step denoising; and a Transformer-based <strong>Flow Prior Multi-Fusion (FPMF)</strong> module refines the reconstruction and enables end-to-end learning of the prior distribution.</em>
</p>

VoRTeC comprises three core technical ingredients:

1. **Flow-State Estimation (FSE).** &nbsp; Treats each compressed latent as a state along the foundation flow's trajectory and solves for the optimal timestep .
2. **Flow Prior Multi-Fusion (FPMF).** &nbsp; A ViT-based module that fuses coarse-grained compressed tokens with fine-grained prior tokens using **multi-scale patchification** — coarse patches capture low-frequency structure, fine patches attend to high-frequency textures.
3. **Contact-Group-to-Group (CGG).** &nbsp; A *training-free* inter-group caching mechanism that routes the **last frame** of each group into the **first frame** of the next via latent-cache, frame-cache and prior-cache, guaranteeing **temporal consistency** across GOPs without any extra supervision or inference overhead.

<p align="center">
  <img src="assets/fpmf.png" alt="Flow Prior Multi-Fusion architecture" width="90%">
  <br>
  <em>Figure: Architecture of the Flow Prior Multi-Fusion module.</em>
</p>

## 📊 Results

### Rate–distortion / Rate–perception curves

<p align="center">
  <img src="assets/rd_curves.png" alt="R–D / R–P curves on HEVC-B and UVG" width="95%">
</p>

VoRTeC consistently outperforms prior **neural**, **generative**, and **traditional** video codecs on HEVC ClassB (720P / 1080P) and UVG (720P / 1080P). With LoRA fine-tuning, **VoRTeC⁺** further extends the gains to **64.53%–74.80%** LPIPS BD-Rate savings.

### Decoding throughput

Encoding / decoding FPS on a single NVIDIA A800 (VTM-17.0 evaluated on an Intel Xeon Gold 6330):

| Method          | 416×240   | 832×480   | 1280×720  | 1920×1080 |
|-----------------|-----------|-----------|-----------|-----------|
| VTM-17.0        | 0.47 / 121.73 | 0.12 / 56.62 | 0.08 / 36.92 | 0.02 / 12.35 |
| DCVC-DC         | 36.24 / 43.88 | 15.63 / 18.71 | 7.47 / 9.36 | 3.43 / 4.36 |
| DCVC-FM         | 36.51 / 43.09 | 16.82 / 17.85 | 7.58 / 8.94 | 3.51 / 4.22 |
| GLC-Video       | 49.39 / 57.22 | 28.60 / 19.92 | 13.25 / 8.82 | 6.36 / 4.19 |
| DiffVC          | 0.85 / 0.88 | 0.29 / 0.29 | 0.09 / 0.09 | 0.02 / 0.02 |
| GNVC-VD         | —          | < 40 / < 7.75 | 17.24 / 2.59 | 6.54 / 0.64 |
| S2VC            | —          | —          | —          | 6.60 / 1.27 |
| YODA            | —          | —          | —          | — / 0.97  |
| **VoRTeC (Ours)** | **84.46 / 105.25** | **20.34 / 32.31** | **8.76 / 12.60** | **3.84 / 3.93** |

> Format: *encoding FPS / decoding FPS*. **Higher is better.**

### Qualitative comparison

<p align="center">
  <img src="assets/qualitative.png" alt="Qualitative comparison with prior codecs" width="95%">
</p>

### Temporal consistency

<p align="center">
  <img src="assets/temporal_consistency.png" alt="Temporal consistency — FloLPIPS and E_warp" width="95%">
</p>

Across both FloLPIPS (optical-flow-aware perceptual distance) and warping error `E_warp` (inter-frame continuity), VoRTeC consistently outperforms diffusion-based baselines and even matches or beats distortion-oriented neural codecs.

## 🛠️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/VoRTeC.git
cd VoRTeC

# 2. Create a fresh environment
conda create -n vortec python=3.10 -y
conda activate vortec

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the Wan2.1-1.3B foundation flow weights from
#    https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
#    and place them under ./checkpoints/Wan2.1-T2V-1.3B/

# 5.TBD
```

> **Hardware** &nbsp; A single NVIDIA GPU with ≥ 24 GB memory (we develop on an A6000 / A800) is sufficient for evaluation. Full training fits on a single 48 GB GPU.



## 📚 Citation

If you find VoRTeC useful for your research, please cite our work. The BibTeX entry will be updated upon acceptance; a placeholder is provided below:

```bibtex
@misc{xia2026vortectamingfoundationflow,
      title={VoRTeC: Taming Foundation Flow for One-step Real time Video Compression}, 
      author={Yichong Xia and Qinhong Wu and Bin chen and Jinpeng Wang and Zeyuan Chen and Haoqian Wang},
      year={2026},
      eprint={2609.02291},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.02291}, 
}
```

The full bibliographic information will be added once the paper is accepted.

## 📬 Contact

For questions about the code or the paper, please open an issue on this repository, or contact `xiayc23@tsinghua.edu.cn` / `chenbin2021@hit.edu.cn`.

## 📄 License

This repository is released under the [Apache License 2.0](./LICENSE).

Note that several third-party components are bundled as sub-directories and retain their original licenses:

| Component | License | Upstream |
|---|---|---|
| `wan/` | Apache 2.0 | github.com/Wan-Video/Wan2.1 |
| `compressai/` | BSD 3-Clause Clear | github.com/InterDigitalInc/CompressAI |

---

<div align="center">
  ⭐ Star this repo if you find VoRTeC useful — it helps us prioritize the full open-source release.
</div>
