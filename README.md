# KnobGen
## Controlling the Sophistication of Artwork in Sketch-Based Diffusion Models

KnobGen is a dual-pathway framework that empowers sketch-based image generation diffusion model by seamlessly adapting to varying levels of sketch complexity and user skill. KnobGen employs a Coarse-Grained Controller (CGC) module for leveraging high-level semantics from both textual and sketch inputs in the early stages of generation, and a Fine-Grained Controller (FGC) module for detailed refinement later in the process. 

![KnobGen Architecture](misc/HEDFusion.jpg)

More details available in our [paper](https://arxiv.org/abs/2410.01595).

## Quick Demo
![KnobGen Architecture](misc/quick_demo.PNG)
## :rocket: News
- [2024-09-27] 🔥 Initial release of KnobGen code!
- [2024-10-02] 🔥 The paper is released on arXiv.

## Table of Contents
Follow steps 1-3 to run our pipeline. 
1. [Installation](#Installation)
2. [Prepare the Dataset](#Prepare-the-Dataset)
3. [Train](#Train)
4. [Inference](#inference)
5. [Results](#Results)

## Installation
To set up the environment, please follow these steps in the terminal:
```shell
git clone https://github.com/aminK8/KnobGen.git
cd KnobGen
conda env create -f environment.yml
conda activate knobgen
```

## Prepare the Dataset

## Train

## Inference

## Results

 Our method democratizes sketch-based image generation by effectively handling a broad spectrum of sketch complexity and user drawing ability—from novice sketches to those made by seasoned artists—while maintaining the natural appearance of the image.

<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./misc/knob_gen_fancy.png" alt="Wide Image" width="500" style="display: block;">
    <img src="misc/knobgen_results_weakness.png" alt="Narrow Image" width="250" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
  </div>
</p>


### Impact of the knob mechanism across varying sketch complexities

![KnobGen Spectrum](misc/knob_spectrum.png)
