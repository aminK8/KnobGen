# KnobGen
## Controlling the Sophistication of Artwork in Sketch-Based Diffusion Models

KnobGen is a dual-pathway framework that empowers sketch-based image generation diffusion model by seamlessly adapting to varying levels of sketch complexity and user skill. KnobGen employs a Coarse-Grained Controller (CGC) module for leveraging high-level semantics from both textual and sketch inputs in the early stages of generation, and a Fine-Grained Controller (FGC) module for detailed refinement later in the process.

Paper Link: [KnobGen: Controlling the Sophistication of Artwork in Sketch-Based Diffusion Models](https://arxiv.org/abs/2410.01595)

![KnobGen Architecture](misc/HEDFusion.jpg)

## :rocket: News
- [2024-09-27] 🔥 Initial release of KnobGen code!
- [2024-10-02] 🔥 The paper is released on arXiv.

## Table of Contents
Follow steps 1-3 to run our pipeline. 
1. [Installation](#Installation)
2. Train
3. Evaluation

## Installation
To set up the environment, please follow these steps in the terminal:
```shell
git clone https://github.com/aminK8/KnobGen.git
cd KnobGen
conda env create -f environment.yml
conda activate knobgen
```

Prepare the Dataset
## Results

 Our method democratizes sketch-based image generation by effectively handling a broad spectrum of sketch complexity and user drawing ability—from novice sketches to those made by seasoned artists—while maintaining the natural appearance of the image.

![KnobGen Result](misc/knob_gen_fancy.png)

### KnobGen vs. baseline on novice sketches

![KnobGen VS Baselines](misc/knobgen_results_weakness.png)


### Impact of the knob mechanism across varying sketch complexities

![KnobGen Spectrum](misc/knob_spectrum.png)
