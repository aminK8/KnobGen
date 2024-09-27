# KnobGen

KnobGen for Condition Diffusion Tasks ([Pouyan Navard*](https://www.linkedin.com/in/pouyan-boreshnavard/), [Amin Karimi Monsefi*](https://7amin.github.io/), [Mengxi Zhou](https://www.linkedin.com/in/mengxi-zhou-23a10b289/), [Wei-Lun (Harry) Chao](https://sites.google.com/view/wei-lun-harry-chao/home), [Alper Yilmaz](https://ceg.osu.edu/people/yilmaz.15), [Rajiv Ramnath](https://cse.osu.edu/people/ramnath.6))

\* These authors contributed equally to this work. 


KnobGen, a dual-pathway framework that democratizes sketch-based image generation by seamlessly adapting to varying levels of sketch complexity and user skill. KnobGen uses a Coarse-Grained Controller (CGC) module for high-level semantics in the early stages of generation and a Fine-Grained Controller (FGC) module for detailed refinement later in the process.


![KnobGen Architecture](misc/KnobGen.png)

# News

- [2024-09-27] 🔥 Initial release of KnobGen code!

# Installation
To set up the environment and start using KnobGen, follow these steps:


1. `conda env create -f environment.yml`
2. `conda activate knobgen`
