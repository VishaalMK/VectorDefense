# VectorDefense: Vectorization as a Defense to Adversarial Examples

VectorDefense is a model- and attack-agnostic transformation that impedes the effect of adversarial perturbations. The main idea is that VectorDefense transforms each input bitmap image into a *vector graphic* image (SVG format) - which is composed of simple geometric primitives (e.g oval and stroke) - via [Potrace](http://potrace.sourceforge.net/), and then rasterizes it back into the bitmap form before feeding it to the classifier.


![VectorDefense Concept](/concept.png)

VectorDefense can be viewed as a stepping stone towards decomposing images into compact, interpretable elements to solve the AX problem. For details, read our [paper](/link)

## Usage

The `.ipynb` files walk through
* Crafting adversarial examples (AXs) using Projected Gradient Descent (PGD) ([Madry et al. 2018](https://arxiv.org/abs/1706.06083))
* Purifying the AXs using the respective hand-designed transformations (viz. VectorDefense, Bit-depth reduction ([Xu et al. 2018](https://arxiv.org/abs/1704.01155)) and Image Quilting [Guo et al. (2018)](https://openreview.net/forum?id=SyJ7ClWCb))
* Subsequently performing white-box attack on these input transformations using Backward Pass Differentiable Approximation (BPDA) ([Athalye et al. 2018](https://arxiv.org/abs/1802.00420))

** VectorDefense**
1. Create the folders `adv_images/ `, `vectorize/` and `rasterize/` to allow VectorDefense transform the input AXs
1. `mnist_defend.sh` is the script that vectorizes and subsequently raterizes the input AXs




