# VectorDefense: Vectorization as a Defense to Adversarial Examples

**Abstract:**
Training deep neural networks on images represented as grids of pixels has brought to light an interesting phenomenon known as adversarial examples. 
Inspired by how humans reconstruct abstract concepts, we attempt to codify the input bitmap image into a set of compact, interpretable elements to avoid being fooled by the adversarial structures.
We take the first step in this direction by experimenting with image vectorization as an input transformation step to map the adversarial examples back into the natural manifold of MNIST handwritten digits.
We compare our method vs. state-of-the-art input transformations and further discuss the trade-offs between a hand-designed and a learned transformation defense. For details, read our [paper](/link)

**:**
VectorDefense is a model- and attack-agnostic transformation that impedes the effect of adversarial perturbations. The main idea is that VectorDefense transforms each input bitmap image into a *vector graphic* image (SVG format) - which is composed of simple geometric primitives (e.g oval and stroke) - via [Potrace](http://potrace.sourceforge.net/), and then raterizes it back into the bitmap form before feeding it to the classifier.


![VectorDefense Concept](/concept.png)

VectorDefense can be viewed as a stepping stone towards decomposing images into compact, interpretable elements to solve the AX problem. For details, read our [paper](/link)

