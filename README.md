# VectorDefense: Vectorization as a Defense to Adversarial Examples

VectorDefense is a model- and attack-agnostic transformation that impedes the effect of adversarial perturbations. The main idea is that VectorDefense transforms each input bitmap image into a *vector graphic* image (SVG format) - which is composed of simple geometric primitives (e.g oval and stroke) - via [Potrace](http://potrace.sourceforge.net/), and then raterizes it back into the bitmap form before feeding it to the classifier.


![VectorDefense Concept](/concept.png)

VectorDefense can be viewed as a stepping stone towards decomposing images into compact, interpretable elements to solve the AX problem.

