# CFC: Two-Level World Model for Policy Training

Plan for this pull request: add [CFC](https://people.csail.mit.edu/frankzydou/projects/CFC/) to the library. No results
yet.

CFC trains a policy through two world models instead of one:

- the **environment world model** covers everything outside the character: the fluid in the paper, and in the same way
  soft bodies, shells and other media the character interacts with;
- the **character (agent) world model** covers the controlled body itself.

External force is the interface between the two levels: the environment world model predicts the forces on the
character, and the character world model predicts its next state from its state, its action and those forces. The
policy is trained by backpropagation through both.

## Plan

1. The character (agent) world model with external forces as input.
2. The environment world model, starting with fluid.
3. Policy training through the two-level world model.
4. A benchmark with results.

## Reference

Z. Dou, C. Peng, X. Lu, X. Ye, L. Fang, Y. Liu, W. Wang, C. Gan, L. Liu, T. Komura. *CFC: Simulating Character-Fluid
Coupling Using a Two-Level World Model.* ACM Transactions on Graphics 44(6), Article 199, 2025.
[DOI](https://doi.org/10.1145/3763318).
