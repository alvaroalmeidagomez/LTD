Sparse CT Reconstruction

This repository provides a numerical implementation of a methodology for sparse tomographic image reconstruction.

More precisely, the proposed reconstruction method is used as an initial step for solving a total variation (TV) minimization problem. The TV problem is implemented using the Chambolle algorithm.

Acknowledgments

This work was developed at the Centro de Modelamiento Matemático (CMM), Chile.

Repository Structure
______________________________________________________

***Data Files***
______________________________________________________

Abdomen.png – Test image of a human abdomen
Head.png – Test image of a human head
Phantom.jpg – Test image of the Shepp–Logan phantom
______________________________________________________

***Source Code***

______________________________________________________
Abdomen.m – MATLAB script for CT reconstruction of the human abdomen using total variation
Head.m – MATLAB script for CT reconstruction of the human head using total variation
Phantom.m – MATLAB script for CT reconstruction of the Shepp–Logan phantom using total variation
Total_variation_CT.m – Main function for total variation-based CT reconstruction
tv_denoise_chambolle.m – Implementation of the Chambolle algorithm for TV denoising
compute_div.m – Computes the divergence operator
compute_grad.m – Computes the gradient operator

