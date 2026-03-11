# SimpleDeformableCross Dataset

## Description
This is a synthetic toy dataset generated for evaluating the Latent Hierarchical Structural model.
The dataset consists of 12x12 grayscale images.
- **Positive class (`y=1`)**: Images containing a "Cross" pattern that undergoes small spatial deformations. The cross is composed of 5 distinct 4x4 parts (center, top, bottom, left, right).
- **Negative class (`y=-1`)**: Images containing random noise or randomly scattered 4x4 blocks that do not form the cross structure.

## Usage
The dataset is loaded directly in the `task_2_1.ipynb` and `task_2_2.ipynb` notebooks. Code in `task_2_1.ipynb` generates arrays `X_train, y_train, X_test, y_test` on the fly.
It contains 200 training samples (100 pos, 100 neg) and 100 testing samples.

## Why it is appropriate
The Latent Hierarchical Structural model proposed in the paper operates on a 1x1, 3x3, and 6x6 grid structure. Our 12x12 images map perfectly onto this. The positive class contains literal spatial deformations of its sub-parts, making it the perfect testbed for the shape deformation features $\Phi_S$ and dynamic programming inference steps described in the paper.
