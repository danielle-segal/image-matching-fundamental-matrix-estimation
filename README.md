# Fundamental Matrix Estimation via Deep Image Matching (RoMa)

Estimating the fundamental matrix between pairs of images using [RoMa](https://github.com/Parskatt/RoMa)
(Robust Dense Feature Matching) for correspondence generation, followed by a two-stage RANSAC fit.

## Overview

Given a pair of uncalibrated images, this project estimates the **fundamental matrix** - the
3×3 matrix that encodes the epipolar geometry relating the two views. Rather than relying on
classical sparse keypoint detectors (e.g. SIFT/ORB), correspondences are generated with **RoMa**,
a dense deep feature matching model, and then filtered and refined into a fundamental matrix
estimate.

## Method

1. **Dense correspondence matching** - RoMa (`roma_outdoor`) generates dense pixel-to-pixel
   correspondences between an image pair, along with a per-match certainty score.
2. **Initial fundamental matrix estimation** - correspondences are passed to
   `cv2.findFundamentalMat` using `USAC_MAGSAC`, a robust RANSAC-family estimator
   (`ransacReprojThreshold=0.5`, `confidence=0.999999`, `maxIters=100000`).
3. **Refinement** - the inlier set from step 2 (when at least 8 inliers are found) is re-fit
   using the classical 8-point algorithm (`cv2.FM_8POINT`) to sharpen the estimate.
4. **Robust fallback** - any failure along the pipeline (missing/corrupt images, insufficient
   correspondences, a degenerate fit) falls back to the identity matrix, so a single bad pair
   never aborts a full run.
   

## Setup

```bash
pip install -r requirements.txt
```

> The script was developed and run on a Kaggle GPU environment (NVIDIA T4).

## Usage

Run python project.py from the project directory. It expects a dataset directory containing:

- `test.csv` - one row per image pair (`sample_id, batch_id, image_1_id, image_2_id`)
- `test_images/<batch_id>/<image_id>.jpg` - the corresponding images

The script writes a `submission.csv` with one estimated fundamental matrix per row, flattened
to a space-separated string.

## Acknowledgments

- [RoMa: Robust Dense Feature Matching](https://github.com/Parskatt/RoMa) (Edstedt et al.)
- OpenCV's `findFundamentalMat` (`USAC_MAGSAC`, 8-point algorithm)
