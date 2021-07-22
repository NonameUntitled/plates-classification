# Plates Classification Task

This is a kaggle competition solution. 

Competition is [here](https://www.kaggle.com/c/platesv2/overview).

My results: Accuracy (competition metric): ~0.93

Leaderboard position: 364/1423

### How was it achieved?

1) Background subtraction (GrabCut algorithm)
2) Transfer learning (ResNet152 as a CNN backbone)
3) Basic augmentations (flip, rotate, jitter)

### How can it be improved?

1) Albumentations library
2) Self-supervised learning
3) Early stopping technique
4) ...
