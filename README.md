# ObjectDetection
1. Implementar SSD base con COCO -> https://github.com/amdegroot/ssd.pytorch
  - We remove all the dropout layers and the fc8 layer. We fine-tune the resulting model using SGD with initial learning rate 10−3, 0.9 momentum, 0.0005 weight decay, and batch size 32
  - Data augmentation is crucial: To make the model more robust to various input object sizes and shapes, each training image is randomly sampled by one of the following options:
– Use the entire original input image.
– Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3,
0.5, 0.7, or 0.9.
– Randomly sample a patch.
The size of each sampled patch is [0.1, 1] of the original image size, and the aspect ratio is between 12 and 2. We keep the overlapped part of the ground truth box if the center of it is in the sampled patch. After the aforementioned sampling step, each sampled patch is resized to fixed size and is horizontally flipped with probability of 0.5, in addition to applying some photo-metric distortions similar to those described in [14]
  - Atrous is faster. As described in Sec. 3, we used the atrous version of a subsampled VGG16, following DeepLab-LargeFOV [17]. If we use the full VGG16, keeping pool5 with 2 × 2 − s2 and not subsampling parameters from fc6 and fc7, and add conv5 3 for prediction, the result is about the same while the speed is about 20% slower
2. Añadir mejoras recientes al SSD para las que sirva COCO
  - Nope esto entonces: https://github.com/Capino512/pytorch-rotation-decoupled-detector
  - Mejora: https://www.nature.com/articles/s41598-022-16208-0
