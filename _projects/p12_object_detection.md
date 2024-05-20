---
title: "Object Detection using Vicreg and RetinaNet"
excerpt: "Using deep learning methods to pretrain and finetune on images for object detection"
last_modified_at: 2022-12-20
header:
  teaser: /assets/images/cv_dog.jpg
---

Example 1

![image1]({{site.url}}{{site.baseurl}}/assets/images/cv_sample
.png)


For this project we aim at carrying out an object detection task with variable sized input. We first researched on recent state-of-the-art methods, and then performed our downstream task using VICreg (Variance-Invariance-Covariance Regularization) (Bardes et al., 2021) to pretrain our ResNet backbone and RetinaNet (Lin et al., 2017) to finetune on labeled images. Our team achieved a mAP score of 0.125 on validation dataset and we identified several ways to improve upon our current approach as well as proposed novel methods that we should try out.

[My github project](https://github.com/cyberzzhhss/object_detect)


[My detailed report](https://github.com/cyberzzhhss/object_detect/blob/master/report.pdf)