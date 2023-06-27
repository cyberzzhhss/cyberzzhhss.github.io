---
title: "Photorealistic Style Transfer"
excerpt: "Using whitening and coloring transformation (WCT) for photorealistic style transfer"
last_modified_at: 2023-4-29
header:
  teaser: /assets/images/cv_nst2.png
---

Example 1

![image1]({{site.url}}{{site.baseurl}}/assets/images/cv_nst1.png)

![image2]({{site.url}}{{site.baseurl}}/assets/images/cv_nst2.png)


We focused on a variant of Whitening and Coloring Transforms (WCT) called PhotoWCT, which specializes in photorealistic style transfer. We experimented with how training on different model architectures and layers affects the preservation of the original content structure in the stylized image. We used the Structural Similarity Index Measure (SSIM) as their main qualitative metric. We also dicussed different training and inference paradigms, as well as smoothing techniques, and presents visual and quantitative results of their experiments.

[My github project](https://github.com/cyberzzhhss/neuralStyleTransfer)


[My detailed report](https://github.com/cyberzzhhss/neuralStyleTransfer/blob/main/report.pdf)