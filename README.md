# Image Dehazing

This project implements a **Pix2Pix**-based **Conditional GAN (cGAN)** for Image Dehazing. 
The generator follows a **U-Net** architecture, while the discriminator is an **aPatchGAN**, which helps in distinguishing between real and generated haze-free images.

## Sample Results


## Installation
```bash
git clone https://github.com/vc940/Image-Dehazing
cd pix2pix-image-dehazing
pip install -r requirements.txt
```
## Dataset
Reside[Standard] Dataset.
<br>
[Dataset Link](https://sites.google.com/view/reside-dehaze-datasets/reside-standard)

## Project Structure
```bash
.
├── Dataset
│   └── Dataloader.py
├── src
│   ├── Load_Data
│   │   └── Dataloader.py
│   ├── main.py
│   ├── Model
│   │   └── generator10.pth
│   ├── Patchgan
│   │   └── Discriminator.py
│   └── Unet
│       ├── __pycache__
│       │   └── unet_model.cpython-310.pyc
│       └── unet_model.py
└── Training_logs

```
## References  
- Isola, P., Zhu, J., Zhou, T., & Efros, A. A. (2017).  
  [Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)](https://arxiv.org/pdf/1611.07004).  
  *IEEE Conference on Computer Vision and Pattern Recognition (CVPR).*
## License  
This project is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.  

