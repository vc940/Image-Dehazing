# Image Dehazing

This project implements a **Pix2Pix**-based **Conditional GAN (cGAN)** for Image Dehazing. 
The generator follows a **U-Net** architecture, while the discriminator is an **aPatchGAN**, which helps in distinguishing between real and generated haze-free images.

## Sample Results
  * Results after training **10** epochs (2 hours) on 2  **T4 GPU's**
    <br>
  Randomly picked images from google
<p align="center">
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/train.jpg" width="300" height="200"/>
  >-------->-------->
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/train dehaze.jpg" width="300" height="200"/>
</p>

<p align="center">
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/hazyy.jpg" width="300" height="200"/>
  >-------->-------->
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/hazyy dehaze.jpg" width="300" height="200"/>
</p>

<p align="center">
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/city.jpeg" width="300" height="200"/>
  >-------->-------->
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/city dehaze.jpg" width="300" height="200"/>
</p>
<p align="center">
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/caption.jpg" width="300" height="500"/>
  >-------->-------->
  <img src="https://github.com/vc940/Image-Dehazing/blob/main/captiondehaze1.png" width="300" height="500"/>
</p>

*Note*: Trained only 10 epochs due to lack of computational resources,Obviously result's will improve if i'll train it more.

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

