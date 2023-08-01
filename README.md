# Pro-UGATIT
## Progressively Unsupervised Generative Attentional Networks with Adaptive Layer Instance Normalization for Image-to-Image Translation

Pro-UGATIT is a state-of-the-art framework aimed at providing advanced image-to-image translation functionality. It builds upon the original UGATIT implementation by [shoutOutYangJie](https://github.com/shoutOutYangJie/Morph-UGATIT), and aims to expand and refine the concept for various image translation tasks.

### Overview
The implementation encompasses various progressive techniques to achieve high-quality image translation through unsupervised learning. Adaptive Layer Instance Normalization (AdaLIN) plays a pivotal role in enhancing the capabilities of the model.

### Features
- **Progressive Learning:** Utilizes a staged learning approach to increase translation accuracy.
- **Unsupervised Approach:** Capable of translating images without the need for labeled data.
- **Adaptive Layer Normalization:** Enhances model flexibility and adaptability.

### Installation
You can clone the repository and install the necessary dependencies using the following commands:



```bash
git clone https://github.com/sam3u7858/pro-ugatit.git
cd Pro-UGATIT
pip install -r requirements.txt


### Training Pro-UGATIT

To train the Pro-UGATIT model with your preferred settings, use the following command:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train/train_ugatit.py --anime False --dataset getreal512og --report_image_freq 80 --direction BtoA --worker 32 --img_size 512
```

Explanation of the command-line options:

- `CUDA_VISIBLE_DEVICES`: Specify the GPU devices to use during training (e.g., "0,1,2,3,4,5,6,7" for multiple GPUs).
- `--anime False`: Set this flag to `False` to indicate that the dataset is not anime-related.
- `--dataset getreal512og`: Choose the dataset named `getreal512og` for training Pro-UGATIT.
- `--report_image_freq 80`: Determine the frequency (every 80 steps) at which intermediate images will be saved for monitoring the training progress.
- `--direction BtoA`: Define the translation direction, where BtoA means translating from domain B to domain A.
- `--worker 32`: Set the number of workers for data loading during training to 32.
- `--img_size 512`: Set the input image size for training to 512x512 pixels.

Make sure to have the required dependencies installed and access to the training dataset before executing the command.

