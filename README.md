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

```
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

# Usage

The provided script is used for image-to-image translation using the Pro-UGATIT model. It takes images from the specified input directory, performs translation using the trained model, and saves the resulting images to the specified output directory. The script can be run from the command line with various options to customize the translation process. Below is a detailed explanation of the script and its command-line arguments.

## Command-Line Arguments

- `--resume`: The path to the pre-trained model weights. Default is "weights/anime/train_latest.pt".
- `--input`: The directory containing the input images for translation. Default is "test_inputs".
- `--saved-dir`: The directory to save the translated output images. Default is "test_outputs".
- `--dataset`: The dataset name for the model. Default is "gyate".
- `--align`: A flag that indicates whether to perform face alignment. Currently unused in the provided script.
- `--anime`: A flag that indicates whether the dataset is related to anime. Default is True.

## Running the Script

To run the image-to-image translation script, execute the following command from the terminal:

```bash
python script_name.py --type MODEL_TYPE --resume WEIGHTS_PATH --input INPUT_DIR --saved-dir OUTPUT_DIR --dataset DATASET_NAME --align --anime
```

- Replace `script_name.py` with the actual name of the Python script containing the provided code.
- `MODEL_TYPE`: Specify the type of model to use for translation. For Pro-UGATIT, use "ugatit".
- `WEIGHTS_PATH`: Provide the path to the pre-trained model weights.
- `INPUT_DIR`: Set the directory containing the input images for translation.
- `OUTPUT_DIR`: Set the directory to save the translated output images.
- `DATASET_NAME`: Specify the dataset name for the model. Default is "gyate".
- Use the `--align` flag to enable face alignment (currently unused).
- Use the `--anime` flag to indicate whether the dataset is related to anime (default is True).

# Data Sources for Pro-UGATIT

This document provides information about the data sources used to train Pro-UGATIT, a progressive unsupervised generative attentional network with adaptive layer instance normalization for image-to-image translation.

## 1. FFHQ Dataset

(https://github.com/NVlabs/ffhq-dataset)

The FFHQ (Flickr-Faces-HQ) dataset is a high-quality dataset of human faces collected from Flickr. It contains over 70,000 high-resolution images of human faces with a resolution of 1024x1024 pixels. The images cover a diverse range of identities, ages, and facial expressions, making it an ideal dataset for training models like Pro-UGATIT.

## 2. Danbooru2021
(https://gwern.net/danbooru2021)

Danbooru2021 is an image dataset collected from the Danbooru imageboard, which contains a large collection of anime and manga artwork. The dataset includes a wide variety of anime-style illustrations and character designs. It is used in conjunction with the FFHQ dataset to create a hybrid dataset for anime-to-real face image translation.

## 3. Anime2Coser Dataset

The Anime2Coser dataset is created by combining data from the FFHQ dataset and Danbooru2021. The following steps are taken to construct this dataset:

1. Real Face Data: Crop the high-resolution human face images from the FFHQ dataset to a resolution of 1024x1024 pixels. These images represent the "real" face domain.

2. Anime Face Data: Select not-monotone images from the Danbooru2021 dataset, which includes a diverse range of anime-style illustrations. Crop the face regions from these anime images to a resolution of 1024x1024 pixels.

By combining the real face data and anime face data, the Anime2Coser dataset is created to train Pro-UGATIT for image-to-image translation between anime-style faces and real human faces.

## Note

Pro-UGATIT leverages the diversity and high quality of the FFHQ dataset for real face images and the artistic and anime-style content from Danbooru2021 to achieve high-quality and realistic image-to-image translations.



