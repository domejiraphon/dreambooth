## 1. Install Requirements

To use this code, you'll need to install the required libraries. Here's an overview of the libraries:

- **Torch and torchvision**: These are deep learning frameworks required to run the model.

- **Transformers**: This library is for tokenization and obtaining text embeddings from the text prompt.

- **TensorBoard**: TensorBoard is for logging and monitoring the training progress.

- **Diffusers**: The Diffusers library is for a stable diffusion model without the need to write it from scratch.

To install the required libraries, run the following command:

```
pip install -r requirements.txt
```
## 2. Training instructions

To train the model, follow these steps:

Note that during the training process, we do not generate images on the fly due to the long training times, unlike the paper. Instead, we generate regularization images in advance, typically around 200-300 images. Once this is done, you can proceed with training. The code dose this step automatically. We also provide the example dataset in `./dataset`

The arguments for training:
   - `--pretrained_model_name_or_path`: Specify the pretrained Stable Diffusion model you want to use for your training.
   - `--instance_data_dir`: This argument should point to the folder containing your instance images. For example: `./datasets/training/dog`.
   - `--class_data_dir`: Similarly, provide the path to the folder containing your regularized images, e.g., `./datasets/regularization/dog`.
   - `--with_prior_preservation`: Enable this option to use the prior loss during training, which can be beneficial for certain tasks.
   - `--instance_prompt`: Define the instance prompt during training. It's important to note that we use "sks" as the rare token, although this may not be universally applicable.
   - `--class_prompt`: Specify the class prompt used during training.
   - `--output_dir`: Specify the folder to store our checkpoints/log. 

```
bash scripts/train.sh
```
You can view all available configuration options for training the model by running the following command or see the code:

```bash
python train_dreambooth.py --help
```
## 3. Inference

Once you have completed the training process, you can perform inference to generate images using the trained model. 
### 3.1. Inference Arguments:

- `--model_id`: Specify the path to your trained model obtained from the previous training step.
- `--prompt`: Set the prompt for generating the output image.
- `--output_path`: Define the path where the generated output image will be saved.

### 3.2. Running Inference:

To run inference using the trained model, you can use the following command:

```bash
bash scripts/inference.sh
```

For more information, please refer to 
https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
