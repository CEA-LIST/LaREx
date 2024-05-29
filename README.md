# LaREx
Official repo of LaREx: Latent Representation Entropy density for distribution shift detection UAI 2024 Paper.

![](https://github.com/CEA-LIST/LaREx/blob/main/object_detection/larex_demo.gif)

# Abstract

Distribution shift detection is paramount in safety-critical tasks that rely on Deep Neural Networks (DNNs). The detection task entails deriving a confidence score to assert whether a new input sample aligns with the training data distribution of the DNN model. While DNN predictive uncertainty offers an intuitive confidence measure, exploring uncertainty-based distribution shift detection with simple sample-based techniques has been relatively overlooked in recent years due to computational overhead and lower performance than plain post-hoc methods. This paper proposes using simple sample-based techniques for estimating uncertainty and employing the entropy density from intermediate representations to detect distribution shifts. We demonstrate the effectiveness of our method using standard benchmark datasets for out-of-distribution detection and across different common perception tasks with Convolutional Neural Network architectures. Our scope extends beyond classification, encompassing image-level distribution shift detection for object detection and semantic segmentation tasks. Our results show that our method's performance is comparable to existing State-of-the-Art methods while being computationally faster and lighter than other Bayesian approaches, affirming its practical utility.

# Use cases
Our method has been tested in three use cases: 
* Image classificcation using ResNet18, 
* Object detection using FasterRCNN, 
* Instance segmentation using DeeplabV3. 
 
Each one of them has its own working environment and usage, detailed below.

## Classification
We make extensive use of the ResNet18 pytorch implementation. For this case, go to the folder `image_classification`, create your environment with python=3.7, then install the `requirements-classification.txt`.

The config file in `config/config.yaml` serves as the unique config for training, extracting latent space samples and analyzing the samples.

### Training
1. Specify your parameters in the configuration files in the `config` folder. 
2. Use the `train_classifier.py` script. By default this script logs to a local mlflow server. The best and last checkpoints will be saved in the folder`lightning_logs`. 

### Extracting samples
1. Choose the ood datasets in the `configs/config.yaml` file. 
2. Specify your model checkpoint in this config file also, from the checkpoint obtained in the previous step. 
3. Choose the baselines to be calculated also in the config file. 
4. Run the `extract_mcd_samples.py` script. All extracted information will be saved in a folder called `Mcd_samples`.

### Analysis of results
Without modifying the `config.yaml` file, run the `analyze_mcd_samples.py`. The results will be automatically logged to an mlflow server. Specify your server uri in the script or comment to log to a local mlflow server. The results will also be saved to a csv file. 

## Object detection
This section builds on the work of [VOS: Virtual Outlier synthesis](https://github.com/deeplearning-wisc/vos). 
1. Refer to VOS installation process, get the BDD100k InD and OoD datasets, and checkpoints from their repository.
2. For the installation of the detectron2 library, one modification in the Region Proposal Network (RPN) was necessary (The addition of a Module Called MCDRpnHead in the `detectron2/modeling/proposal_generator/rpn.py` script). For convenience, the modified version fo the detectron2 library can be cloned from [here](https://github.com/danielm322/detectron2)


### Extracting samples
1. Specify your ood dataset in the `configs/Inference//mcd.yaml` parameter: `OOD_DATASET` to be either 'openimages_ood_val' for OpenImages OoD, or 'coco_ood_val_bdd' for OoD COCO 

2. run:
```python
python get_mcd_and_entropy_samples_bdd_ood.py 
--dataset-dir path/to/dataset/dir
--test-dataset bdd_custom_val 
--config-file BDD-Detection/faster-rcnn/vanilla.yaml 
--inference-config Inference/mcd.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
Extracted samples will be saved in the `MCD_evaluation_data` folder.

### Analyze results
1. Specify the name of your extracted files, and the OoD dataset to analyze in the `configs/MCD_evaluation/config_rcnn.yaml` file.
2. run `python mcd_analysis.py`. The results will be automatically logged to an mlflow server, and saved to a csv.

## Instance segmentation
1. Create an environment with python=3.7, and install the `requirements-segmentation.txt`.
2. Download the datasets [Woodscapes](https://github.com/valeoai/WoodScape) or [cityscapes](https://www.cityscapes-dataset.com/)
3. Train model running:
   ```python
   python train_deeplab_v3p.py -m deeplabv3p-backbone-dropblock2d --batch 16 --epochs 100 --loss_type focal_loss --dataset woodscape --datapath /your_path_to_dataset/WoodScape
   ```
4. Use notebooks for feature extraction, and results analysis


# Acknowledgments
This work has been supported by the French government under the "France 2030” program, as part of the SystemX Technological Research Institute within the [confiance.ai](https://www.confiance.ai/) Program.

This publication was made possible by the use of the FactoryIA supercomputer, financially supported by the Ile-de-France Regional Council.

# Citation

If you found any part of this code is useful in your research, please consider citing our paper:
> @article{arnez,
      title={Latent Representation Entropy Density for distribution shift detection}, 
      author={Arnez, Montoya, Radermacher, Terrier},
      journal={Uncertainty in Artificial Intelligence UAI},
      year={2024}
> }