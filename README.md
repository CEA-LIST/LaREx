# LaREx
Official repo of LaREx: Latent Representation Entropy density for distribution shift detection UAI 2024 Paper.

![](https://github.com/CEA-LIST/LaREx/blob/main/object_detection/larex_demo.gif)

# Abstract

Distribution shift detection is paramount in safety-critical tasks that rely on Deep Neural Networks (DNNs). The detection task entails deriving a confidence score to assert whether a new input sample aligns with the training data distribution of the DNN model. While DNN predictive uncertainty offers an intuitive confidence measure, exploring uncertainty-based distribution shift detection with simple sample-based techniques has been relatively overlooked in recent years due to computational overhead and lower performance than plain post-hoc methods. This paper proposes using simple sample-based techniques for estimating uncertainty and employing the entropy density from intermediate representations to detect distribution shifts. We demonstrate the effectiveness of our method using standard benchmark datasets for out-of-distribution detection and across different common perception tasks with Convolutional Neural Network architectures. Our scope extends beyond classification, encompassing image-level distribution shift detection for object detection and semantic segmentation tasks. Our results show that our method's performance is comparable to existing State-of-the-Art methods while being computationally faster and lighter than other Bayesian approaches, affirming its practical utility.

# Use cases
Our method has been tested in three use cases: Image classificcation using ResNet18, Object detection using FasterRCNN, and Instance segmentation using DeeplabV3. Each one of them has its own usage.

## Classification
We make extensive use of the ResNet18 pytorch implementation. For this case, go to the folder `image_classification`, create your environment with python=3.7, then install the `requirements-classification.txt`.

The config file in `config/config.yaml` serves as the unique config for training, extracting latent space samples and analyzing the samples.

### Training
Specify your parameters in the configuration files in the `config` folder. Use the `train_classifier.py` script. By default this script logs to a local mlflow server. The best and last checkpoints will be saved in the folder`lightning_logs/`. 

### Extracting samples
Choose the ood datasets in the `configs/config.yaml` file. Specify your model checkpoint in this config file also, from the checkpoint obtained in the previous step. Choose the baselines to be calculated also in the config file. Run the `extract_mcd_samples.py` script. All extracted information will be saved in a folder called `Mcd_samples`.

### Analysis of results
Without modifying the `config.yaml` file, run the `analyze_mcd_samples.py`. The results will be automatically logged to an mlflow server. Specify your server uri in the script or comment to log to a local mlflow server. The results will also be saved to a csv file. 

## Object detection
This section builds on the work of [VOS: Virtual Outlier synthesis](https://github.com/deeplearning-wisc/vos). 

## Instance segmentation

# Citation

If you found any part of this code is useful in your research, please consider citing our paper:
> @article{arnez,
      title={Latent Representation Entropy Density for distribution shift detection}, 
      author={Arnez, Montoya, Radermacher, Terrier},
      journal={Uncertainty in Artificial Intelligence UAI},
      year={2024}
> }