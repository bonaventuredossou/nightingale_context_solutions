# [Identifying High-risk Breast Cancer Using Digital Pathology Images](https://app.nightingalescience.org/contests/3jmp2y128nxd)
    - Solution of [Bonaventure F. P. Dossou](https://bonaventuredossou.github.io/)
**Disclaimer: All pretrained-models I used for the final solution are open-source, and pretrained on open-source datasets (ImageNet V1 and ImageNet V2). If you are using this repository, make sure you created the necessary folders (most would be auto-created for you though)**

# Install required libraries with `pip intall -r requirements.txt`

# Data Creation

My approach is based on the usage of pretrained vision models. Given an input `x`, and a model `M`, 
the goal is to predict `p(y|x)`, where `y = {0, 1, 2, 3, 4}`: this is a classification problem. Hence I had to extract slides from biopsies, and attribute to each of them the corresponding class. For classes that have not been staged, I have given the class 1, as EDA showed that the majority of staged biopsies are from stage 1.

The resulting dataset is the located at `project/breast_cancer` with two folders: `train` and `val`, each of which contains folders named after the class of images (i.e. 0, 1, ..., 4). This was do follow the structure required by the class `datasets.ImageFolder` from `torch` library. It is important to specify that images have been downsampled to the size (224, 224). `train` is `80%` of total images, and `val` contains the remaining `20%`.

The notebook to create the dataset is [here](project/DataCreation.ipynb), and the runtime was ~3-4 days to complete. 

# Modeling Approach

For the modeling, I have first of tried many vision models, 10 of them to be specific: `[resnetX]`(https://pytorch.org/vision/main/models/resnet.html) where `X={18, 50, 152}`, `[efficientnetV2-m]`(https://pytorch.org/vision/stable/models/efficientnetv2.html), `[ConvNext-base]`(https://pytorch.org/vision/stable/models/convnext.html), `[wide-resnet101_v2]`(https://pytorch.org/vision/stable/models/wide_resnet.html), `[vgg19_bn]`(https://pytorch.org/vision/stable/models/vgg.html), `[regnet_x_32gf]`(https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_x_32gf.html), `[swin_b]`(https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_b.html), 
and `[maxvit]`(https://pytorch.org/vision/stable/models/maxvit.html). They are all high-performing computer vision models. I chose those models size specially so that I could cope with the available RAM memory available.

I finetuned `entirely` each model (instead of just updating the reshaped layer which in most cases is the classifier layer)
I experimented every model with various learning rates, with the `AdamW` optimizer. The performances are summarized in the table below (for 50 training epochs, and `batch_size` of 32 - after experiments with different values, that was the optimal one):

Model | resnet18 | resnet50 | resnet152 | efficientnet_m | convnext_base | wide_resnet101_v2 | vgg19_bn | regnet_x_32gf | swin_b | maxvit |
|:---: |:---: |:---: | :---: |:---: | :---: | :---: | :---: | :---: |
`lr = 1e-4` | 1.009611 | **0.97050** | 1.005862| 0.9258 | 1.0099 | **0.9312** | **0.939045** | 0.992109 | 0.898370 | **0.855656** |
`lr = 1e-5` | **1.001620** | 1.008508 | **1.00190** | 0.9879 | **0.9574** | 1.012101 | 0.994687 | **0.988756** | **0.897878** | 0.893357 |
`lr = 4e-4` | 1.020936 | 0.986704 | 1.007281 | **0.8297** | 1.1101 | **1.0339** | 1.107765 | 1.000450 | 1.231371 | 1.098370 |

The best learning rate for each model are set in **bold*. My solution is a `Deep Ensemble` model (combination of several models). The folder `project/best_final_weights` contains the best weights of each model, respectively with hyperparameters explained above. In the folder `project/best_predictions` there are all submitted files, including the [best-scoring](project/best_predictions/final_predictions_deep_ensemble_32_50_with_AdamW_28.csv) on the leaderboard. The best scoring file is **only** combining the average predicitions of models which have losses < 1. The folder `project/pretrained_weights` contains initial pretrained weights (before adaptation to our problem).

The modeling notebook can be found [here](project/Modeling.ipynb)