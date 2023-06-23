# [Identifying High-risk Breast Cancer Using Digital Pathology Images Phase 1](https://app.nightingalescience.org/contests/3jmp2y128nxd)

**Solution of [Bonaventure F. P. Dossou](https://bonaventuredossou.github.io/)

Disclaimer: All pretrained-models I used for the final solution are open-source, and pretrained on open-source datasets (ImageNet V1 and ImageNet V2). If you are using this repository, make sure you created the necessary folders (most would be auto-created for you though)

**Install required libraries with `pip intall -r requirements.txt`

# Data Creation

My approach is based on the usage of pretrained vision models. Given an input `x`, and a model `M`, 
the goal is to predict `p(y|x)`, where `y = {0, 1, 2, 3, 4}`: this is a classification problem. Hence I had to extract slides from biopsies, and attribute to each of them the corresponding class. For classes that have not been staged, I have given the class 1, as EDA showed that the majority of staged biopsies are from stage 1.

The resulting dataset is the located at `project/breast_cancer` with two folders: `train` and `val`, each of which contains folders named after the class of images (i.e. 0, 1, ..., 4). This was do follow the structure required by the class `datasets.ImageFolder` from `torch` library. It is important to specify that images have been downsampled to the size (224, 224). `train` is `80%` of total images, and `val` contains the remaining `20%`.

The notebook to create the dataset is [here](project/DataCreation.ipynb), and the runtime was ~3-4 days to complete. 

# Modeling Approach

For the modeling, I have first of tried many vision models, 10 of them to be specific: [resnetX](https://pytorch.org/vision/main/models/resnet.html) where `X={18, 50, 152}`, [efficientnetV2-m](https://pytorch.org/vision/stable/models/efficientnetv2.html), [ConvNext-base](https://pytorch.org/vision/stable/models/convnext.html), [wide-resnet101_v2](https://pytorch.org/vision/stable/models/wide_resnet.html), [vgg19_bn](https://pytorch.org/vision/stable/models/vgg.html), [regnet_x_32gf](https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_x_32gf.html), [swin_b](https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_b.html), 
and [maxvit](https://pytorch.org/vision/stable/models/maxvit.html). They are all high-performing computer vision models. I chose those models size specially so that I could cope with the available RAM memory available.

I finetuned `entirely` each model (instead of just updating the reshaped layer which in most cases is the classifier layer)
I experimented every model with various learning rates, with the `AdamW` optimizer. The performances are summarized in the table below (for 50 training epochs, and `batch_size` of 32 - after experiments with different values, that was the optimal one):

Model | resnet18 | resnet50 | resnet152 | efficientnet_m | convnext_base | wide_resnet101_v2 | vgg19_bn | regnet_x_32gf | swin_b | maxvit |
|:---: |:---: |:---: | :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: |
`lr = 1e-4` | 1.009611 | **0.97050** | 1.005862| 0.9258 | 1.0099 | **0.9312** | **0.939045** | 0.992109 | 0.898370 | **0.855656** |
`lr = 1e-5` | **1.001620** | 1.008508 | **1.00190** | 0.9879 | **0.9574** | 1.012101 | 0.994687 | **0.988756** | **0.897878** | 0.893357 |
`lr = 4e-4` | 1.020936 | 0.986704 | 1.007281 | **0.8297** | 1.1101 | **1.0339** | 1.107765 | 1.000450 | 1.231371 | 1.098370 |

The best learning rate for each model are set in **bold*. My solution is a `Deep Ensemble` model (combination of several models). The folder `project/best_final_weights` contains the best weights of each model, respectively with hyperparameters explained above. The best scoring file is **only** combining the average predicitions of models which have losses < 1. The folder `project/pretrained_weights` contains initial pretrained weights (before adaptation to our problem).

The modeling notebook can be found [here](project/Modeling.ipynb)

# Update about the [Phase 2](https://app.nightingalescience.org/contests/vd8g98zv9w0p) of the context

# Best Solution Approach

My best solution for this phase of the context, is to directly evaluate on the adapted pretrained computer vision models trained and obtained from the Phase 1 of the context. Training on phase 1 and phase two datasets merged, did not perform well compared to the approach stated above. For more details about the modeling, please refer to my [phase 1 solution](https://github.com/bonaventuredossou/nightingale_winning_solution). The adapted modeling notebook can be found [here](project/Modeling_Phase_2.ipynb).


# 3 - Update about the [AIM-AHEAD Health Equity Data Challenge 2023](https://app.nightingalescience.org/contests/8lo46ovm2g1j)

# Wombcare Team Solution

- For our modeling we tried the following:
    - computed the means and standard deviations across the datasets, that we used for the transformation of our input images
    - We included the dice loss in the training of our network which provided some gains in the performance. The dice loss uses the [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) which is widely used in semantic segmentation tasks. The following [blog](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2) provides an interesting explanation of the dice coefficient. The dice loss is computed as follows: `dice_loss = 1 - dice_coefficient`
    - We built a deep ensemble model which learns joint representations from individual pretrained computer vision models
    - On top of the deep ensemble model, we trained each individual pretrained computer vision models. The idea is to leverage what models learn separately and what they learn together and take advantage of it
    - The final predictions combine the product of the geometric and arithmetic means, of the predictions of the joint representation model, and of each individual model

You can find more details about both the [dataset creation](project/WombcareDataCreation.ipynb) and the [modeling](project/ModelingWombcare.ipynb).

# Citations
Please cite these papers if you use this work:

- @inproceedings{dossou2023pretrained,
    title={Pretrained Vision Models for Predicting High-Risk Breast Cancer Stage},
    author={Bonaventure F. P. Dossou and Yenoukoume S. K. Gbenou and Miglanche Ghomsi Nono},
    booktitle={2023 ICLR First Workshop on Machine Learning {\&} Global Health},
    year={2023},
    url={https://arxiv.org/abs/2303.10730},
    eprint={2303.10730},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
} 

- @article{nightingale2022,
  author = {Mullainathan, Sendhil and Obermeyer, Ziad},
  title = {Solving medicine's data bottleneck: Nightingale Open Science},
  journal = {Nature Medicine},
  year = {2022},
  month = may,
  day = {01},
  volume = {28},
  number = {5},
  pages = {897-899},
  issn = {1546-170X},
  doi = {10.1038/s41591-022-01804-4},
  url = {https://doi.org/10.1038/s41591-022-01804-4}
}

- @dataset{brca-psj-path,
  author = {Bifulco, Carlo and Piening, Brian and Bower, Tucker and Robicsek, Ari and Weerasinghe, Roshanthi and Lee, Soohee and Foster, Nick and Juergens, Nathan and Risley, Josh and Nachimuthu, Senthil and Haynes, Katy and Obermeyer, Ziad},
  title = {Identifying high-risk breast cancer using digital pathology images},
  publisher = {Nightingale Open Science},
  year = {2021},
  doi = {10.48815/N5159B},
  url = {https://doi.org/10.48815/N5159B}
}