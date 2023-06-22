# Solution of Wombcare Team (Yenoukoume S. K. Gbenou, Bonaventure F. P. Dossou, Miglanche Ghomsi Nono) for the [AIM-AHEAD Health Equity Data Challenge 2023](https://app.nightingalescience.org/contests/8lo46ovm2g1j)

# Solution Approach

- For our modelling we tried the following:
    - computed the means and standard deviations across our datasets, that we ued for the transformation of our input images
    - We included the dice loss into the training of our network which provided some gains in the performance. The dice loss uses the [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) which is widely used in semantic segmentation tasks. The following [blog](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2) provides and intresting explanation of the dice coefficient. The dice loss is computed as follow: `dice_loss = 1 - dice_coefficient`
    - We built a deep ensemble model which learns joint representations from indvidual pretrained computer vision models
    - On top of the deep ensemble model, we trained each individual pretrained computer vision models. The idea is leverage what models learn seperately and what they learn together, and take advantage of it
    - The final predictions combine the product of the geometric and arithmetic means, of the predictions of the joint representation model, and of each individual model

You can find more details about both the [dataset creation](project/DataCreation.ipynb) and the [modeling](project/ModelingWombcare.ipynb).

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