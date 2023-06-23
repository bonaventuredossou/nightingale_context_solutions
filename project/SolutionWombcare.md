# Solution of Wombcare Team (Yenoukoume S. K. Gbenou, Bonaventure F. P. Dossou) for the [AIM-AHEAD Health Equity Data Challenge 2023](https://app.nightingalescience.org/contests/8lo46ovm2g1j)

# Solution Approach

- For our modeling we tried the following:
    - computed the means and standard deviations across the datasets, that we used for the transformation of our input images
    - We included the dice loss in the training of our network which provided some gains in the performance. The dice loss uses the [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) which is widely used in semantic segmentation tasks. The following [blog](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2) provides an interesting explanation of the dice coefficient. The dice loss is computed as follows: `dice_loss = 1 - dice_coefficient`
    - We built a deep ensemble model which learns joint representations from individual pretrained computer vision models
    - On top of the deep ensemble model, we trained each individual pretrained computer vision models. The idea is to leverage what models learn separately and what they learn together and take advantage of it
    - The final predictions combine the product of the geometric and arithmetic means, of the predictions of the joint representation model, and of each individual model

You can find more details about both the [dataset creation](project/WombcareDataCreation.ipynb) and the [modeling](project/ModelingWombcare.ipynb).