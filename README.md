# Projected Gradient Descent against ResNet50 on CIFAR 10 (Untargeted)
Projected Gradient Descent (Untargeted) against ResNet-50 on CIFAR-10 - Adversarial Machine Learning

Read about this project on my blog: [Part 3: Projected Gradient Descent (PGD)](https://sidthoviti.com/part-3-projected-gradient-descent-pgd)

This project tests the robustness of a [fine-tuned ResNet50](https://sidthoviti.com/fine-tuning-resnet50-pretrained-on-imagenet-for-cifar-10/) for CIFAR-10 against FGSM.

The pretrained ResNet50 model can be found [here](https://github.com/sidthoviti/FGSM-against-ResNet-50-on-CIFAR-10/blob/main/best_model.pth).

The attack success rate and robustness of the model were tested against various perturbation sizes.

![Comparison of Adversarial Accuracies and Attack Success Rates](https://raw.githubusercontent.com/sidthoviti/Projected-Gradient-Descent-against-ResNet50-on-CIFAR-10-Untargeted/aa6900f8f312cc964ac315a9dc5f11d19e0a5151/results/compare_adv%20acc%20and%20adv%20success%20rate_pgd.png)

* After training the model for 60 epochs, it achieved a clean test accuracy of 92.63%.
* The model is slightly overfitting.
* For epsilon = 0.01, the adversarial accuracy dropped to 35.84%, showing a significant number of successful attacks. With higher epsilon values, the adversarial accuracy decreased sharply, reaching just 1.24% at epsilon = 0.5.
* The attack success rate also rose with higher epsilon values, from 64.16% at epsilon = 0.01 to 98.76% at epsilon = 0.5.
* As the epsilon value increased, the adversarial accuracy decreased further, reaching 24.62% at an epsilon of 0.5.

![Comparison of FGSM and PGD](https://raw.githubusercontent.com/sidthoviti/Projected-Gradient-Descent-against-ResNet50-on-CIFAR-10-Untargeted/aa6900f8f312cc964ac315a9dc5f11d19e0a5151/results/pgd_fgsm_compare.png)

* Adversarial Accuracy: The adversarial accuracy for both FGSM and PGD decreases as the epsilon value increases, but the decline is steeper with PGD. For instance, at epsilon 0.01, FGSM has an adversarial accuracy of 47.02%, whereas PGD achieves only 35.84%. This trend continues with larger epsilon values, highlighting PGD's greater effectiveness in reducing the model’s accuracy. By epsilon 0.5, FGSM's adversarial accuracy is 24.60%, while PGD's is drastically lower at 1.23%.

* Attack Success Rate: The attack success rate for both FGSM and PGD increases with higher epsilon values, indicating that both attacks become more successful at fooling the model with stronger perturbations. However, PGD consistently achieves a higher attack success rate compared to FGSM. For example, at epsilon 0.01, FGSM’s success rate is 52.98%, while PGD’s is 64.16%. This gap widens with larger epsilon values, with FGSM achieving a success rate of 75.40% at epsilon 0.5, compared to PGD's 98.77%.
