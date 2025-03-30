# DL4DS Midterm Challenge

This repository contains the code and results for my DL4DS midterm challenge. The challenge is divided into three parts:

- **Part 1:** A baseline model using a simple CNN trained from scratch on CIFAR-100.
- **Part 2:** More sophisticated CNN architectures (using only convolutional, pooling, and linear layers).
- **Part 3**: Transfer Learning and Fine-Tuning using a pretrained model.

Each Part has its own directory (Part1, Part2, and Part3) containing the respective code and a `submission_ood.csv` file generated as output by running the scripts.

---

## Table of Contents

1. [AI Disclosure](#ai-disclosure)  
2. [Model Description and Hyperparamter Tuning for Part 1 and 2](#model-description-and-hyperparamter-tuning-for-part-1-and-2)  
3. [Model Description and Hyperparamter Tuning for Part 3](#model-description-and-hyperparamter-tuning-for-part-3)  
4. [Regularization Techniques](#regularization-techniques)  
5. [Data Augmentation Strategy](#data-augmentation-strategy)  
6. [Results Analysis](#results-analysis)  
7. [Experiment Tracking Summary](#experiment-tracking-summary)  


---

## AI Disclosure

- **Tools Used:**  
I used ChatGPT as a supporting tool for understanding, exploring methods to improve accuracy and debugging models.

- **For Part 1:**
  - **My Contributions:**
    - I defined the network architecture from the provided starter code template. This simple CNN consists of two convolutional blocks and two fully connected layers.
    - I wrote the training and validation functions to ensure the complete pipeline from data loading to evaluation was functional.
    - I experimented with different data transformations to improve the model’s performance after the initial run.
  - **AI Assistance:**
    - I leveraged ChatGPT to understand various data augmentation techniques and get suggestions on how to structure the training pipeline.
    - ChatGPT helped me explore alternative transformations and provided ideas on how to tweak the code to improve the initial accuracy.

- **For Part 2:**
  - **My Contributions:**
    - I built upon the initial code to implement a more sophisticated CNN architecture and experimented with different models by integrating architectures like ResNet18 and DenseNet121.
    - After testing various architectures, I refined the ResNet18 model further by tuning hyperparameters (such as learning rate, batch size, and weight decay) to achieve higher accuracy than in Part 1.
  - **AI Assistance:**
    - ChatGPT was instrumental in explaining the differences between various architectures and suggesting ways to modify them for CIFAR-100.
    - I received guidance on how to implement training and validation routines for the more complex architectures.
    - AI suggestions helped me compare ResNet18 with DenseNet121, and ultimately, I opted for a refined ResNet18 approach.

- **For Part 3:**
  - **My Contributions:**
    - I implemented transfer learning using the ResNet18 model by fine-tuning the pretrained weights on the CIFAR-100 dataset.
    - I conducted extensive hyperparameter tuning and experimented with multiple fine-tuning strategies to not only surpass the benchmark score but also significantly improve performance and integrated advanced techniques to further boost the model’s performance.
  - **AI Assistance:**
    - ChatGPT provided detailed explanations of various fine-tuning strategies (e.g., freezing layers, using OneCycleLR, and MixUp augmentation) that I was less familiar with.
    - I used AI responses to clarify the implementation details of these techniques, and after some trial and error, I was able to successfully integrate them into my code.
    - AI acted as a valuable supporting tool, especially when I encountered confusion with the implementation of certain approaches; this allowed me to experiment with 4–5 different strategies before settling on the one that yielded the best results.

- **Detailed Code Comments:**
  - I have included comprehensive inline comments throughout the codebase (in the Part1, Part2, and Part3 directories) to explain every function and code block.
  - Some of these comments were refined with the help of ChatGPT to ensure clarity and uniformity.

*Note: All AI usage has been fully disclosed as required.*
<br>

---


## Model Description and Hyperparamter Tuning for Part 1 and 2

### Part 1: Simple CNN

The baseline model for Part 1 is a simple Convolutional Neural Network (CNN) built from scratch. The architecture is designed as follows:

- **Convolutional Block 1:**
  - **Conv1:** Converts the 3-channel RGB input to 32 feature maps using a 3×3 kernel with padding=1 (to maintain spatial dimensions).
  - **Conv2:** Increases the channel count from 32 to 64 using a 3×3 kernel.
  - **MaxPool:** A 2×2 pooling layer halves the spatial dimensions from 32×32 to 16×16.

- **Convolutional Block 2:**
  - **Conv3:** Converts 64 channels to 128 channels.
  - **Conv4:** Converts 128 channels to 256 channels.
  - **MaxPool:** Another 2×2 pooling layer reduces the spatial dimensions from 16×16 to 8×8.

- **Fully Connected Layers:**
  - The output of the convolutional blocks is flattened (256 channels × 8 × 8 = 16384 features).
  - **FC1:** Reduces 16384 features to 512.
  - **FC2:** Outputs 100 logits corresponding to the 100 classes in CIFAR-100.


The simple CNN architecture chosen for Part 1 serves as an effective baseline for the CIFAR-100 classification task. This design strikes a balance between simplicity and sufficient representational capacity. By employing two convolutional blocks, the network progressively captures low-level features in the early layers and higher-level abstractions in later layers, while the pooling operations efficiently reduce the spatial dimensions. This reduction not only decreases computational complexity but also helps in learning more robust features by aggregating local information. The fully connected layers then translate these learned features into class predictions, providing a clear and interpretable output for 100 classes. Overall, this architecture allows for rapid prototyping and debugging, making it an ideal starting point before exploring more complex architectures in later parts of the challenge.

For training this baseline model, the following hyperparameters were tuned:

- **Learning Rate:** Initial experiments were conducted with various learning rates. A value of **0.1** was found to be effective for convergence with the SGD optimizer.

- **Batch Size:** A batch size of **128** was selected based on GPU memory constraints and stable training behavior.

- **Number of Epochs:** The model was trained for `50 epochs`. Preliminary runs with fewer epochs (20–30) helped validate the pipeline before final training.

- **Optimizer:** **SGD** with a momentum of **0.9** and a weight decay of **5e-4** was used to update the model parameters.

- **Learning Rate Scheduler:** A **StepLR** scheduler was employed to reduce the learning rate by a factor of **0.1** every **10 epochs** to facilitate convergence over time.


<br>

### Part 2: More sophisticated models - ResNet18

For Part 2, I implemented a modified ResNet18 architecture tailored for CIFAR-100:

- **Input Adaptation:**  
  - The original 7×7 convolution (stride 2) is replaced with a 3×3 convolution (stride 1, padding 1) to better handle 32×32 images.
  - The aggressive max pooling layer is replaced with an identity mapping to preserve spatial dimensions.

- **Output Layer Modification:**  
  - The final fully connected layer is replaced to output 100 logits (one for each CIFAR-100 class).


To adapt ResNet18 for CIFAR-100’s 32×32 images, I modified the standard architecture to preserve spatial detail and improve performance on small images. The original ResNet18 starts with a 7×7 convolution with stride 2, which overly downsamples the input; thus, I replaced it with a 3×3 convolution (stride 1, padding 1) to maintain a richer spatial resolution. Additionally, I removed the initial max pooling layer by substituting it with an identity mapping, ensuring that early feature maps retain more information. The final fully connected layer was also replaced to output 100 logits, matching the number of classes in CIFAR-100. Although I experimented with DenseNet121 due to its efficient feature reuse, the modified ResNet18 struck a better balance between accuracy and computational efficiency, making it the preferred choice for this part.

Key hyperparameters were tuned as follows:

- **Learning Rate:** A base learning rate of **0.1** was used with SGD. A `CosineAnnealingLR` scheduler was employed to gradually reduce the learning rate over **50 epochs**.

- **Batch Size:** A batch size of **128** was chosen to balance computational efficiency and training stability.

- **Epochs:** The model was trained for `50 epochs` to allow sufficient convergence.

- **Optimizer Settings:** SGD with momentum of **0.9** and weight decay of **5e-4** was used to help stabilize training and mitigate overfitting.
<br>

---

## Model Description and Hyperparamter Tuning for Part 3

### Part 3: Transfer Learning with ResNet18

For Part 3, we use a ResNet18 model pretrained on ImageNet and adapt it for the CIFAR-100 dataset. The key modifications include:

- **Input Adaptation:**  
  - The original ResNet18 starts with a 7×7 convolution with stride 2, which is too aggressive for 32×32 images.
  - We replace this with a 3×3 convolution (stride=1, padding=1) to better preserve spatial details in CIFAR-100 images.
  - The initial max pooling layer is removed (replaced by an identity mapping) to avoid excessive downsampling.

- **Output Layer Modification:**  
  - The final fully connected layer is replaced to output 100 logits (one for each CIFAR-100 class).


For Part 3, I leveraged transfer learning by using a ResNet18 model pretrained on ImageNet. Pretraining on ImageNet allows the model to start with a rich set of learned features, which can be fine-tuned for the CIFAR-100 task with significantly less training time and improved performance compared to training from scratch. Since CIFAR-100 images are 32×32 in size, the original ResNet18 architecture—designed for 224×224 images—required modifications. Specifically, the initial 7×7 convolution with stride 2, which aggressively downsamples large images, was replaced with a 3×3 convolution (stride 1, padding 1) to better preserve spatial details in the smaller images. Additionally, the max pooling layer following this convolution was removed (replaced with an identity mapping) to avoid excessive downsampling. Finally, the final fully connected layer was replaced so that the model outputs 100 logits, aligning with the number of classes in CIFAR-100. This careful adaptation of a pretrained ResNet18 harnesses the power of transfer learning while tailoring the network architecture to the specific requirements of CIFAR-100, ultimately leading to faster convergence and improved performance.


### Hyperparameter Tuning
Key hyperparameters were tuned based on iterative experiments and tracking performance with wandb:

- **Optimizer & Learning Rate:**  
  - **Optimizer:** AdamW was used for fine-tuning.
  - **Base Learning Rate:** Set to **1e-4**.
  - **Scheduler:** OneCycleLR dynamically adjusts the learning rate over **50 epochs** to help with convergence.

- **Batch Size:**  
  - A batch size of **128** was selected to ensure efficient training given GPU memory constraints.

- **Training Duration:**  
  - The model was trained for **50 epochs**, providing enough iterations for the fine-tuning process without overfitting.

---

## Regularization Techniques

To enhance generalization and mitigate overfitting, we applied the following regularization methods in Part 3:

- **Label Smoothing:**  
  - Applied in the CrossEntropyLoss with a smoothing factor of **0.1** to prevent overconfident predictions.
  
- **MixUp Augmentation:**  
  - MixUp is applied during training (with an alpha parameter of **0.4**) to mix training examples, resulting in smoother decision boundaries and improved robustness.
<br>

---

## Data Augmentation Strategy

For Part 3, the training data is augmented to provide more varied inputs which helps in regularization:

- **Training Transformations:**  
  - **RandomCrop:** Randomly crops the image to 32×32 with 4 pixels of padding.
  - **RandomHorizontalFlip:** Randomly flips images horizontally.
  - **AutoAugment:** Uses the CIFAR10 policy to automatically apply a diverse set of augmentations.
  - **ColorJitter:** Randomly adjusts brightness, contrast, saturation, and hue.
  - **ToTensor & Normalize:** Converts images to tensors and normalizes them with a mean of 0.5 and standard deviation of 0.5 per channel.

- **Testing/Validation Transformations:**  
  - Only conversion to tensor and normalization are applied to ensure consistency during evaluation.

These augmentation techniques help the model learn robust features by exposing it to a variety of transformations during training.
<br>

---

## Results Analysis

### Part 1: Simple CNN

- **Overall Performance:**  
  - The baseline Simple CNN achieved a test accuracy of `0.29100` on CIFAR-100.

- **Observations:**  
  - The model converged quickly during training, confirming that the data pipeline and training loop were implemented correctly.
  - There is a noticeable gap between training accuracy and validation/test accuracy, indicating challenges in generalization.

- **Strengths:**  
  - **Simplicity:** The straightforward architecture makes it easy to understand and serves as a solid baseline.
  - **Fast Training:** Due to its limited depth and number of parameters, the model trains quickly, enabling rapid iteration and debugging.
  - **Pipeline Verification:** The model’s performance validates the overall pipeline—from data loading and augmentation to training and evaluation.

- **Weaknesses:**  
  - **Limited Representational Capacity:** The simple design might not capture the complex features necessary for CIFAR-100, leading to lower accuracy compared to more sophisticated architectures.
  - **Generalization:** The gap between training and validation/test accuracies suggests the model struggles with generalization, potentially due to overfitting.
  
- **Potential Improvements:**  
  - **Enhanced Architecture:** Exploring deeper or more complex models (e.g., ResNet variants) could improve performance.
  - **Regularization Techniques:** Incorporating methods such as dropout, MixUp, or CutMix may help reduce overfitting and boost generalization.
  - **Hyperparameter Refinement:** Further tuning of learning rate, batch size, and training duration might yield better performance.
<br>

### Part 2: ResNet18

- **Overall Performance:**  
  - The refined ResNet18 model achieved a test accuracy of `0.32719` on CIFAR-100, significantly improving upon the baseline from Part 1.

- **Observations:**  
  - The use of pretrained ResNet18 weights accelerated convergence and enhanced feature extraction.
  - Modifying the first convolution and removing the aggressive max pooling allowed the network to better adapt to 32×32 images.
  - Data augmentation techniques such as AutoAugment and ColorJitter contributed to improved generalization.
  - Despite these improvements, some overfitting is still evident, as seen in the gap between training and validation accuracies.

- **Strengths:**  
  - **Leveraging Pretrained Features:** Utilizing pretrained weights provided a strong initialization, resulting in faster convergence and higher accuracy.
  - **Architectural Adaptation:** Modifications to the input layers (3×3 convolution and removal of max pooling) effectively adapted the network for smaller images.
  - **Effective Data Augmentation:** Advanced augmentation strategies (e.g., AutoAugment) helped improve generalization and robustness.

- **Weaknesses:**  
  - **Hyperparameter Sensitivity:** The model’s performance is highly sensitive to learning rate and augmentation parameters, requiring careful tuning.
  - **Residual Overfitting:** Although less pronounced than in Part 1, some overfitting remains, suggesting room for additional regularization.

- **Potential Improvements:**  
  - Experiment with additional regularization methods such as MixUp or CutMix.
  - Further tune hyperparameters and explore alternative learning rate schedules.
  - Consider ensembling multiple refined models to further boost performance.
<br>

### Part 3: Transfer Learning with ResNet18

- **Overall Performance:**  
  - The fine-tuned ResNet18 model achieved a test accuracy of `0.57304` on CIFAR-100, representing a significant improvement over the baseline models.

- **Observations:**  
  - Fine-tuning a pretrained ResNet18 with advanced data augmentations (AutoAugment, ColorJitter, MixUp) enabled the model to leverage robust features from ImageNet and adapt effectively to CIFAR-100.
  - The modifications to the first convolutional layer (switching from a 7×7 to a 3×3 kernel with stride 1 and padding 1) and the removal of the max pooling layer helped preserve spatial information in the 32×32 images.
  - The **AdamW optimizer** proved to be highly effective during fine-tuning. Its decoupling of weight decay from the gradient update improved training stability and generalization.
  - The OneCycleLR learning rate scheduler dynamically adjusted the learning rate, facilitating smoother convergence over 50 epochs.
  - Despite strong performance, the model's performance remains sensitive to hyperparameter choices and data augmentation settings.

- **Strengths:**  
  - **Leveraging Pretrained Features:** Utilizing pretrained ResNet18 provided a strong initialization, accelerating convergence and boosting final accuracy.
  - **Effective Optimization:** The **AdamW optimizer** helped stabilize training and prevented overfitting by effectively decoupling weight decay from gradient updates.
  - **Adaptation for Small Images:** Modifications in the input layers allowed the model to better handle the smaller CIFAR-100 image size.
  - **Robust Data Augmentation:** Advanced augmentation strategies contributed to improved generalization and robustness.

- **Weaknesses:**  
  - **Hyperparameter Sensitivity:** The model’s performance is highly dependent on careful tuning of learning rate, augmentation parameters, and fine-tuning strategies.
  - **Overfitting Risk:** A slight gap between training and validation performance suggests that further regularization might be needed.

- **Potential Improvements:**  
  - Explore additional regularization techniques (e.g., CutMix or dropout) to further mitigate overfitting.
  - Experiment with discriminative learning rates across different layers or alternative learning rate schedules.
  - Consider ensembling multiple fine-tuned models to boost overall performance.
<br>

---

## Experiment Tracking Summary:

- All experiments were logged to Weights and Biases (wandb) for real-time monitoring of training and validation metrics.
  
![image](https://github.com/user-attachments/assets/bc33c769-119b-4e49-a0a0-24b5a698c8d8)
<br>


The screenshot above shows a set of training runs logged in the experiment tracking tool (Weights & Biases). Each line represents a different run configuration or hyperparameter setting, and the charts display how metrics such as training/validation loss, training/validation accuracy, and learning rate evolve over time. We can see that most models converge steadily, with validation accuracy curves leveling off near the end of training. The training accuracy tends to climb faster, indicating that the networks learn to fit the training data effectively, while the gap between training and validation metrics suggests varying degrees of overfitting among the runs. The learning rate curves illustrate how the scheduler adjusts over epochs, and differences in the final performance across runs highlight the importance of tuning hyperparameters (like learning rate, batch size, or augmentation strategies) for achieving optimal results.
<br>

![image](https://github.com/user-attachments/assets/f5205788-cb9f-428e-b0f4-5b50248587b0)
<br>


The screenshot above shows the training and validation metrics for the best-performing model over its entire training cycle. In particular, the val_loss curve consistently decreases, while val_acc rises and stabilizes at a high level, indicating effective learning and good generalization on the validation set. Simultaneously, the train_loss and train_acc charts reveal that the model converges smoothly on the training data without drastically overfitting, as evidenced by the validation metrics closely tracking the training curves. The lr plot demonstrates how the dynamic learning rate schedule (e.g., OneCycleLR) evolves over epochs, allowing the model to adapt its learning pace. Together, these metrics confirm that the chosen hyperparameters, data augmentation strategies, and architectural modifications synergized to produce a strong final performance on CIFAR-100.







