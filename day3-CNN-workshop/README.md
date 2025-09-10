## 📘 Core Concept Questions

<details>
<summary><b>1. What advantages do CNNs have over traditional fully connected neural networks for image data?</b></summary>

* CNNs exploit the **spatial structure** of images, unlike fully connected networks that treat all pixels as independent features.
* They use **weight sharing** with convolutional filters, drastically reducing the number of parameters.
* This makes CNNs more efficient, less prone to overfitting, and better at capturing **local patterns** (edges, textures, shapes).

</details>

---

<details>
<summary><b>2. What is the role of convolutional filters/kernels in a CNN?</b></summary>

* Filters (kernels) are small matrices that **slide over the input image** to detect specific features such as edges, corners, or textures.
* Early layers learn low-level features (edges, gradients), while deeper layers learn higher-level features (object parts, shapes).
* They enable the CNN to **learn hierarchical feature representations**.

</details>

---

<details>
<summary><b>3. Why do we use pooling layers, and what is the difference between MaxPooling and AveragePooling?</b></summary>

* Pooling layers **reduce spatial dimensions** (downsampling), which:

  * Lowers computational cost.
  * Provides translation invariance (small shifts in input don’t affect recognition).
* **MaxPooling**: Takes the maximum value in the pooling window → captures the most prominent feature.
* **AveragePooling**: Takes the average value → smooths features but may lose sharp details.

</details>

---

<details>
<summary><b>4. Why is normalization of image pixels important before training?</b></summary>

* Ensures all features (pixels) are on a **similar scale**, improving training stability.
* Prevents exploding/vanishing gradients.
* Speeds up convergence by making optimization easier for gradient descent.
* Typical normalization: scaling pixel values to `[0, 1]` or standardizing with mean 0 and variance 1.

</details>

---

<details>
<summary><b>5. How does the softmax activation function work in multi-class classification?</b></summary>

* Softmax converts raw output logits into **probabilities** that sum to 1.
* Formula:

  $$
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
  $$
* The class with the highest probability is chosen as the prediction.
* Useful for **mutually exclusive classes** (only one correct label).

</details>

---
---

<details>
<summary><b>6. What strategies can help prevent overfitting in CNNs?</b></summary>

* **Dropout** → randomly disables neurons during training.
* **Data augmentation** → generates variations of training images (flips, rotations, crops).
* **Regularization (L2 weight decay)** → penalizes large weights.
* **Early stopping** → halts training when validation loss stops improving.
* **Batch normalization** → stabilizes learning and adds slight regularization.

</details>

---

<details>
<summary><b>7. What does the confusion matrix tell you about model performance?</b></summary>

* A confusion matrix shows **true labels vs. predicted labels**.
* Helps identify:

  * Correct predictions (diagonal values).
  * Misclassifications between specific classes.
  * Whether the model is biased toward certain classes.
* Useful for calculating **precision, recall, F1-score**, and spotting systematic errors.

</details>

---

<details>
<summary><b>8. If you wanted to improve the CNN, what architectural or data changes would you try?</b></summary>

* **Architectural changes**:

  * Add more convolutional layers to capture complex features.
  * Use **ResNet/Inception blocks** for deeper networks.
  * Add **batch normalization** for stability.
* **Data-related changes**:

  * Collect more diverse training data.
  * Apply stronger data augmentation.
  * Use **transfer learning** from pretrained models (e.g., VGG, ResNet).
* **Training tweaks**:

  * Tune learning rate with schedulers.
  * Use optimizers like Adam or RMSProp.
  * Experiment with different filter sizes and strides.

</details>

