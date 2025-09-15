## 🔑 Core Concept Questions & Answers

### 1. Why is hyperparameter tuning important, and what trade-offs does it involve?

Hyperparameters (learning rate, batch size, optimizer, dropout rate, etc.) control how a model learns.

* **Importance**: Proper tuning improves convergence speed, accuracy, and generalization.
* **Trade-offs**:

  * Larger batch sizes = faster training but less generalization.
  * Smaller learning rate = more stable training but slower convergence.
  * Deeper networks = higher accuracy but more risk of overfitting.

---

### 2. How does model pruning or compression impact performance and resource usage?

* **Pruning** removes redundant neurons/weights.
* **Compression** (like quantization or TFLite conversion) reduces memory footprint and model size.
* **Impact**:

  * Faster inference, lower latency.
  * Reduced RAM/flash storage requirements.
  * Possible accuracy drop (trade-off).

---

### 3. Why is dropout effective in preventing overfitting?

Dropout randomly “turns off” neurons during training.

* Forces the network to learn **redundant, robust features**.
* Prevents over-reliance on specific neurons.
* Acts as a form of **regularization**, improving generalization to unseen data.

---

### 4. What challenges arise when deploying deep learning models in production?

* **Size/latency**: Large models don’t fit well on mobile/edge devices.
* **Scalability**: Serving predictions at scale with low latency is tough.
* **Hardware diversity**: Models may behave differently across CPU/GPU/TPU.
* **Monitoring & drift**: Data distribution shifts degrade performance over time.

---

### 5. How does TensorFlow Lite (or ONNX, TorchScript) help in deployment optimization?

* **TensorFlow Lite**: Optimized for mobile/IoT — reduces size, adds quantization, accelerates inference.
* **ONNX**: Cross-framework portability (PyTorch → TensorFlow → Caffe, etc.).
* **TorchScript**: Compiles PyTorch models into deployable graph format for C++/mobile runtimes.

---

### 6. What is the balance between model accuracy and efficiency in real-world applications?

* High accuracy may require **large models**, but they’re slow & resource-hungry.
* Efficient models may trade **some accuracy** for:

  * Lower latency (important in real-time apps like trading, self-driving).
  * Smaller memory footprint (needed for phones, IoT).
* The balance depends on **use case constraints** (e.g., medical diagnosis = prioritize accuracy, real-time AR app = prioritize speed).

---

### 7. How can hardware (GPU, TPU, Edge devices) influence optimization strategies?

* **GPU**: Good for parallel training/inference of large models.
* **TPU**: Specialized for large-scale matrix ops → faster training of deep models.
* **Edge devices**: Require pruning/quantization due to limited compute.
* **Strategy**: Choose optimization (quantization, distillation, pruning) based on target hardware.

---

### 8. Looking ahead, how might optimization differ for Transformer-based models compared to CNNs/RNNs?

* **Transformers** are compute-heavy (self-attention scales quadratically with sequence length).
* Optimizations include:

  * **Sparse attention** (reduce quadratic complexity).
  * **Knowledge distillation** (smaller “student” transformers).
  * **Quantization/low-rank factorization** of attention weights.
* Unlike CNNs (spatial locality) or RNNs (sequential), Transformers rely on **global context**, so optimization often focuses on reducing **attention complexity**.

---
