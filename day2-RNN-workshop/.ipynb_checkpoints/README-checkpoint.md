## **Core Concept Questions**

---

### **1. Benefit of using RNNs (or LSTMs) over feedforward networks for time-series data**

* **Feedforward networks** treat inputs independently → they don’t remember past information.
* **RNNs/LSTMs** maintain a **hidden state** that carries temporal information from previous time steps → making them naturally suited for sequential data.
* LSTMs specifically mitigate the **vanishing gradient problem** (which limits vanilla RNNs), enabling them to capture **long-term dependencies** in time series (e.g., trends spanning many days in stock prices).

---

### **2. Why sequence framing (input windows) is important**

* Time-series forecasting requires context: the **past N time steps** influence the future.
* Framing with input windows (e.g., last 30 days → predict tomorrow) provides the model with sequential context.
* Without framing, the model only sees isolated points and cannot learn temporal patterns.

---

### **3. Impact of feature scaling**

* RNNs/LSTMs use gradient-based optimization. If features have vastly different scales, training becomes unstable (gradients may explode or vanish).
* Scaling (e.g., MinMaxScaler, StandardScaler) ensures:

  * Faster convergence.
  * Stable training.
  * Features contribute equally to the learning process.

---

### **4. SimpleRNN vs LSTM for long-term dependencies**

* **SimpleRNN:**

  * Captures short-term dependencies.
  * Suffers from vanishing gradients → struggles with long sequences.
* **LSTM:**

  * Uses gates (input, forget, output) to selectively remember/forget information.
  * Can capture both short- and long-term dependencies.
  * More computationally expensive but significantly better for stock prediction tasks.

---

### **5. Regression metrics for stock price prediction**

* **MAE (Mean Absolute Error):**

  * Measures average absolute deviation from true values.
  * Easy to interpret (in same units as stock price).
* **RMSE (Root Mean Squared Error):**

  * Penalizes larger errors more than MAE.
  * Useful since large deviations (bad forecasts) matter more in finance.
* Typically **both are reported**:

  * MAE → interpretability.
  * RMSE → sensitivity to large errors.

---

### **6. How to assess overfitting**

* Monitor **train vs validation loss**: if training error keeps dropping but validation error rises → overfitting.
* Check **prediction performance on unseen test data**.
* Use techniques like dropout, early stopping, and regularization to mitigate it.

---

### **7. Extending the model to improve performance**

* **More features:** e.g., trading volume, moving averages, technical indicators, sentiment data.
* **Deeper networks:** stacked LSTMs or hybrid models (e.g., CNN-LSTM for pattern extraction).
* **Attention mechanisms:** focus on the most relevant time steps.
* **Ensemble methods:** combine multiple models to improve robustness.

---

### **8. Why shuffle (or not shuffle) sequential data**

* **Do not shuffle sequential time-series training data** — temporal order matters.
* Shuffling would destroy dependencies between time steps.
* Exception: within **mini-batches**, you might shuffle **different sequences** (not the steps within them) for more stable gradient updates.

---

### **9. Visualizing predictions vs actual values**

* Plot **time series line charts**: actual prices vs predicted prices over time.
* Scatter plot of **predicted vs actual values** to check linear correlation.
* Residual plots (errors vs time) to spot systematic biases.

---

### **10. Real-world challenges with RNNs for stock price prediction**

* **Noisy & non-stationary data:** stock prices have high volatility, regime shifts, and external influences (e.g., news, policy).
* **Data sparsity & anomalies:** sudden events (earnings, crashes) are hard to predict.
* **Overfitting risk:** models may fit past noise instead of generalizable patterns.
* **Latency & efficiency:** financial predictions often need real-time or low-latency inference.
* **Feature engineering:** external factors (macro data, sentiment, etc.) strongly influence prices but are difficult to integrate.