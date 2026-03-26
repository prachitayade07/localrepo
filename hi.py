import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np

# Simulated values for visual explanation (replace with real data if available)
models = ['K-Means (URL)', 'Logistic Regression (SMS)']
accuracies = [85.2, 96.4]
# --- Bar Chart: Accuracy Comparison ---
plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracies, color=['steelblue', 'mediumseagreen'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.1, yval + 1, f"{yval}%", fontsize=10)
plt.tight_layout()
plt.show()

# --- Simulated Precision-Recall Curve ---
plt.figure(figsize=(6, 4))
# Simulated curves for visualization (replace with model data if available)
precision_kmeans = np.linspace(0.6, 0.85, 100)
recall_kmeans = np.linspace(0.8, 0.6, 100)
precision_logreg = np.linspace(0.9, 0.98, 100)
recall_logreg = np.linspace(0.9, 0.85, 100)

plt.plot(recall_kmeans, precision_kmeans, label='K-Means (URL)', color='orange')
plt.plot(recall_logreg, precision_logreg, label='Logistic Regression (SMS)', color='green')
plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
