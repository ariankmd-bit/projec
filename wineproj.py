import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# =============================
# 1) Load Wine dataset
# =============================
data = load_wine()
X = data.data
y = data.target

# =============================
# 2) Reduce to 2D with PCA
# =============================
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# =============================
# 3) Train Model
# =============================
model = LogisticRegression(max_iter=2000)
model.fit(X_2d, y)

# =============================
# 4) Plot Decision Boundary
# =============================

# Create mesh grid
x_min, x_max = X_2d[:,0].min() - 1, X_2d[:,0].max() + 1
y_min, y_max = X_2d[:,1].min() - 1, X_2d[:,1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

# Predict grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision background
plt.contourf(xx, yy, Z, alpha=0.35, cmap="plasma")

# Plot real points
scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=y, edgecolors="black", cmap="plasma")

plt.title("Wine Dataset Classification with PCA + Logistic Regression", fontsize=14)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# =============================
# ✔ FIXED: Add Legend Manually
# =============================
import matplotlib.patches as mpatches

class0 = mpatches.Patch(color=plt.cm.plasma(0.1), label=data.target_names[0])
class1 = mpatches.Patch(color=plt.cm.plasma(0.5), label=data.target_names[1])
class2 = mpatches.Patch(color=plt.cm.plasma(0.9), label=data.target_names[2])

plt.legend(handles=[class0, class1, class2], title="Wine Types")

plt.show()