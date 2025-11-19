import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1) Load dataset

data = load_wine()
X = data.data 
y = data.target


# 2) Scale 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=3000, solver="lbfgs", multi_class="auto")
model.fit(X_scaled, y)

# 3) Reduce to 2D for visualization

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)


# 4) Plot Decision Boundary based on TRUE 13-D model

x_min, x_max = X_2d[:,0].min() - 1, X_2d[:,0].max() + 1
y_min, y_max = X_2d[:,1].min() - 1, X_2d[:,1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

# Flatten grid → 2D → inverse PCA → scale → predict
grid_2d = np.c_[xx.ravel(), yy.ravel()] # 2D points
grid_13d = pca.inverse_transform(grid_2d) # back to 13D PCA space
grid_13d_scaled = grid_13d # already scaled before PCA
Z = model.predict(grid_13d_scaled)
Z = Z.reshape(xx.shape)

# 5) Plot background

plt.contourf(xx, yy, Z, alpha=0.35, cmap="plasma")


# 6) Plot 

plt.scatter(X_2d[:,0], X_2d[:,1], c=y, edgecolors="black", cmap="plasma")

plt.title("Wine Classification (13 Features Model + 2D PCA Visualization)", fontsize=14)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# 7) Legend

import matplotlib.patches as mpatches
class0 = mpatches.Patch(color=plt.cm.plasma(0.1), label=data.target_names[0])
class1 = mpatches.Patch(color=plt.cm.plasma(0.5), label=data.target_names[1])
class2 = mpatches.Patch(color=plt.cm.plasma(0.9), label=data.target_names[2])

plt.legend(handles=[class0, class1, class2], title="Wine Types")

plt.show()


# 8) Accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_scaled)
print("Model Accuracy:", accuracy_score(y, y_pred))