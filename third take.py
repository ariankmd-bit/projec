import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D

# 1) Load dataset
data = load_wine()
X = data.data # 13 features
y = data.target


# 2) Scale 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=3000, solver="lbfgs", multi_class="auto")
model.fit(X_scaled, y)

# =============================
# 3) Reduce to 3D for visualization

pca = PCA(n_components=3)
X_3d = pca.fit_transform(X_scaled)

# =============================
# 4) 3D Decision Surface


# Mesh grid for 3D
x_min, x_max = X_3d[:,0].min()-1, X_3d[:,0].max()+1
y_min, y_max = X_3d[:,1].min()-1, X_3d[:,1].max()+1
z_min, z_max = X_3d[:,2].min()-1, X_3d[:,2].max()+1

xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, 25),
    np.linspace(y_min, y_max, 25),
    np.linspace(z_min, z_max, 25)
)

grid_3d = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] # 3D PCA points
grid_13d = pca.inverse_transform(grid_3d) # Back to 13D
Z = model.predict(grid_13d) # Predict
Z = Z.reshape(xx.shape)

# 5) Plot in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Decision surface
ax.scatter(xx, yy, zz, c=Z, alpha=0.1, s=5, cmap="plasma")

# Real data points
sc = ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2],
                c=y, cmap="plasma", edgecolors="black", s=60)

# Labels
ax.set_title("Wine Classification (13 Features Model + 3D PCA Visualization)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")

# Legend
import matplotlib.patches as mpatches
class0 = mpatches.Patch(color=plt.cm.plasma(0.1), label=data.target_names[0])
class1 = mpatches.Patch(color=plt.cm.plasma(0.5), label=data.target_names[1])
class2 = mpatches.Patch(color=plt.cm.plasma(0.9), label=data.target_names[2])
plt.legend(handles=[class0, class1, class2], title="Wine Types")

plt.show()

# =============================
# 6) Print Accuracy
# =============================
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_scaled)
print("Model Accuracy:", accuracy_score(y, y_pred))