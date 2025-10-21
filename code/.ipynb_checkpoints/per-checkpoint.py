import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

# ------------------------------
# Example: 2D synthetic data
# ------------------------------
rng = np.random.RandomState(0)
n_per_class = 20
X_pos = rng.randn(n_per_class, 2) + np.array([2.0, 2.0])
X_neg = rng.randn(n_per_class, 2) + np.array([-2.0, -2.0])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])

# shuffle dataset
perm = rng.permutation(len(y))
X = X[perm]
y = y[perm]

# augment with bias term
X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

# ------------------------------
# Perceptron training
# ------------------------------
def sign(v):
    return 1 if v >= 0 else -1

w = np.zeros(3)
eta = 1.0
max_epochs = 20

snapshots = []
step = 0

for epoch in range(1, max_epochs+1):
    errors = 0
    for i in range(len(y)):
        xi, yi = X_aug[i], y[i]
        y_pred = sign(np.dot(w, xi))
        if y_pred != yi:
            w = w + eta * yi * xi
            step += 1
            snapshots.append({
                "step": step,
                "epoch": epoch,
                "index": i,
                "x0": X[i,0],
                "x1": X[i,1],
                "y": yi,
                "w0": w[0],
                "w1": w[1],
                "b": w[2]
            })
            errors += 1
    if errors == 0:
        break

# ------------------------------
# Animation in Jupyter
# ------------------------------
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xs_line = np.linspace(x_min, x_max, 200)

def decision_boundary_points(wvec):
    w0, w1, b = wvec
    if abs(w1) > 1e-8:
        ys = -(w0 * xs_line + b) / w1
        return xs_line, ys
    elif abs(w0) > 1e-8:
        x_vert = -b / w0
        return np.array([x_vert, x_vert]), np.array([y_min, y_max])
    else:
        return np.array([]), np.array([])

fig, ax = plt.subplots(figsize=(6,6))
scatter_pos = ax.scatter(X[y==1,0], X[y==1,1], marker='o')
scatter_neg = ax.scatter(X[y==-1,0], X[y==-1,1], marker='s')
boundary_line, = ax.plot([], [])
updated_marker = ax.scatter([], [], marker='x', s=120)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

def init():
    boundary_line.set_data([], [])
    updated_marker.set_offsets(np.empty((0,2)))
    return boundary_line, updated_marker

def update(frame_idx):
    snap = snapshots[frame_idx]
    wvec = np.array([snap["w0"], snap["w1"], snap["b"]])
    xs, ys = decision_boundary_points(wvec)
    boundary_line.set_data(xs, ys)
    updated_marker.set_offsets([[snap["x0"], snap["x1"]]])
    ax.set_title(f"Step {snap['step']} Epoch {snap['epoch']} Updated idx {snap['index']}")
    return boundary_line, updated_marker

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=len(snapshots), interval=500, blit=False)

plt.close(fig)  # prevent duplicate static plot
display(HTML(anim.to_jshtml()))

