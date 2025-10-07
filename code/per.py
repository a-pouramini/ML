# Perceptron step-by-step demo with plots and an animation
# - 2D synthetic, linearly separable data
# - Labels y in {-1, +1}
# - Perceptron update: w <- w + eta * y * x (bias included as an extra weight)
#
# The script trains the perceptron, records every weight update (snapshot),
# then creates:
#  1) an animation showing how the decision boundary evolves after each update,
#  2) a plot of misclassifications per epoch,
#  3) a final plot of the converged boundary,
#  4) a small table summarizing the first few updates.
#
# Notes on plotting: uses matplotlib (no explicit color settings).
# This cell will save a GIF to /mnt/data/perceptron_training.gif and display it.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import os
from IPython.display import Image, display

# optional helper available in this environment; fallback if not present
try:
    from caas_jupyter_tools import display_dataframe_to_user
    have_display_df = True
except Exception:
    have_display_df = False

# ------------------------------
# Data generation (2D, separable)
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

# augment with bias term (x_3 = 1)
X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

# ------------------------------
# Perceptron training
# ------------------------------
def sign(v):
    return 1 if v >= 0 else -1

w = np.zeros(3)          # [w0, w1, bias]
eta = 1.0
max_epochs = 20

snapshots = []  # list of dicts containing step, epoch, index, x, y, w
miscounts = []  # misclassification count per epoch
step = 0
converged_epoch = None

for epoch in range(1, max_epochs + 1):
    errors = 0
    # iterate in fixed order for deterministic behavior
    for i in range(len(y)):
        xi = X_aug[i]
        yi = y[i]
        y_pred = sign(np.dot(w, xi))
        if y_pred != yi:
            # perform update and record snapshot
            w = w + eta * yi * xi
            step += 1
            snapshots.append({
                "step": step,
                "epoch": epoch,
                "index": int(i),
                "x0": float(X[i, 0]),
                "x1": float(X[i, 1]),
                "y": int(yi),
                "w0": float(w[0]),
                "w1": float(w[1]),
                "b": float(w[2]),
            })
            errors += 1
    miscounts.append(errors)
    if errors == 0:
        converged_epoch = epoch
        break

# If we never updated (rare), still want one snapshot showing initial state
if len(snapshots) == 0:
    snapshots.append({
        "step": 0, "epoch": 0, "index": -1,
        "x0": float(np.nan), "x1": float(np.nan), "y": 0,
        "w0": float(w[0]), "w1": float(w[1]), "b": float(w[2])
    })

# ------------------------------
# Prepare frames for animation
# ------------------------------
total_updates = len(snapshots)
cap_frames = 60  # cap number of frames in animation to keep runtime reasonable
if total_updates <= cap_frames:
    sel_idxs = list(range(total_updates))
else:
    sel_idxs = np.linspace(0, total_updates - 1, cap_frames, dtype=int).tolist()

selected_snapshots = [snapshots[i] for i in sel_idxs]

# plotting bounds
pad = 1.0
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
xs_for_line = np.linspace(x_min, x_max, 200)

def decision_boundary_points(wvec):
    # wvec is array-like [w0, w1, b] for w0*x + w1*y + b = 0
    w0, w1, b = wvec
    if abs(w1) > 1e-8:
        ys = -(w0 * xs_for_line + b) / w1
        return xs_for_line, ys
    elif abs(w0) > 1e-8:
        # vertical line at x = -b/w0 (constant)
        x_vert = -b / w0
        return np.array([x_vert, x_vert]), np.array([y_min, y_max])
    else:
        # degenerate (both near zero) -- return empty
        return np.array([]), np.array([])

# ------------------------------
# Create animation frames
# ------------------------------
fig_anim = plt.figure(figsize=(6, 6))
ax_anim = fig_anim.add_subplot(111)
# create a scatter of the dataset (we will redraw every frame)
scatter_pos = ax_anim.scatter(X[y == 1, 0], X[y == 1, 1], marker='o')
scatter_neg = ax_anim.scatter(X[y == -1, 0], X[y == -1, 1], marker='s')
boundary_line, = ax_anim.plot([], [])  # placeholder for boundary
updated_marker = ax_anim.scatter([], [], marker='x', s=120)  # updated point

ax_anim.set_xlim(x_min, x_max)
ax_anim.set_ylim(y_min, y_max)
ax_anim.set_xlabel("x0")
ax_anim.set_ylabel("x1")
ax_anim.set_title("Perceptron training animation")

def init():
    boundary_line.set_data([], [])
    updated_marker.set_offsets(np.empty((0, 2)))
    return boundary_line, updated_marker

def update(frame_idx):
    ax_anim.collections.clear()  # remove previous scatter layers (but not the line)
    # replot dataset (so legend and points stay consistent)
    ax_anim.scatter(X[y == 1, 0], X[y == 1, 1], marker='o')
    ax_anim.scatter(X[y == -1, 0], X[y == -1, 1], marker='s')
    snap = selected_snapshots[frame_idx]
    w_snapshot = np.array([snap["w0"], snap["w1"], snap["b"]], dtype=float)
    xs_line, ys_line = decision_boundary_points(w_snapshot)
    # update boundary
    if xs_line.size > 0:
        boundary_line.set_data(xs_line, ys_line)
    else:
        boundary_line.set_data([], [])
    # highlight the updated point (if valid index)
    idx = snap["index"]
    if idx >= 0:
        updated_marker = ax_anim.scatter([X[idx, 0]], [X[idx, 1]], marker='x', s=160)
    # annotate weights and step
    title = f"step {snap['step']}  epoch {snap['epoch']}  updated idx {snap['index']}\n"
    title += f"w = [{snap['w0']:.2f}, {snap['w1']:.2f}], b = {snap['b']:.2f}"
    ax_anim.set_title(title)
    return boundary_line, updated_marker

anim = animation.FuncAnimation(fig_anim, update, init_func=init,
                               frames=len(selected_snapshots), interval=500, blit=False)

# save animation to a GIF file
out_gif_path = "perceptron_training.gif"
try:
    anim.save(out_gif_path, writer='pillow', fps=2)
    saved_gif = True
except Exception as e:
    saved_gif = False
    print("Warning: could not save GIF animation:", e)

plt.close(fig_anim)  # close to avoid duplicate display later

# ------------------------------
# Plot misclassifications per epoch
# ------------------------------
fig_mis = plt.figure(figsize=(6, 3.5))
ax_mis = fig_mis.add_subplot(111)
ax_mis.plot(range(1, len(miscounts) + 1), miscounts, marker='o')
ax_mis.set_xlabel("Epoch")
ax_mis.set_ylabel("Number of updates (misclassified points)")
ax_mis.set_title("Perceptron updates per epoch (misclassifications)")
ax_mis.grid(True)
fig_mis_path = "perceptron_miscounts.png"
fig_mis.savefig(fig_mis_path)
plt.close(fig_mis)

# ------------------------------
# Final decision boundary plot
# ------------------------------
fig_final = plt.figure(figsize=(6, 6))
ax_final = fig_final.add_subplot(111)
ax_final.scatter(X[y == 1, 0], X[y == 1, 1], marker='o')
ax_final.scatter(X[y == -1, 0], X[y == -1, 1], marker='s')
xs_fin, ys_fin = decision_boundary_points(np.array([w[0], w[1], w[2]]))
if xs_fin.size > 0:
    ax_final.plot(xs_fin, ys_fin)
ax_final.set_xlim(x_min, x_max)
ax_final.set_ylim(y_min, y_max)
ax_final.set_xlabel("x0")
ax_final.set_ylabel("x1")
ax_final.set_title(f"Final perceptron boundary after training (epochs={len(miscounts)})")
fig_final_path = "perceptron_final.png"
fig_final.savefig(fig_final_path)
plt.close(fig_final)

# ------------------------------
# Summarize a few snapshots in a table
# ------------------------------
snap_df = pd.DataFrame(snapshots)
if "step" in snap_df.columns:
    # show only the first 10 snapshots to keep it concise
    snap_summary = snap_df[["step", "epoch", "index", "x0", "x1", "y", "w0", "w1", "b"]].head(10)
else:
    snap_summary = snap_df.head(10)

# display dataframe using helper if available, else print
if have_display_df:
    display_dataframe_to_user("Perceptron update snapshots (first 10)", snap_summary)
else:
    print("\nFirst few updates (snapshot summary):")
    display(snap_summary)

# ------------------------------
# Final metrics and outputs
# ------------------------------
final_preds = np.array([1 if np.dot(w, X_aug[i]) >= 0 else -1 for i in range(len(y))])
accuracy = (final_preds == y).mean()

print(f"\nTraining finished. Total updates (steps): {total_updates}")
print(f"Epochs run: {len(miscounts)} (converged epoch if any: {converged_epoch})")
print(f"Final weights: w0={w[0]:.4f}, w1={w[1]:.4f}, b={w[2]:.4f}")
print(f"Training accuracy: {accuracy*100:.2f}%")

# show outputs (GIF, miscounts plot, final boundary)
if saved_gif and os.path.exists(out_gif_path):
    print("\nAnimation saved to:", out_gif_path)
    display(Image(filename=out_gif_path))
else:
    print("\nAnimation could not be saved; showing the last snapshot as static image instead.")
    # draw the last selected snapshot as fallback
    last_snap = selected_snapshots[-1]
    fig_fb = plt.figure(figsize=(6,6))
    ax_fb = fig_fb.add_subplot(111)
    ax_fb.scatter(X[y == 1, 0], X[y == 1, 1], marker='o')
    ax_fb.scatter(X[y == -1, 0], X[y == -1, 1], marker='s')
    xs_line, ys_line = decision_boundary_points(np.array([last_snap["w0"], last_snap["w1"], last_snap["b"]]))
    if xs_line.size > 0:
        ax_fb.plot(xs_line, ys_line)
    ax_fb.set_title(f"Snapshot step {last_snap['step']} epoch {last_snap['epoch']}")
    display(fig_fb)
    plt.close(fig_fb)

# show miscounts and final boundary images saved to files
if os.path.exists(fig_mis_path):
    display(Image(filename=fig_mis_path))
if os.path.exists(fig_final_path):
    display(Image(filename=fig_final_path))

# provide file paths for download if desired
print("\nFiles created (in ):")
for p in [out_gif_path, fig_mis_path, fig_final_path]:
    if os.path.exists(p):
        print("  -", p)

