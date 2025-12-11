import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from scipy.stats import norm as normal_dist  # Avoid naming conflict
#from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

def plot_cov_ellipse(cov, mean, ax, n_std=2.0, **kwargs):
    """
    Plot a covariance ellipse centered at `mean` for a given 2x2 covariance matrix.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)

# --- Setup ---
NUM_DATAPOINTS = 1000
file_path = "params.npy"
data = np.load(file_path)  # shape [T, d+1] (last col is nlls)
true_param = data[0, :-1] * (2 * (data[0, -2] > 0) - 1)

"""
iso = IsolationForest(contamination=0.01, random_state=0)
labels = iso.fit_predict(data[1:,:])          # +1 = inlier, -1 = outlier
data = np.vstack((data[0,:], data[1:,:][labels == 1]))
"""

env = EllipticEnvelope(contamination=0.1)
labels = env.fit_predict(data[1:,:])
data = np.vstack((data[0,:], data[1:,:][labels == 1]))

nlls = data[1:, -1]

# Filter out possibly bad MLE fits
idx = np.where(np.isnan(nlls))
data = np.delete(data, idx[0]+1, axis=0)
nlls = np.delete(nlls, idx)
#idx = np.argsort(nlls)[:int(nlls.shape[0]*0.95)]
#data = data[idx+1,:]
print(np.max(nlls))
print(np.min(nlls))

# Preprocessing
data = data[:, :-1]
# Multiply by sign of last dimension to keep signs consistent
data = data * np.tile(2 * (data[:, -1] > 0) - 1, (data.shape[1], 1)).T
data = (data[1:, :] - true_param) * np.sqrt(NUM_DATAPOINTS)


T, d = data.shape
df = pd.DataFrame(data, columns=[f"Dim {i+1}" for i in range(d)])

# Load kernel matrix
K = np.load('kernel.npy')
Kinv = np.linalg.pinv(K) / 4
print("Kernel inverse:\n", Kinv)
print("Shape:", K.shape)

# --- Alpha mapping from nlls ---
#norm = Normalize(vmin=nlls.min(), vmax=nlls.max())
#alphas = 1.0 - norm(nlls)  # max alpha = 1.0, min = 0.0
likelihood = np.exp(-nlls)
norm = Normalize(vmin=likelihood.min(),
                   vmax=likelihood.max())
alphas = norm(likelihood)
print(alphas)


# --- Create custom pairplot ---
sns.set(style="whitegrid")
pairgrid = sns.PairGrid(df, diag_sharey=False, corner=True,
                        height = 4, aspect=1)

def custom_hist(x, edgecolor='k', bins=20, density=True, label=None,
                color=None):
    ax = plt.gca()

    # Find index of diagonal plot. Super hacky annoying matplotlib
    idx = None
    for k in range(len(pairgrid.axes)):
        if pairgrid.axes[k, k]._position.bounds == ax._position.bounds:
            idx = k
            break

    plt.hist(x, edgecolor=edgecolor, bins=bins, density=density, label=label,
             color=color)
    x_vals = np.linspace(x.min(), x.max(), 300)
    std_dev = np.sqrt(Kinv[idx, idx])

    pdf = normal_dist.pdf(x_vals, loc=0.0, scale=std_dev)
    ax.plot(x_vals, pdf, color='red', lw=2)
    ax.set_xlabel('')
    ax.set_ylabel('')

#pairgrid.map_diag(plt.hist, edgecolor="k", bins=20, density=True)
pairgrid.map_diag(custom_hist, edgecolor="k", bins=20, density=True)

# Scatter with alpha based on nlls
for i in range(d):
    for j in range(i+1):
        ax = pairgrid.axes[i, j]
        ax.set_xlabel('')
        ax.set_ylabel('')
        labx = ax.get_xticklabels()
        laby = ax.get_yticklabels()

        _ = plt.setp(laby, visible=False)
        _ = plt.setp(labx, visible=False)
        if i == j:
            continue
        _ = plt.setp(laby, visible=True)
        _ = plt.setp(labx, visible=True)
        x = df[f"Dim {j+1}"]
        y = df[f"Dim {i+1}"]
        for xi, yi, alpha in zip(x, y, alphas):
            ax.scatter(xi, yi, color='black', s=10, alpha=alpha, edgecolor='none')

        # Add ellipse from Kinv
        cov_2d = Kinv[[j, i]][:, [j, i]]
        mean_2d = [0, 0]
        plot_cov_ellipse(
            cov_2d, mean_2d, ax,
            n_std=2.0,
            edgecolor='red',
            facecolor='none',
            lw=2,
            zorder=5
        )

# --- Save ---
plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')

