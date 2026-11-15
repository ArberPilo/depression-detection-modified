"""Put the per-model figures side by side for the report and slides."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import RESULTS_DIR


def grid(paths, out, ncols):
    nrows = (len(paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    for ax in getattr(axes, "flat", [axes]):
        ax.axis("off")
    for ax, p in zip(axes.flat, paths):
        ax.imshow(mpimg.imread(p))
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


E = os.path.join(RESULTS_DIR, "experiments")
C = os.path.join(RESULTS_DIR, "cross_dataset")
names = ["Exp1_original_LSTM_CNN", "Exp2_modified_BiLSTM_CNN", "Exp3_GRU"]
grid([os.path.join(E, f"{n}_confusion.png") for n in names + ["Exp4_TFIDF_LogReg"]],
     os.path.join(E, "all_confusion_matrices.png"), 4)
grid([os.path.join(E, f"{n}_loss.png") for n in names], os.path.join(E, "all_loss_curves.png"), 3)
grid([os.path.join(C, f"confusion_{a}_to_{b}.png") for a, b in
      [("Twitter", "Twitter"), ("Twitter", "Reddit"), ("Reddit", "Reddit"), ("Reddit", "Twitter")]],
     os.path.join(C, "all_cross_confusion_matrices.png"), 4)
print("ok")
