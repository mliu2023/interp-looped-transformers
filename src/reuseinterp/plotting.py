"""Optional plotting of saved raw synthetic engineering fixtures only."""
from pathlib import Path
import os
import tempfile


def plot_fixtures(fixture_dir, output):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="reuse-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from reuseinterp.fixtures import recompute_fixtures
    metrics = recompute_fixtures(Path(fixture_dir))
    matplotlib.rcParams["svg.hashsalt"] = "reuseinterp-fixtures-v1"
    names = ["stable_latent_meaning", "positive_gain_offset", "sign_reversal", "changed_upstream_fixed_weights"]
    raw = [metrics["fixtures"][n]["raw_gap"] for n in names]
    cal = [float(metrics["fixtures"][n]["p1_gap"]["estimate"]) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - .18, raw, .36, label="Raw gap")
    ax.bar(x + .18, cal, .36, label="Positive-affine calibrated gap")
    ax.axhline(.05, color="black", ls="--", lw=.8, label="Fixture portability margin")
    ax.set_xticks(x)
    ax.set_xticklabels(["Stable meaning", "Gain / offset", "Sign reversal", "Changed upstream"])
    ax.set_ylabel("Bidirectional NMSE transfer gap")
    ax.set_title("SYNTHETIC SOFTWARE FIXTURES — NO LOOPVL RESULTS")
    ax.legend()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output), metadata={"Date": None})
