"""Visualisation: an ASCII view (always available), a PNG, and an optional animated GIF.

The images show the path coloured by strategy, found clues, stations, the final belief, and an
entropy/battery timeline so changes of strategy are visible. Ground truth is drawn only as a
clearly labelled evaluation overlay.
"""
import numpy as np

STRATEGY_COLORS = {
    "localize": "#2b59c3", "search-clues": "#8a4fc7", "approach-eve": "#1f9d63",
    "return": "#e07a1f", "recharge": "#d4a017", "finish": "#6b7280", "random": "#9aa3b2",
}
SHADES = " .:-=+*#%@"


def _final_pos(res):
    return res.path[-1]


def ascii_map(res, reveal: bool = False) -> str:
    a, env = res.agent, res.env
    W, H = a.W, a.H
    g = [["?" if a.known[y, x] == -1 else ("#" if a.known[y, x] == 1 else " ") for x in range(W)] for y in range(H)]
    for x, y in res.path:
        if g[y][x] == " ":
            g[y][x] = "."
    for c in a.clues.values():
        g[c["pos"][1]][c["pos"][0]] = "c"
    for x, y in a.stations:
        g[y][x] = "S"
    if reveal:
        t = env.ground_truth()
        g[t["eve_pos"][1]][t["eve_pos"][0]] = "E"
    fx, fy = _final_pos(res)
    g[fy][fx] = "R"
    return "\n".join("".join(r) for r in g)


def ascii_belief(res) -> str:
    a = res.agent
    p = a.belief.p.reshape(a.H, a.W)
    top = p.max() or 1.0
    rows = []
    for y in range(a.H):
        rows.append("".join("#" if a.known[y, x] == 1 else SHADES[min(len(SHADES) - 1, int(p[y, x] / top * (len(SHADES) - 1)))]
                            for x in range(a.W)))
    return "\n".join(rows)


def ascii_report(res) -> str:
    return ("MAP  (R robot, S station, c clue found, E EVE [evaluation only], . path, # wall, ? unexplored)\n"
            + ascii_map(res, reveal=True)
            + "\n\nBELIEF at end of mission (darker = more probable, # = wall)\n" + ascii_belief(res) + "\n")


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _draw_map(ax, res, upto=None, belief=None):
    from matplotlib.collections import LineCollection

    a, env = res.agent, res.env
    t = env.ground_truth()
    ax.imshow(np.where(a.known == -1, 0.88, np.where(a.known == 1, 0.1, 1.0)), cmap="gray", vmin=0, vmax=1)
    if belief is not None:
        ax.imshow(np.ma.masked_less(belief / max(belief.max(), 1e-12), 0.02), cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    n = len(res.path) if upto is None else upto
    pts = np.array(res.path[:n], dtype=float)
    if len(pts) > 1:
        segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        cols = [STRATEGY_COLORS.get(res.strategies[i], "#999") for i in range(len(segs))]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2))
    for x, y in a.stations:
        ax.plot(x, y, "s", color="#d4a017", ms=9, mec="k")
    for c in a.clues.values():
        ax.plot(*c["pos"], "D", color="#8a4fc7", ms=7, mec="k")
    for d in t["decoys"]:
        ax.plot(*d, "x", color="#c0392b", ms=8)
    ax.plot(*t["eve_pos"], "*", color="#1f9d63", ms=15, mec="k")
    ax.plot(*res.path[max(0, n - 1)], "o", color="k", ms=6)
    ax.set_xticks([]); ax.set_yticks([])


def save_png(res, path):
    plt = _setup()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    a = res.agent
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    _draw_map(axes[0], res)
    axes[0].set_title("Path coloured by strategy")
    handles = [Line2D([0], [0], color=c, lw=3, label=k) for k, c in STRATEGY_COLORS.items()
               if k in set(res.strategies)]
    handles += [Line2D([0], [0], marker="s", color="w", mfc="#d4a017", mec="k", label="station"),
                Line2D([0], [0], marker="D", color="w", mfc="#8a4fc7", mec="k", label="clue found"),
                Line2D([0], [0], marker="*", color="w", mfc="#1f9d63", mec="k", ms=12, label="EVE (evaluation)"),
                Line2D([0], [0], marker="x", color="#c0392b", lw=0, label="decoy source (evaluation)")]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7, frameon=False)

    p = a.belief.p.reshape(a.H, a.W)
    axes[1].imshow(np.where(a.known == 1, np.nan, p), cmap="Reds")
    mp = a.belief.map_cell()
    axes[1].plot(*mp, "k+", ms=14, mew=2)
    if a.found:
        axes[1].plot(*a.pos, "o", mfc="none", mec="#1f9d63", ms=16, mew=2)
        axes[1].set_title(f"Belief before the successful inspection (+ best guess {mp}; green ring = where EVE was confirmed)",
                          fontsize=8)
    else:
        axes[1].set_title(f"Final belief (best guess {mp}, confidence {a.belief.mass_near(mp, 2):.0%})")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    ax = axes[2]
    steps = np.arange(len(res.entropy))
    for i, s in enumerate(res.strategies):
        ax.axvspan(i, i + 1, color=STRATEGY_COLORS.get(s, "#999"), alpha=0.12, lw=0)
    ax.plot(steps, res.entropy, color="#2b59c3", label="uncertainty (entropy, nats)")
    ax.set_xlabel("step"); ax.set_ylabel("entropy")
    ax2 = ax.twinx()
    ax2.plot(steps, res.battery, color="#e07a1f", label="battery")
    ax2.set_ylabel("battery")
    ax.set_title("Uncertainty and battery (shading = strategy)")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{res.agent.name} agent: {res.outcome}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_gif(res, path, max_frames: int = 60):
    plt = _setup()
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = len(res.path)
    frames = sorted(set(np.linspace(1, n, min(max_frames, n)).astype(int).tolist()))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    def draw(k):
        ax.clear()
        _draw_map(ax, res, upto=k, belief=res.snapshots[min(k - 1, len(res.snapshots) - 1)].astype(float))
        s = res.strategies[min(k - 1, len(res.strategies) - 1)] if res.strategies else ""
        ax.set_title(f"step {k - 1}: {s}")

    FuncAnimation(fig, draw, frames=frames).save(path, writer=PillowWriter(fps=6))
    plt.close(fig)
