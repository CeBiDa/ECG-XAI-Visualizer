import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

from ecgxai import theme
from ecgxai.segments import SEGMENTS

_BAR_ORDER = list(SEGMENTS) + ["Other"]


def _style_axes(ax, t, fs):
    ax.set_facecolor(theme.SURFACE2)
    for sp in ax.spines.values():
        sp.set_color(theme.BORDER)
        sp.set_linewidth(0.6)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.minorticks_on()
    if fs and len(t):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=11, steps=[1, 2, 5, 10]))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.ticklabel_format(axis="x", useOffset=False)
    ax.grid(which="major", color=theme.GRID_MAJOR, linewidth=0.6)
    ax.grid(which="minor", color=theme.GRID_MINOR, linewidth=0.4)
    ax.tick_params(colors=theme.MUTED, labelsize=7, length=2)
    ax.margins(x=0)


def placeholder(fig, text):
    fig.clf()
    fig.patch.set_facecolor(theme.BG)
    fig.text(0.5, 0.5, text, ha="center", va="center", color=theme.MUTED, fontsize=11)


def draw_leads(fig, signals, names, fs=None, visible=None, window=None, color=None, t0=0.0):
    fig.clf()
    fig.patch.set_facecolor(theme.BG)
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] == 0:
        placeholder(fig, "No signal to show")
        return
    names = [str(n) for n in names]
    shown = [i for i, n in enumerate(names) if visible is None or n in visible]
    if not shown:
        placeholder(fig, "No leads selected")
        return
    try:
        fs = float(fs) if fs else None
    except (TypeError, ValueError):
        fs = None
    n = sig.shape[1]
    t = (float(t0) + np.arange(n) / fs) if fs else np.arange(n, dtype=float)

    k = len(shown)
    cols = 1 if k <= 6 else 2
    rows = int(np.ceil(k / cols))
    axes = fig.subplots(rows, cols, sharex=True, squeeze=False)
    for j, i in enumerate(shown):
        ax = axes[j % rows][j // rows]
        if window and fs:
            ax.axvspan(window[0], window[1], color=theme.WINDOW, alpha=0.09, lw=0, zorder=0)
        ax.plot(t, sig[i], color=color or theme.ACCENT, linewidth=0.8)
        _style_axes(ax, t, fs)
        ax.ecg_time_axis = True          # the GUI zooms and pans these
        ax.text(0.012, 0.95, names[i], transform=ax.transAxes, color=theme.LEAD_LABEL,
                fontsize=8, fontweight="bold", va="top")
    for j in range(k, rows * cols):
        axes[j % rows][j // rows].set_visible(False)
    for c in range(cols):
        last = min(rows, k - c * rows) - 1
        if last >= 0:
            axes[last][c].xaxis.set_tick_params(labelbottom=True)
            axes[last][c].set_xlabel("seconds" if fs else "samples", color=theme.MUTED, fontsize=8)
    fig.subplots_adjust(left=0.055 if cols == 2 else 0.045, right=0.995, top=0.985,
                        bottom=0.075 if rows > 3 else 0.12, hspace=0.35, wspace=0.11)


def _runs(labels):
    lab = np.asarray(labels)
    if lab.size == 0:
        return []
    edges = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [lab.size]))
    return [(int(lab[a]), int(a), int(b)) for a, b in zip(starts, stops) if lab[a] >= 0]


def draw_xai(fig, signal, fs, attribution, labels=None, segment_result=None,
             title="", subtitle="", subtitle_color=None, top_percentile=90.0, t0=0.0):
    fig.clf()
    fig.patch.set_facecolor(theme.BG)
    sig = np.asarray(signal, dtype=np.float64)
    attr = np.asarray(attribution, dtype=np.float64)
    if sig.size == 0 or attr.shape != sig.shape:
        placeholder(fig, "Nothing to explain")
        return
    fs = float(fs)
    t = float(t0) + np.arange(sig.size) / fs

    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], height_ratios=[3, 1.15],
                          left=0.055, right=0.975, top=0.80, bottom=0.11,
                          wspace=0.07, hspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    axa = fig.add_subplot(gs[1, 0], sharex=ax)
    axb = fig.add_subplot(gs[:, 1])
    ax.ecg_time_axis = axa.ecg_time_axis = True

    if labels is not None:
        for seg, a, b in _runs(labels):
            name = SEGMENTS[seg]
            ax.axvspan(t[a], t[b - 1] + 1.0 / fs, color=theme.SEGMENT_COLORS[name],
                       alpha=0.17, lw=0, zorder=0)
    ax.plot(t, sig, color=theme.TRACE_XAI, linewidth=1.0, zorder=2)
    zero = float((segment_result or {}).get("zero") or 0.0)   # SmoothGrad x SIGN: ~0.5
    thr = float(np.percentile(attr, top_percentile))
    top = (attr >= thr) & (attr > zero)
    ax.scatter(t[top], sig[top], s=7, color=theme.XAI, alpha=0.9, linewidths=0, zorder=3)
    _style_axes(ax, t, fs)
    ax.tick_params(labelbottom=False)

    axa.fill_between(t, min(max(zero, 0.0), 1.0), attr, color=theme.XAI, alpha=0.28, linewidth=0)
    if 0.0 < zero < 1.0:
        axa.axhline(zero, color=theme.MUTED, linewidth=0.6, alpha=0.8)
    axa.plot(t, attr, color=theme.XAI, linewidth=0.9)
    axa.axhline(thr, color=theme.XAI, linestyle=(0, (3, 3)), linewidth=0.7, alpha=0.8)
    _style_axes(axa, t, fs)
    axa.set_ylim(0, 1.05)
    axa.yaxis.set_major_locator(MultipleLocator(0.5))
    axa.set_ylabel("attribution", color=theme.MUTED, fontsize=7.5)
    axa.set_xlabel("seconds", color=theme.MUTED, fontsize=8)

    shares = (segment_result or {}).get("shares")
    vals = [float(shares.get(nm, 0.0)) for nm in _BAR_ORDER] if shares else [0.0] * len(_BAR_ORDER)
    axb.set_facecolor(theme.SURFACE2)
    for sp in axb.spines.values():
        sp.set_color(theme.BORDER)
        sp.set_linewidth(0.6)
    ypos = np.arange(len(_BAR_ORDER)) * 1.0
    axb.barh(ypos, vals, height=0.42, color=[theme.SEGMENT_COLORS[nm] for nm in _BAR_ORDER])
    axb.set_yticks([])
    axb.set_ylim(len(_BAR_ORDER) - 0.45, -0.85)
    axb.set_xlim(0, 100)
    axb.xaxis.set_major_locator(MultipleLocator(25))
    axb.grid(axis="x", color=theme.GRID_MAJOR, linewidth=0.6)
    axb.set_axisbelow(True)
    axb.tick_params(colors=theme.MUTED, labelsize=7.5, length=2)
    axb.set_xlabel("% of top attribution", color=theme.MUTED, fontsize=8)
    axb.set_title(f"Top {100 - float(top_percentile):g} % attribution",
                  color=theme.MUTED, fontsize=8.5, loc="left", pad=6)
    for y, nm in zip(ypos, _BAR_ORDER):
        axb.text(1.5, y - 0.27, theme.SEGMENT_LABELS[nm], va="bottom", ha="left", fontsize=8,
                 color=theme.TEXT)
    if shares:
        for y, v in zip(ypos, vals):
            inside = v > 78
            axb.text(v - 1.5 if inside else v + 1.5, y, f"{v:.1f} %", va="center",
                     ha="right" if inside else "left", fontsize=8,
                     color=theme.BG if inside else theme.TEXT)
    else:
        note = (segment_result or {}).get("note") or "No segment analysis"
        axb.text(50, len(_BAR_ORDER) / 2 - 0.5, _wrap(note, 22), ha="center", va="center",
                 color=theme.MUTED, fontsize=8,
                 bbox=dict(facecolor=theme.SURFACE2, edgecolor="none", pad=3))

    if title:
        fig.text(0.055, 0.965, title, color=theme.TEXT, fontsize=11.5, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.055, 0.905, subtitle, color=subtitle_color or theme.MUTED, fontsize=9.5, va="top")
    handles = [Patch(facecolor=theme.SEGMENT_COLORS[nm], alpha=0.45, label=theme.SEGMENT_LABELS[nm])
               for nm in SEGMENTS]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=4, color=theme.XAI,
                          label=f"top {100 - float(top_percentile):g} % attribution"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.805), ncol=5,
               frameon=False, fontsize=7.5, labelcolor=theme.MUTED, handlelength=1.2,
               columnspacing=1.2, borderaxespad=0)


def _wrap(text, width):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def probability_text(p, threshold):
    d = 3
    while d < 8 and (round(p, d) >= threshold) != (p >= threshold):
        d += 1
    if (round(p, d) >= threshold) != (p >= threshold):
        return repr(float(p))
    return f"{p:.{d}f}"


def xai_titles(record_name, lead_label, method_name, prediction, target_label, positive):
    p, thr = prediction["probability"], prediction["threshold"]
    rel = "≥" if p >= thr else "<"
    name = str(record_name)
    if len(name) > 64:
        name = name[:30] + "…" + name[-33:]
    # Lead and method first: a narrow viewer clips the end of the title.
    title = f"lead {lead_label}  ·  {method_name}  ·  {name}"
    subtitle = (f"{prediction['label']}  (p = {probability_text(p, thr)} {rel} {thr:.10g})  ·  "
                f"explains: {target_label}")
    return title, subtitle, (theme.FLAG if positive else theme.ACCENT)


def save_xai_figure(path, signal, fs, attribution, labels, segment_result, title, subtitle,
                    subtitle_color, top_percentile=90.0, dpi=150, t0=0.0):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5.0), dpi=dpi)
    FigureCanvasAgg(fig)
    draw_xai(fig, signal, fs, attribution, labels, segment_result, title, subtitle,
             subtitle_color, top_percentile, t0)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
