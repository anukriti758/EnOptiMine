"""
=============================================================================
EV Charging OCPM — Complete three-Problem Methodology
JPL Adaptive Charging Network (ACN-Data)
=============================================================================

PROBLEMS ADDRESSED
------------------
P1  Station Idle-Blocking          3,478 sessions (47.7%) idle >= 2h post-charging
P2  UserInput Lifecycle Integrity  1,217 revised; 2,396 departure mismatches (40.6%)
P3  Lifecycle Ordering Anomalies   1,378 anomalous sessions (18.9%)

THREE STAGES PER PROBLEM
-------------------------
Discovery     — What does the problem look like in the OC-DFG / process model?
Conformance   — Which sessions / objects deviate from the normative lifecycle?
Enhancement   — What process optimisations resolve the problem?

INPUT  : jpl_acn_ocel2.json   (OCEL 2.0 — produced by acn_to_ocel2.py)
OUTPUT : ./outputs/            (CSV tables, PNG charts, printed diagnostics)

VISUALISATION DESIGN (v2)
--------------------------
All enhancement figures use a consistent academic colour system:
  Primary blue   #2C5F8A  — main data series
  Teal           #2A7F6F  — secondary series / conforming
  Amber          #B87333  — moderate severity / warning
  Crimson        #8B2020  — severe / critical
  Slate          #4A5568  — neutral / unclaimed
  Light fills    #EBF3FB / #F0F7F4 — alternating row backgrounds

Typography: DejaVu Sans (matplotlib default), 9 pt body, 10 pt titles.
All figures saved at 300 DPI with tight bounding box.
Panel labels (a), (b), ... in bold 11 pt at upper-left of each subplot.
No text overlaps: annotations placed inside bars when bar is wide enough,
otherwise in a bbox offset to the right.
=============================================================================
"""

import warnings
from pathlib import Path
from typing  import Optional
from collections import defaultdict, Counter

import pandas as pd
import numpy  as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker  as mticker
from matplotlib.gridspec import GridSpec

import pm4py

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OCEL_PATH  = Path("./drive/MyDrive/jpl-ocel.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IDLE_2H         = 2.0
IDLE_6H         = 6.0
SHORTFALL_KWH   = 2.0
DEP_MISMATCH_H  = 1.0
LATE_REVISION_M = 30
CLUSTER_CAP     = 52
HABITUAL_IDLE_H = 4.0
HABITUAL_WINDOW = 10

ACT_CONNECT    = "connect EV"
ACT_START      = "start charging"
ACT_COMPLETE   = "complete charging"
ACT_DISCONNECT = "disconnect EV"
ACT_SUBMIT     = "submit charging request"
ACT_REVISE     = "revise charging request"

OT_SESSION   = "session"
OT_STATION   = "station"
OT_CLUSTER   = "cluster"
OT_SPACE     = "space"
OT_USER      = "user"
OT_USERINPUT = "userinput"

SCHEMA: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION COLOUR SYSTEM  (used by all plot functions)
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "blue"    : "#2C5F8A",   # primary data — bars, histograms
    "teal"    : "#2A7F6F",   # conforming / positive outcome
    "amber"   : "#B87333",   # moderate severity / warning
    "crimson" : "#8B2020",   # severe / critical
    "slate"   : "#4A5568",   # neutral / unclaimed
    "purple"  : "#5B4A8A",   # revision / userinput
    "grey"    : "#7A8599",   # reference lines, axes
    "bg_light": "#F7F9FC",   # figure background
    "bg_strip": "#EBF3FB",   # alternating bar strip
    "grid"    : "#D5DCE8",   # light grid lines
}

# Severity gradient for blocking charts
SEVERITY_3 = [C["teal"], C["amber"], C["crimson"]]

def _apply_base_style(ax, title: str, xlabel: str, ylabel: str,
                      panel_label: str = "", grid_axis: str = "y"):
    """Apply consistent research-paper styling to a single axes."""
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, loc="left",
                 color="#1A2744")
    ax.set_xlabel(xlabel, fontsize=9, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=5)
    ax.tick_params(axis="both", labelsize=8, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["grid"])
    ax.spines["bottom"].set_color(C["grid"])
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=C["grid"], linewidth=0.6, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=C["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if panel_label:
        ax.text(-0.08, 1.06, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left",
                color="#1A2744")


def _annotate_hbar(ax, bar, value_str: str, ax_xlim_max: float,
                   threshold_pct: float = 0.25):
    """
    Place annotation inside the bar if it is wide enough (> threshold_pct of xlim),
    otherwise place it outside with a tight bbox — never overlapping the bar end.
    """
    w = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    if w > ax_xlim_max * threshold_pct:
        ax.text(w * 0.97, y, value_str, va="center", ha="right",
                fontsize=8, color="white", fontweight="bold")
    else:
        ax.text(w + ax_xlim_max * 0.015, y, value_str, va="center", ha="left",
                fontsize=8, color="#1A2744",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=C["grid"], lw=0.6))


def _fig_caption(fig, text: str):
    """Add a small italic caption below the figure."""
    fig.text(0.5, -0.01, text, ha="center", fontsize=7.5,
             style="italic", color=C["slate"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (unchanged logic from original)
# ─────────────────────────────────────────────────────────────────────────────

def _sniff_schema(ocel: pm4py.OCEL) -> dict:
    ev_cols  = list(ocel.events.columns)
    rel_cols = list(ocel.relations.columns)
    obj_cols = list(ocel.objects.columns)

    def _pick(candidates, cols, label):
        for c in candidates:
            if c in cols: return c
        raise RuntimeError(f"Cannot detect '{label}'. Available: {cols}")

    schema = {
        "eid":      _pick(["ocel:eid","ocel_id","event_id","eid"],    ev_cols,  "event-id"),
        "activity": _pick(["ocel:activity","activity"],                ev_cols,  "activity"),
        "timestamp":_pick(["ocel:timestamp","timestamp"],              ev_cols,  "timestamp"),
        "r_eid":    _pick(["ocel:eid","ocel_id","event_id","eid"],    rel_cols, "event-id in relations"),
        "oid":      _pick(["ocel:oid","object_id","oid"],             rel_cols, "object-id"),
        "otype":    _pick(["ocel:type","object_type","type"],         rel_cols, "object-type"),
        "o_oid":    _pick(["ocel:oid","object_id","oid"],             obj_cols, "object-id in objects"),
        "o_otype":  _pick(["ocel:type","object_type","type"],         obj_cols, "object-type in objects"),
    }
    print("  [Schema]", {k: v for k,v in schema.items()})
    return schema


def load_ocel(path: Path) -> pm4py.OCEL:
    print(f"\nLoading OCEL 2.0: {path}")
    ocel = pm4py.read_ocel2_json(str(path))
    print(f"  Events: {len(ocel.events):,}  "
          f"Objects: {len(ocel.objects):,}  "
          f"Relations: {len(ocel.relations):,}")
    global SCHEMA
    SCHEMA = _sniff_schema(ocel)
    ot = ocel.objects[SCHEMA["o_otype"]].value_counts()
    print("  Object types:", dict(ot))
    return ocel


def _build_session_df(ocel: pm4py.OCEL) -> pd.DataFrame:
    S   = SCHEMA
    ev  = ocel.events.copy()
    rel = ocel.relations.copy()

    ev[S["timestamp"]] = pd.to_datetime(ev[S["timestamp"]], utc=True, errors="coerce")
    ev  = ev.rename(columns={S["eid"]:"eid", S["activity"]:"activity", S["timestamp"]:"ts"})
    rel = rel.rename(columns={S["r_eid"]:"eid", S["oid"]:"oid", S["otype"]:"otype"})

    merged = rel.merge(ev[["eid","activity","ts"]], on="eid", how="left")

    sess = (merged[merged["otype"] == OT_SESSION][["oid","eid","activity","ts"]]
            .rename(columns={"oid":"session_id"}))

    PIVOT = {ACT_CONNECT:"conn", ACT_COMPLETE:"done", ACT_DISCONNECT:"disc"}
    pr = (sess[sess["activity"].isin(PIVOT)]
          .assign(col=lambda d: d["activity"].map(PIVOT))
          .sort_values("ts")
          .drop_duplicates(["session_id","col"]))
    tl = (pr.pivot(index="session_id", columns="col", values="ts").reset_index())
    for c in PIVOT.values():
        if c not in tl.columns: tl[c] = pd.NaT

    st_map = (merged[merged["otype"]==OT_STATION][["eid","oid"]]
              .rename(columns={"oid":"station_id"}))
    sess_st = (sess[["session_id","eid"]].merge(st_map, on="eid")
               .drop_duplicates("session_id")[["session_id","station_id"]])

    u_map = (merged[merged["otype"]==OT_USER][["eid","oid"]]
             .rename(columns={"oid":"user_id"}))
    sess_u = (sess[["session_id","eid"]].merge(u_map, on="eid")
              .drop_duplicates("session_id")[["session_id","user_id"]])

    disc_ev = ev[ev["activity"]==ACT_DISCONNECT]
    kwh_col = next((c for c in disc_ev.columns
                    if c not in ("eid","activity","ts") and
                    ("kwh" in c.lower() or "kWh" in c)), None)
    if kwh_col:
        dk = (disc_ev[["eid",kwh_col]].dropna(subset=[kwh_col])
              .rename(columns={kwh_col:"energy_kwh"}))
        sess_kwh = (sess[["session_id","eid"]].merge(dk, on="eid")
                    .drop_duplicates("session_id")[["session_id","energy_kwh"]])
    else:
        sess_kwh = pd.DataFrame(columns=["session_id","energy_kwh"])

    subm_ev = ev[ev["activity"].isin([ACT_SUBMIT, ACT_REVISE])]
    dep_col = next((c for c in subm_ev.columns
                    if c not in ("eid","activity","ts") and
                    "departure" in c.lower()), None)
    if dep_col:
        sd = (subm_ev[["eid",dep_col]].dropna(subset=[dep_col])
              .rename(columns={dep_col:"req_departure"}))
        sess_dep = (sess[["session_id","eid"]].merge(sd, on="eid")
                    .sort_values("eid")
                    .drop_duplicates("session_id")[["session_id","req_departure"]])
    else:
        sess_dep = pd.DataFrame(columns=["session_id","req_departure"])

    rev_ev   = ev[ev["activity"]==ACT_REVISE]
    sess_rev = (sess[["session_id","eid"]].merge(rev_ev[["eid"]], on="eid")
                .groupby("session_id").size().rename("n_revisions").reset_index())

    df = (tl
          .merge(sess_st, on="session_id", how="left")
          .merge(sess_u,  on="session_id", how="left")
          .merge(sess_kwh,on="session_id", how="left")
          .merge(sess_dep,on="session_id", how="left")
          .merge(sess_rev,on="session_id", how="left"))

    df["n_revisions"] = df["n_revisions"].fillna(0).astype(int)
    if "req_departure" in df.columns:
        df["req_departure"] = pd.to_datetime(
            df["req_departure"], utc=True, errors="coerce")

    df["idle_h"]      = np.where(
        df["done"].notna() & df["disc"].notna(),
        (df["disc"] - df["done"]).dt.total_seconds() / 3600, np.nan)
    df["is_claimed"]  = df["user_id"].notna()
    df["has_revision"]= df["n_revisions"] > 0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — OC-DFG DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def stage0_discover_ocdfg(ocel: pm4py.OCEL) -> dict:
    print("\n" + "="*65)
    print("STAGE 0 · OC-DFG Discovery (full log)")
    print("="*65)

    ocdfg = pm4py.discover_ocdfg(ocel)
    S     = SCHEMA
    ev    = ocel.events.rename(columns={S["activity"]:"activity"})

    print("\n[0.1] Activity event counts:")
    for act, cnt in ev["activity"].value_counts().sort_index().items():
        print(f"  {act:42s}  E={cnt:,}")

    print("\n[0.2] OC-DFG arcs (event_couples):")
    arcs = {}
    for (src, tgt, ot), count in ocdfg.get("event_couples", {}).items():
        arcs[(src, tgt, ot)] = count

    if arcs:
        arc_df = pd.DataFrame(
            [{"from":k[0],"to":k[1],"object_type":k[2],"EC":v}
             for k,v in arcs.items()]
        ).sort_values("EC", ascending=False).reset_index(drop=True)
        print(arc_df.to_string(index=False))
        arc_df.to_csv(OUTPUT_DIR / "ocdfg_arcs_full.csv", index=False)

    try:
        pm4py.save_vis_ocdfg(ocdfg, str(OUTPUT_DIR / "ocdfg_full.png"))
        print(f"\n  Saved → {OUTPUT_DIR / 'ocdfg_full.png'}")
    except Exception as e:
        print(f"  [WARN] OC-DFG visualisation: {e}")

    _save_ot_perspective_ocdfgs(ocel)
    return ocdfg


def _save_ot_perspective_ocdfgs(ocel: pm4py.OCEL):
    S = SCHEMA
    object_types = ocel.objects[S["o_otype"]].unique().tolist()
    print("\n[0.3] Per-object-type perspective OC-DFGs:")
    for ot in object_types:
        try:
            ot_ocel  = pm4py.filter_ocel_object_types(ocel, [ot])
            ot_ocdfg = pm4py.discover_ocdfg(ot_ocel)
            out = OUTPUT_DIR / f"ocdfg_perspective_{ot.replace(' ','_')}.png"
            pm4py.save_vis_ocdfg(ot_ocdfg, str(out))
            print(f"  {ot:15s} → {out.name}")
        except Exception as e:
            print(f"  {ot:15s} → [WARN] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 1 — STATION IDLE-BLOCKING
# ─────────────────────────────────────────────────────────────────────────────

def problem1_idle_blocking(ocel: pm4py.OCEL, df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*65)
    print("PROBLEM 1 · Station Idle-Blocking")
    print("="*65)

    idle_df = df[df["idle_h"].notna() & (df["idle_h"] > 0)].copy()

    station_rank = (idle_df.groupby("station_id")
                   .agg(total_idle_h  = ("idle_h","sum"),
                        mean_idle_h   = ("idle_h","mean"),
                        n_sessions    = ("idle_h","count"),
                        n_ge_2h       = ("idle_h", lambda x: (x>=IDLE_2H).sum()),
                        n_ge_6h       = ("idle_h", lambda x: (x>=IDLE_6H).sum()))
                   .round(2)
                   .sort_values("total_idle_h", ascending=False)
                   .reset_index())

    print("\n[P1-D2] Top-10 blocking stations:")
    print(station_rank.head(10).to_string(index=False))
    station_rank.to_csv(OUTPUT_DIR / "p1_station_idle_ranking.csv", index=False)

    onset_hour = pd.to_datetime(df["done"], utc=True, errors="coerce").dt.hour
    hourly = (df.assign(onset_hour=onset_hour)
              .dropna(subset=["onset_hour"])
              .groupby("onset_hour").size()
              .reindex(range(24), fill_value=0)
              .rename("n_blocking_onset"))

    df = df.copy()
    df["p1_idle_h"] = df["idle_h"].fillna(0)
    df["p1_label"]  = "CONFORMING"
    df.loc[df["p1_idle_h"] >= IDLE_2H, "p1_label"] = "IDLE_BLOCKING_MOD"
    df.loc[df["p1_idle_h"] >= IDLE_6H, "p1_label"] = "IDLE_BLOCKING_SEV"

    label_counts = df["p1_label"].value_counts()
    total = len(df)
    for lbl, cnt in label_counts.items():
        print(f"  {lbl:25s}  {cnt:6,}  ({100*cnt/total:.1f}%)")

    df["p1_is_blocking"] = df["p1_idle_h"] >= IDLE_2H

    st_conf = (df.groupby("station_id")["p1_is_blocking"]
               .agg(["sum","count"])
               .rename(columns={"sum":"n_blocking","count":"n_sessions"})
               .assign(blocking_rate=lambda x: x["n_blocking"]/x["n_sessions"])
               .sort_values("blocking_rate", ascending=False)
               .reset_index())
    st_conf.to_csv(OUTPUT_DIR / "p1_station_conformance.csv", index=False)

    baseline_h = df["p1_idle_h"].sum()
    print(f"\n[P1-E1] Baseline idle station-hours: {baseline_h:,.1f}")

    enhancements = []
    for threshold_h, label in [
            (1.0, "Alert at 1 h"),
            (2.0, "Graduated fee (2 h free)"),
            (3.0, "Graduated fee (3 h free)")]:
        mask  = df["p1_idle_h"] > threshold_h
        saved = (df.loc[mask,"p1_idle_h"] - threshold_h).sum()
        enhancements.append({
            "intervention":      label,
            "sessions_affected": int(mask.sum()),
            "idle_h_reclaimed":  round(saved, 1),
            "pct_baseline":      round(100 * saved / baseline_h, 1)})

    onset_h   = pd.to_datetime(df["done"], utc=True, errors="coerce").dt.hour
    pattern_b = onset_h.between(16, 20) & (df["p1_idle_h"] >= IDLE_2H)
    saved_b   = (df.loc[pattern_b,"p1_idle_h"] - IDLE_2H).clip(0).sum()
    enhancements.append({
        "intervention":      "Dynamic fee (peak hours only)",
        "sessions_affected": int(pattern_b.sum()),
        "idle_h_reclaimed":  round(saved_b, 1),
        "pct_baseline":      round(100 * saved_b / baseline_h, 1)})

    enh_df = pd.DataFrame(enhancements)
    print("\n[P1-E2] Intervention impact:")
    print(enh_df.to_string(index=False))
    enh_df.to_csv(OUTPUT_DIR / "p1_interventions.csv", index=False)

    df_sorted = df.sort_values("conn")
    df_sorted["rolling_mean_idle"] = (
        df_sorted.groupby("user_id")["p1_idle_h"]
        .transform(lambda x: x.rolling(HABITUAL_WINDOW, min_periods=3).mean()))
    habitual = df_sorted[df_sorted["rolling_mean_idle"].fillna(0) > HABITUAL_IDLE_H]
    top_habitual = (habitual.groupby("user_id")
                    .agg(mean_idle_h=("p1_idle_h","mean"),
                         n_sessions=("p1_idle_h","count"))
                    .sort_values("mean_idle_h", ascending=False)
                    .head(10).round(2))
    top_habitual.to_csv(OUTPUT_DIR / "p1_habitual_users.csv")

    _plot_p1(idle_df, station_rank, hourly, enh_df, baseline_h)
    return df


def _plot_p1(idle_df, station_rank, hourly, enh_df, baseline_h):
    """
    Four-panel figure for P1 — Station Idle-Blocking.
    (a) Idle duration histogram with severity thresholds
    (b) Top-10 stations ranked by total idle-hours (horizontal bar)
    (c) Blocking onset hour of day (bar chart with peak window)
    (d) Intervention simulation — idle-hours reclaimed (horizontal bar)
    """
    fig = plt.figure(figsize=(16, 11), facecolor=C["bg_light"])
    fig.patch.set_facecolor(C["bg_light"])
    gs  = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38,
                   left=0.08, right=0.97, top=0.91, bottom=0.08)

    # ── (a) Idle duration histogram ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")

    bins = np.linspace(0, 24, 49)
    vals = idle_df["idle_h"].clip(upper=24).values
    n, edges, patches = ax1.hist(vals, bins=bins, color=C["blue"],
                                 edgecolor="white", linewidth=0.4, zorder=3)
    # Colour bars by severity zone
    for patch, left in zip(patches, edges[:-1]):
        if left >= IDLE_6H:
            patch.set_facecolor(C["crimson"])
        elif left >= IDLE_2H:
            patch.set_facecolor(C["amber"])

    ax1.axvline(IDLE_2H, color=C["amber"],   ls="--", lw=1.6, zorder=4)
    ax1.axvline(IDLE_6H, color=C["crimson"], ls="--", lw=1.6, zorder=4)

    # Threshold labels placed just above the x-axis, avoiding top overlap
    ax1.text(IDLE_2H + 0.2, ax1.get_ylim()[1] * 0.03,
             f"Moderate\n≥ {int(IDLE_2H)} h",
             color=C["amber"], fontsize=7.5, va="bottom", style="italic")
    ax1.text(IDLE_6H + 0.2, ax1.get_ylim()[1] * 0.03,
             f"Severe\n≥ {int(IDLE_6H)} h",
             color=C["crimson"], fontsize=7.5, va="bottom", style="italic")

    # Shaded severity zones
    ymax = max(n) * 1.12
    ax1.set_ylim(0, ymax)
    ax1.axvspan(IDLE_2H, IDLE_6H, alpha=0.06, color=C["amber"], zorder=0)
    ax1.axvspan(IDLE_6H, 24,       alpha=0.06, color=C["crimson"], zorder=0)

    _apply_base_style(ax1,
        title="Idle-Blocking Duration Distribution",
        xlabel="Post-charge idle time (hours, capped at 24 h)",
        ylabel="Number of sessions",
        panel_label="(a)")

    # Inset statistics box
    n_2h = (idle_df["idle_h"] >= IDLE_2H).sum()
    n_6h = (idle_df["idle_h"] >= IDLE_6H).sum()
    stats_txt = (f"n = {len(idle_df):,}\n"
                 f"≥ 2 h : {n_2h:,} ({100*n_2h/len(idle_df):.0f}%)\n"
                 f"≥ 6 h : {n_6h:,} ({100*n_6h/len(idle_df):.0f}%)")
    ax1.text(0.97, 0.97, stats_txt, transform=ax1.transAxes,
             fontsize=7.5, va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C["grid"], lw=0.8))

    # ── (b) Top-10 stations ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("white")

    top10 = station_rank.head(10).copy()
    top10["station_short"] = top10["station_id"].str[-9:]  # last 9 chars

    # Colour by blocking rate bands
    def _station_colour(i):
        if i < 2:   return C["crimson"]
        elif i < 5: return C["amber"]
        else:       return C["blue"]

    bar_colors = [_station_colour(i) for i in range(len(top10))]
    y_pos = np.arange(len(top10))
    bars  = ax2.barh(y_pos, top10["total_idle_h"][::-1].values,
                     color=bar_colors[::-1], height=0.65,
                     edgecolor="white", linewidth=0.4)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top10["station_short"][::-1].values, fontsize=8)

    xlim_max = top10["total_idle_h"].max() * 1.22
    ax2.set_xlim(0, xlim_max)

    for bar, row in zip(bars, top10.iloc[::-1].itertuples()):
        pct = 100 * row.n_ge_2h / max(row.n_sessions, 1)
        label_str = f"{row.total_idle_h:.0f} h  ({pct:.0f}% blocking)"
        _annotate_hbar(ax2, bar, label_str, xlim_max, threshold_pct=0.30)

    legend_patches = [
        mpatches.Patch(color=C["crimson"], label="Rank 1–2 (critical)"),
        mpatches.Patch(color=C["amber"],   label="Rank 3–5 (elevated)"),
        mpatches.Patch(color=C["blue"],    label="Rank 6–10"),
    ]
    ax2.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
               framealpha=0.9, edgecolor=C["grid"])

    _apply_base_style(ax2,
        title="Top 10 Stations — Total Idle-Hours",
        xlabel="Total idle station-hours",
        ylabel="Station ID (last 9 characters)",
        panel_label="(b)", grid_axis="x")

    # ── (c) Hourly blocking onset ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("white")

    bar_clr = [C["crimson"] if 16 <= h <= 20 else C["blue"] for h in range(24)]
    ax3.bar(hourly.index, hourly.values, color=bar_clr,
            width=0.75, edgecolor="white", linewidth=0.3, zorder=3)

    # Peak window annotation (span + bracket label above)
    ax3.axvspan(15.5, 20.5, alpha=0.08, color=C["crimson"], zorder=0)
    ymax3 = hourly.max() * 1.18
    ax3.set_ylim(0, ymax3)
    ax3.annotate("", xy=(20.5, ymax3 * 0.92), xytext=(15.5, ymax3 * 0.92),
                 arrowprops=dict(arrowstyle="<->", color=C["crimson"], lw=1.2))
    ax3.text(18, ymax3 * 0.95, "Peak onset\n16:00–20:00",
             ha="center", va="bottom", fontsize=7.5,
             color=C["crimson"], fontweight="bold")

    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)],
                        fontsize=7.5, rotation=30, ha="right")
    _apply_base_style(ax3,
        title="Temporal Distribution of Idle-Blocking Onset",
        xlabel="Hour of day (local time, doneChargingTime)",
        ylabel="Blocking onset events",
        panel_label="(c)")

    # ── (d) Intervention simulation ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("white")

    intv_colors = [C["teal"], C["blue"], C["slate"], C["amber"]]
    y_pos4  = np.arange(len(enh_df))
    bars4   = ax4.barh(y_pos4,
                       enh_df["idle_h_reclaimed"][::-1].values,
                       color=intv_colors,
                       height=0.55, edgecolor="white", linewidth=0.4)

    # Baseline reference line at 50%
    if baseline_h > 0:
        ref_x = baseline_h * 0.5
        ax4.axvline(ref_x, color=C["grey"], ls=(0, (5,3)), lw=1.2, zorder=2)
        ax4.text(ref_x + baseline_h * 0.008, len(enh_df) - 0.1,
                 "50% baseline", fontsize=7.5, color=C["grey"],
                 va="bottom", ha="left")

    xlim4 = enh_df["idle_h_reclaimed"].max() * 1.28
    ax4.set_xlim(0, xlim4)
    ax4.set_yticks(y_pos4)

    # Wrap long intervention labels
    labels4 = enh_df["intervention"][::-1].tolist()
    ax4.set_yticklabels(labels4, fontsize=8)

    for bar, (_, row) in zip(bars4, enh_df.iloc[::-1].iterrows()):
        txt = f"{row['idle_h_reclaimed']:,.0f} h  ({row['pct_baseline']:.1f}%)"
        _annotate_hbar(ax4, bar, txt, xlim4, threshold_pct=0.28)

    _apply_base_style(ax4,
        title="Counterfactual Simulation — Idle-Hours Reclaimed",
        xlabel="Station-hours reclaimed (counterfactual estimate)",
        ylabel="Intervention policy",
        panel_label="(d)", grid_axis="x")

    fig.suptitle(
        "Figure P1 — Station Idle-Blocking: Discovery, Conformance and Enhancement Analysis\n"
        "JPL Adaptive Charging Network  |  7,299 sessions  |  52 EVSEs  |  Sep 2018 – Feb 2019",
        fontsize=10.5, fontweight="bold", color="#1A2744", y=0.98)

    _fig_caption(fig,
        "Colour coding: teal = conforming / best outcome, amber = moderate severity, "
        "crimson = severe. Bars annotated with absolute and percentage values.")

    out = OUTPUT_DIR / "p1_idle_blocking_analysis.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=C["bg_light"])
    plt.close(fig)
    print(f"\n  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 2 — USERINPUT LIFECYCLE INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def problem3_userinput_lifecycle(ocel: pm4py.OCEL, df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*65)
    print("PROBLEM 3 · UserInput Lifecycle Integrity")
    print("="*65)

    rev_dist = Counter(df["n_revisions"])
    print("[P3-D2] Revision count distribution:")
    for k in sorted(rev_dist.keys()):
        print(f"  {k:2d} revisions: {rev_dist[k]:,} sessions")

    df3 = df.copy()
    df3["dep_mismatch_h"] = np.where(
        df3["req_departure"].notna() & df3["disc"].notna(),
        (df3["disc"] - df3["req_departure"]).dt.total_seconds() / 3600, np.nan)
    df3["has_dep_mismatch"] = df3["dep_mismatch_h"].fillna(0) > DEP_MISMATCH_H

    n_mismatch = df3["has_dep_mismatch"].sum()
    n_claimed  = df3["is_claimed"].sum()
    print(f"\n[P3-D3] Departure mismatch: {n_mismatch:,} sessions "
          f"({100*n_mismatch/max(n_claimed,1):.1f}% of claimed)")
    print(f"  Mean excess: {df3.loc[df3['has_dep_mismatch'],'dep_mismatch_h'].mean():.1f} h")
    print(f"  Max excess:  {df3.loc[df3['has_dep_mismatch'],'dep_mismatch_h'].max():.1f} h")

    df3["p3_label"] = "CONFORMING_USERINPUT"
    df3.loc[df3["has_dep_mismatch"], "p3_label"] = "DEPARTURE_MISMATCH"
    df3.loc[~df3["is_claimed"],       "p3_label"] = "UNCLAIMED"

    label_counts = df3["p3_label"].value_counts()
    total = len(df3)
    for lbl, cnt in label_counts.items():
        print(f"  {lbl:35s}  {cnt:6,}  ({100*cnt/total:.1f}%)")

    df3[["session_id","station_id","user_id","conn","done","disc",
         "req_departure","n_revisions","dep_mismatch_h","p3_label"]]\
        .to_csv(OUTPUT_DIR / "p3_userinput_conformance.csv", index=False)

    user_mismatch = (df3[df3["is_claimed"] & df3["has_dep_mismatch"]]
                    .groupby("user_id")
                    .agg(n_mismatch=("has_dep_mismatch","sum"),
                         mean_excess_h=("dep_mismatch_h","mean"))
                    .sort_values("mean_excess_h", ascending=False)
                    .round(2).head(10))
    user_mismatch.to_csv(OUTPUT_DIR / "p3_user_mismatch_ranking.csv")

    _plot_p3(df3)
    return df3


def _plot_p3(df3):
    """
    Two-panel figure for P3 — UserInput Lifecycle Integrity.
    (a) Departure mismatch distribution (disconnect − declared departure)
    (b) Revision count per session — log-scale bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             facecolor=C["bg_light"],
                             gridspec_kw={"wspace": 0.38})
    fig.patch.set_facecolor(C["bg_light"])

    # ── (a) Departure mismatch histogram ─────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("white")

    mismatch = df3["dep_mismatch_h"].dropna()
    bins_m   = np.linspace(mismatch.clip(-4, 30).min(), 30, 68)
    vals_m   = mismatch.clip(-4, 30).values

    n_m, edges_m, patches_m = ax1.hist(vals_m, bins=bins_m,
                                        color=C["purple"],
                                        edgecolor="white", linewidth=0.3, zorder=3)

    # Colour coding: negative = arrived early (teal), >1h mismatch = crimson
    for patch, left in zip(patches_m, edges_m[:-1]):
        if left < 0:
            patch.set_facecolor(C["teal"])
        elif left > DEP_MISMATCH_H:
            patch.set_facecolor(C["crimson"])
        else:
            patch.set_facecolor(C["amber"])

    ax1.axvline(DEP_MISMATCH_H, color=C["crimson"], ls="--", lw=1.6, zorder=4)
    ax1.axvline(0,              color=C["grey"],    ls="-",  lw=0.8, zorder=4, alpha=0.7)

    # Annotated regions
    ymax_m = max(n_m) * 1.15
    ax1.set_ylim(0, ymax_m)
    ax1.axvspan(-4, 0,               alpha=0.07, color=C["teal"],    zorder=0)
    ax1.axvspan(DEP_MISMATCH_H, 30,  alpha=0.07, color=C["crimson"], zorder=0)

    n_over = (mismatch > DEP_MISMATCH_H).sum()
    pct_ov  = 100 * n_over / max(len(mismatch), 1)
    ax1.text(0.97, 0.97,
             f"Mismatch > {DEP_MISMATCH_H} h:\n{n_over:,}  ({pct_ov:.1f}%)",
             transform=ax1.transAxes, fontsize=8, va="top", ha="right",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C["grid"], lw=0.8))

    ax1.text(DEP_MISMATCH_H + 0.3, ymax_m * 0.85,
             f"Threshold\n{DEP_MISMATCH_H} h",
             fontsize=7.5, color=C["crimson"], style="italic", va="top")

    mean_ex = df3.loc[df3["has_dep_mismatch"], "dep_mismatch_h"].mean()
    ax1.axvline(mean_ex, color=C["amber"], ls=":", lw=1.4, zorder=4)
    ax1.text(mean_ex + 0.3, ymax_m * 0.65,
             f"Mean excess\n{mean_ex:.1f} h",
             fontsize=7.5, color=C["amber"], style="italic")

    _apply_base_style(ax1,
        title="Departure Mismatch Distribution",
        xlabel="Δ departure (disconnect − declared departure, hours)",
        ylabel="Number of sessions",
        panel_label="(a)")

    # ── (b) Revision count distribution (log scale) ───────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")

    rev_counts    = df3["n_revisions"].value_counts().sort_index()
    max_rev_shown = 10
    rev_display   = rev_counts.reindex(range(max_rev_shown + 1), fill_value=0)
    x_labels      = [str(i) if i < max_rev_shown else f"{max_rev_shown}+"
                     for i in range(max_rev_shown + 1)]

    # Also aggregate the tail (>= max_rev_shown)
    tail_count = rev_counts[rev_counts.index >= max_rev_shown].sum()
    rev_display.iloc[-1] = tail_count

    bar_cols_rev = [C["slate"] if i == 0 else C["purple"]
                    for i in range(len(rev_display))]
    bars_r = ax2.bar(range(len(rev_display)), rev_display.values,
                     color=bar_cols_rev, width=0.65,
                     edgecolor="white", linewidth=0.4, zorder=3)

    ax2.set_yscale("log")
    ax2.set_xticks(range(len(rev_display)))
    ax2.set_xticklabels(x_labels, fontsize=8.5)

    # Annotate each bar with raw count
    for bar, val in zip(bars_r, rev_display.values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     val * 1.25,
                     f"{val:,}",
                     ha="center", va="bottom", fontsize=7.5, color="#1A2744")

    # Fraction with >= 1 revision
    n_revised  = (df3["n_revisions"] >= 1).sum()
    pct_rev    = 100 * n_revised / max(len(df3), 1)
    ax2.text(0.97, 0.97,
             f"Sessions with ≥ 1 revision:\n{n_revised:,}  ({pct_rev:.1f}%)",
             transform=ax2.transAxes, fontsize=8, va="top", ha="right",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C["grid"], lw=0.8))

    legend_patches = [
        mpatches.Patch(color=C["slate"],  label="No revision (initial only)"),
        mpatches.Patch(color=C["purple"], label="≥ 1 revision"),
    ]
    ax2.legend(handles=legend_patches, fontsize=8, loc="upper right",
               framealpha=0.9, edgecolor=C["grid"])

    _apply_base_style(ax2,
        title="UserInput Revision Count per Session (log scale)",
        xlabel="Number of UserInput revisions",
        ylabel="Number of sessions (log scale)",
        panel_label="(b)")

    fig.suptitle(
        "Figure P3 — UserInput Lifecycle Integrity: "
        "Departure Mismatch and Revision Behaviour\n"
        "2,447 sessions with departure mismatch > 1 h  |  "
        "2,502 revision events across 1,217 sessions",
        fontsize=10.5, fontweight="bold", color="#1A2744", y=1.02)

    _fig_caption(fig,
        "Teal = arrived before declared departure. Amber = within 1 h overstay. "
        "Crimson = departure mismatch > 1 h (conformance violation).")

    out = OUTPUT_DIR / "p3_userinput_lifecycle.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=C["bg_light"])
    plt.close(fig)
    print(f"\n  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 3 — LIFECYCLE ORDERING AND DATA INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def problem4_lifecycle_ordering(ocel: pm4py.OCEL, df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*65)
    print("PROBLEM 4 · Lifecycle Ordering and Data Integrity")
    print("="*65)

    df4 = df.copy()
    df4["ordering_violation"] = (
        df4["done"].notna() & df4["disc"].notna() &
        (df4["done"] > df4["disc"]))
    df4["missing_done"] = (
        df4["done"].isna() &
        df4.get("energy_kwh", pd.Series(0, index=df4.index)).fillna(0) > 0)

    n_ov = df4["ordering_violation"].sum()
    n_md = df4["missing_done"].sum()
    print(f"  Ordering violations: {n_ov:,}  |  Missing done: {n_md:,}")

    station_anomalies = (df4[df4["ordering_violation"] | df4["missing_done"]]
                        .groupby("station_id")
                        .agg(n_ordering=("ordering_violation","sum"),
                             n_missing=("missing_done","sum"))
                        .assign(total=lambda x: x["n_ordering"]+x["n_missing"])
                        .sort_values("total", ascending=False)
                        .head(15).reset_index())
    station_anomalies.to_csv(OUTPUT_DIR / "p4_station_anomalies.csv", index=False)

    df4["p4_label"] = "CONFORMING"
    df4.loc[df4["ordering_violation"], "p4_label"] = "ORDERING_VIOLATION"
    df4.loc[df4["missing_done"],       "p4_label"] = "MISSING_DONE_SIGNAL"

    label_counts = df4["p4_label"].value_counts()
    total = len(df4)
    for lbl, cnt in label_counts.items():
        print(f"  {lbl:30s}  {cnt:6,}  ({100*cnt/total:.1f}%)")

    station_quality = (df4.groupby("station_id")
                       .agg(n_total=("session_id","count"),
                            n_anomalies=("p4_label",
                                lambda x: (x != "CONFORMING").sum()))
                       .assign(anomaly_rate=lambda x: x["n_anomalies"]/x["n_total"])
                       .sort_values("anomaly_rate", ascending=False)
                       .reset_index())
    station_quality.to_csv(OUTPUT_DIR / "p4_station_quality_scores.csv", index=False)

    df4[["session_id","station_id","user_id","conn","done","disc",
         "ordering_violation","missing_done","p4_label"]]\
        .to_csv(OUTPUT_DIR / "p4_lifecycle_conformance.csv", index=False)

    _plot_p4(df4, station_anomalies)
    return df4


def _plot_p4(df4, station_anomalies):
    """
    Two-panel figure for P4 — Lifecycle Ordering Anomalies.
    (a) Conformance label distribution — horizontal stacked proportion bar
        (more informative than a pie chart, no text overlap)
    (b) Top-10 anomaly stations — stacked horizontal bar (ordering vs missing)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             facecolor=C["bg_light"],
                             gridspec_kw={"wspace": 0.40})
    fig.patch.set_facecolor(C["bg_light"])

    # ── (a) Conformance profile — stacked proportion bar ─────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("white")

    label_order  = ["CONFORMING", "ORDERING_VIOLATION", "MISSING_DONE_SIGNAL"]
    label_colors = [C["teal"], C["crimson"], C["amber"]]
    label_short  = ["Conforming", "Ordering\nviolation", "Missing\ndone signal"]

    counts  = [df4["p4_label"].value_counts().get(l, 0) for l in label_order]
    total   = sum(counts)
    pcts    = [100 * c / max(total, 1) for c in counts]

    # Horizontal stacked bar (one row)
    left = 0
    for cnt, pct, color, short in zip(counts, pcts, label_colors, label_short):
        ax1.barh(0, pct, left=left, color=color, height=0.35,
                 edgecolor="white", linewidth=0.5)
        # Label inside if wide enough, else skip (annotate below)
        if pct > 5:
            ax1.text(left + pct / 2, 0,
                     f"{short}\n{pct:.1f}%\n({cnt:,})",
                     ha="center", va="center", fontsize=8,
                     color="white" if pct > 10 else "#1A2744",
                     fontweight="bold", linespacing=1.3)
        left += pct

    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.set_xlabel("Proportion of sessions (%)", fontsize=9, labelpad=5)

    legend_patches = [
        mpatches.Patch(color=c, label=f"{s}  ({cnt:,})")
        for c, s, cnt in zip(label_colors, label_short, counts)
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.40), ncol=3, framealpha=0.9,
               edgecolor=C["grid"])

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_color(C["grid"])
    ax1.tick_params(axis="x", labelsize=8, length=3)
    ax1.text(-0.08, 1.14, "(a)", transform=ax1.transAxes,
             fontsize=11, fontweight="bold", color="#1A2744")
    ax1.set_title("Lifecycle Conformance Profile — All Sessions",
                  fontsize=10, fontweight="bold", pad=8, loc="left",
                  color="#1A2744")

    # ── (b) Top-10 anomaly stations ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")

    if not station_anomalies.empty:
        top = station_anomalies.head(10).copy()
        top["station_short"] = top["station_id"].str[-9:]

        y_pos = np.arange(len(top))
        ax2.barh(y_pos, top["n_missing"][::-1].values,
                 color=C["amber"], height=0.55, edgecolor="white",
                 linewidth=0.4, label="Missing done-charging signal",
                 zorder=3)
        ax2.barh(y_pos, top["n_ordering"][::-1].values,
                 left=top["n_missing"][::-1].values,
                 color=C["crimson"], height=0.55, edgecolor="white",
                 linewidth=0.4, label="Ordering violation (complete charging > disconnect EV)",
                 zorder=3)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top["station_short"][::-1].values, fontsize=8)

        xlim_b = top["total"].max() * 1.30
        ax2.set_xlim(0, xlim_b)
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        for i, (_, row) in enumerate(top.iloc[::-1].iterrows()):
            total_anom = row["total"]
            n_sess     = df4[df4["station_id"] == row["station_id"]].shape[0]
            rate       = 100 * total_anom / max(n_sess, 1)
            ax2.text(total_anom + xlim_b * 0.02, i,
                     f"{int(total_anom)} ({rate:.1f}%)",
                     va="center", ha="left", fontsize=7.5, color="#1A2744",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec=C["grid"], lw=0.5))

        ax2.legend(fontsize=8, loc="lower right",
                   framealpha=0.9, edgecolor=C["grid"])

    _apply_base_style(ax2,
        title="Top 10 Stations — Anomaly Count by Type",
        xlabel="Number of anomalous sessions",
        ylabel="Station ID (last 9 characters)",
        panel_label="(b)", grid_axis="x")

    fig.suptitle(
        "Figure P4 — Lifecycle Ordering and Data Integrity Anomalies\n"
        "111 ordering violations (complete charging > disconnect EV)  |  "
        "1,267 sessions with missing doneChargingTime",
        fontsize=10.5, fontweight="bold", color="#1A2744", y=1.02)

    _fig_caption(fig,
        "Ordering violations indicate EVSE firmware logging delays. "
        "Missing done-signal sessions are structurally excluded from idle-time analysis.")

    out = OUTPUT_DIR / "p4_lifecycle_ordering.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=C["bg_light"])
    plt.close(fig)
    print(f"\n  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PROBLEM SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def cross_problem_summary(df4: pd.DataFrame):
    print("\n" + "="*65)
    print("CROSS-PROBLEM ANALYSIS")
    print("="*65)

    p1 = df4.get("p1_is_blocking",   pd.Series(False, index=df4.index)).fillna(False)
    p2 = df4.get("has_shortfall",    pd.Series(False, index=df4.index)).fillna(False)
    p3 = df4.get("has_dep_mismatch", pd.Series(False, index=df4.index)).fillna(False)
    p4 = (df4.get("p4_label", pd.Series("CONFORMING", index=df4.index))
              != "CONFORMING")

    df4["n_problems"] = (p1.astype(int) + p2.astype(int) +
                         p3.astype(int) + p4.astype(int))

    print("\n  Sessions affected by N simultaneous problems:")
    for n in range(5):
        cnt = (df4["n_problems"] == n).sum()
        print(f"  {n} problems: {cnt:6,}  ({100*cnt/len(df4):.1f}%)")

    df4[["session_id","station_id","user_id","n_problems",
         "p1_label","p2_label","p3_label","p4_label"]]\
        .to_csv(OUTPUT_DIR / "cross_problem_labels.csv", index=False)
    print(f"\n  Saved → {OUTPUT_DIR / 'cross_problem_labels.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print(" EV Charging — Complete Four-Problem OCPM Methodology")
    print(" JPL Adaptive Charging Network (ACN-Data)")
    print("="*65)

    if not OCEL_PATH.exists():
        raise FileNotFoundError(
            f"\nOCEL file not found: {OCEL_PATH}\n"
            "Generate it with acn_to_ocel2.py and update OCEL_PATH above.")

    ocel = load_ocel(OCEL_PATH)

    print("\nBuilding session timelines …")
    df = _build_session_df(ocel)
    print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    ocdfg = stage0_discover_ocdfg(ocel)
    df    = problem1_idle_blocking(ocel, df)
    df    = problem2_energy_shortfall(ocel, df)
    df    = problem3_userinput_lifecycle(ocel, df)
    df    = problem4_lifecycle_ordering(ocel, df)
    cross_problem_summary(df)

    print("\n" + "="*65)
    print(f" PIPELINE COMPLETE — outputs in: {OUTPUT_DIR.resolve()}")
    print("="*65)
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name:50s} {f.stat().st_size/1024:6.1f} KB")


if __name__ == "__main__":
    main()
