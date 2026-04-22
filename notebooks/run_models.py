#!/usr/bin/env python3
"""
Author: Claude (Anthropic) — generated for the encoding project

Interactive model-fitting launcher for  GLM encoding pipeline.

Presents a terminal UI for selecting which models to fit and which diagnostics
to compute, then dispatches to fit_all_models helpers.

Run from project root:
    python notebooks/run_models.py
"""

import curses
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── resolve imports before curses takes over the terminal ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from encoding_utils import (
    CONFIG, build_model_registry, load_and_prepare_data,
    fit_glm_all_units, model_csv_path,
    make_history_formula, make_history_transform,
)

os.chdir(CONFIG["base_dir"])

# ── model metadata ────────────────────────────────────────────────────────────

CATEGORIES = [
    ("Null / baseline", ["null", "null_outbound"]),
    ("Single-variable", ["trial_type", "choice", "speed_spline", "pos_spline",
                         "branch_pos_spline", "trial_progress_spline"]),
    ("Multi-variable",  ["full_model", "temporal_model", "choice_full_model",
                         "choice_temporal_model"]),
]

ACTIONS = ["fit", "diagnostics"]


def _run_name(name, fit_history=False):
    return f"{name}_history" if fit_history else name


def _csv_exists(name, fit_history=False):
    return os.path.exists(model_csv_path(_run_name(name, fit_history)))


# ── curses UI ─────────────────────────────────────────────────────────────────

def run_selector(stdscr):
    curses.curs_set(0)
    curses.use_default_colors()

    # colour pairs
    curses.init_pair(1, curses.COLOR_GREEN, -1)    # selected / cached
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # category header
    curses.init_pair(3, curses.COLOR_CYAN, -1)     # keybind hints
    curses.init_pair(4, curses.COLOR_RED, -1)       # not cached
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # cursor

    GREEN  = curses.color_pair(1) | curses.A_BOLD
    YELLOW = curses.color_pair(2) | curses.A_BOLD
    CYAN   = curses.color_pair(3)
    RED    = curses.color_pair(4)
    CURSOR = curses.color_pair(5) | curses.A_BOLD

    registry = build_model_registry()

    # Build flat item list: (type, key, category_idx)
    # type: "header" | "model"
    items = []
    for cat_idx, (cat_name, models) in enumerate(CATEGORIES):
        items.append(("header", cat_name, cat_idx))
        for m in models:
            items.append(("model", m, cat_idx))

    # state
    selected_fit  = set()
    selected_diag = set()
    refit = False
    fit_history = False
    cursor = 0
    action_col = 0  # 0 = fit, 1 = diagnostics
    scroll_offset = 0

    def all_models():
        return [key for typ, key, _ in items if typ == "model"]

    def models_in_category(cat_idx):
        return [key for typ, key, ci in items if typ == "model" and ci == cat_idx]

    def model_indices():
        return [i for i, (typ, _, _) in enumerate(items) if typ == "model"]

    # skip cursor to nearest model row
    def snap_cursor():
        nonlocal cursor
        if items[cursor][0] == "header":
            for i in range(cursor, len(items)):
                if items[i][0] == "model":
                    cursor = i
                    return
            for i in range(cursor, -1, -1):
                if items[i][0] == "model":
                    cursor = i
                    return

    snap_cursor()

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        # ensure cursor is visible
        visible_rows = max_y - 8  # reserve space for header + footer
        if cursor < scroll_offset:
            scroll_offset = cursor
        if cursor >= scroll_offset + visible_rows:
            scroll_offset = cursor - visible_rows + 1

        # ── header ────────────────────────────────────────────────────────
        title = "GLM Encoding Models — Interactive Launcher"
        stdscr.addnstr(0, 0, title, max_x - 1, curses.A_BOLD | curses.A_UNDERLINE)

        col_fit_x  = 42
        col_diag_x = 52
        col_cache_x = 66

        stdscr.addnstr(2, 0, "  Model", max_x - 1, curses.A_BOLD)
        fit_attr  = curses.A_BOLD | curses.A_UNDERLINE if action_col == 0 else curses.A_BOLD
        diag_attr = curses.A_BOLD | curses.A_UNDERLINE if action_col == 1 else curses.A_BOLD
        if col_fit_x < max_x:
            stdscr.addnstr(2, col_fit_x, "Fit", max_x - col_fit_x - 1, fit_attr)
        if col_diag_x < max_x:
            stdscr.addnstr(2, col_diag_x, "Diagnostics", max_x - col_diag_x - 1, diag_attr)
        if col_cache_x < max_x:
            stdscr.addnstr(2, col_cache_x, "Cached?", max_x - col_cache_x - 1, curses.A_BOLD)

        # ── model list ────────────────────────────────────────────────────
        row = 3
        for idx in range(scroll_offset, min(len(items), scroll_offset + visible_rows)):
            if row >= max_y - 5:
                break
            typ, key, cat_idx = items[idx]
            is_cursor = (idx == cursor)

            if typ == "header":
                label = f"  ── {key} ──"
                stdscr.addnstr(row, 0, label, max_x - 1, YELLOW)
            else:
                # model row
                entry = registry[key]
                dataset_tag = "out" if entry["dataset"] == "outbound" else "all"
                label = f"    {key}"
                cached = _csv_exists(key, fit_history=fit_history)

                attr = CURSOR if is_cursor else curses.A_NORMAL
                stdscr.addnstr(row, 0, label.ljust(40), min(40, max_x - 1), attr)

                # dataset tag
                if 41 < max_x:
                    stdscr.addnstr(row, 40, f"({dataset_tag})", max_x - 41, curses.A_DIM)

                # fit checkbox
                if col_fit_x + 3 < max_x:
                    f_mark = "[x]" if key in selected_fit else "[ ]"
                    f_attr = GREEN if key in selected_fit else curses.A_DIM
                    if is_cursor and action_col == 0:
                        f_attr = CURSOR
                    stdscr.addnstr(row, col_fit_x, f_mark, max_x - col_fit_x - 1, f_attr)

                # diagnostics checkbox
                if col_diag_x + 5 < max_x:
                    d_mark = "  [x]" if key in selected_diag else "  [ ]"
                    d_attr = GREEN if key in selected_diag else curses.A_DIM
                    if is_cursor and action_col == 1:
                        d_attr = CURSOR
                    stdscr.addnstr(row, col_diag_x, d_mark, max_x - col_diag_x - 1, d_attr)

                # cache status
                if col_cache_x + 3 < max_x:
                    if cached:
                        stdscr.addnstr(row, col_cache_x, "  yes", max_x - col_cache_x - 1, GREEN)
                    else:
                        stdscr.addnstr(row, col_cache_x, "  no", max_x - col_cache_x - 1, RED)

            row += 1

        # ── refit toggle ──────────────────────────────────────────────────
        refit_row = max_y - 5
        if refit_row > 0 and refit_row < max_y:
            refit_mark = "[x]" if refit else "[ ]"
            refit_attr = RED | curses.A_BOLD if refit else curses.A_DIM
            stdscr.addnstr(refit_row, 0, f"  Force refit (ignore cache): {refit_mark}", max_x - 1, refit_attr)

        fit_hist_row = max_y - 6
        if fit_hist_row > 0 and fit_hist_row < max_y:
            hist_mark = "[x]" if fit_history else "[ ]"
            hist_attr = GREEN if fit_history else curses.A_DIM
            stdscr.addnstr(
                fit_hist_row, 0,
                f"  Fit history terms (all models): {hist_mark}",
                max_x - 1, hist_attr
            )

        # ── summary ───────────────────────────────────────────────────────
        summary_row = max_y - 4
        if summary_row > 0 and summary_row < max_y:
            n_fit  = len(selected_fit)
            n_diag = len(selected_diag)
            stdscr.addnstr(summary_row, 0,
                           f"  Selected: {n_fit} fit, {n_diag} diagnostics",
                           max_x - 1, curses.A_BOLD)

        # ── keybinds ──────────────────────────────────────────────────────
        help_row = max_y - 2
        if help_row > 0 and help_row < max_y:
            hints = "SPACE toggle  TAB column  a select-all  n select-none  c select-category  h fit_history  r refit  ENTER run  q quit"
            stdscr.addnstr(help_row, 0, hints[:max_x - 1], max_x - 1, CYAN)

        stdscr.refresh()

        # ── input ─────────────────────────────────────────────────────────
        key = stdscr.getch()

        if key == ord("q") or key == 27:  # q or Esc
            return None, None, False, False

        elif key == curses.KEY_UP or key == ord("k"):
            idx_list = model_indices()
            cur_pos = idx_list.index(cursor) if cursor in idx_list else 0
            if cur_pos > 0:
                cursor = idx_list[cur_pos - 1]

        elif key == curses.KEY_DOWN or key == ord("j"):
            idx_list = model_indices()
            cur_pos = idx_list.index(cursor) if cursor in idx_list else 0
            if cur_pos < len(idx_list) - 1:
                cursor = idx_list[cur_pos + 1]

        elif key == ord("\t") or key == curses.KEY_LEFT or key == curses.KEY_RIGHT:
            action_col = 1 - action_col

        elif key == ord(" "):
            # toggle current item
            if items[cursor][0] == "model":
                m = items[cursor][1]
                s = selected_fit if action_col == 0 else selected_diag
                if m in s:
                    s.discard(m)
                else:
                    s.add(m)

        elif key == ord("a"):
            # select all
            s = selected_fit if action_col == 0 else selected_diag
            for m in all_models():
                s.add(m)

        elif key == ord("n"):
            # select none
            s = selected_fit if action_col == 0 else selected_diag
            s.clear()

        elif key == ord("c"):
            # toggle entire category of current cursor
            if items[cursor][0] == "model":
                cat_idx = items[cursor][2]
                cat_models = models_in_category(cat_idx)
                s = selected_fit if action_col == 0 else selected_diag
                if all(m in s for m in cat_models):
                    for m in cat_models:
                        s.discard(m)
                else:
                    for m in cat_models:
                        s.add(m)

        elif key == ord("r"):
            refit = not refit

        elif key == ord("h"):
            fit_history = not fit_history


        elif key == ord("\n") or key == curses.KEY_ENTER:
            if selected_fit or selected_diag:
                return selected_fit.copy(), selected_diag.copy(), refit, fit_history

    return None, None, False, False


# ── execution ─────────────────────────────────────────────────────────────────

def run_selected(selected_fit, selected_diag, refit, fit_history=False):
    """Load data and run the selected fits and diagnostics."""
    print("\nLoading data...")
    data = load_and_prepare_data()
    cov_df_common        = data["cov_df_common"]
    cov_df_out_common    = data["cov_df_out_common"]
    spike_counts_common  = data["spike_counts_common"]
    spike_counts_out_common = data["spike_counts_out_common"]
    unit_ids             = data["unit_ids"]
    base_dir             = CONFIG["base_dir"]

    registry = build_model_registry()
    datasets = {
        "common":   (cov_df_common, spike_counts_common),
        "outbound": (cov_df_out_common, spike_counts_out_common),
    }

    print(f"  {len(cov_df_common):,} bins (common), "
          f"{len(cov_df_out_common):,} bins (outbound), "
          f"{len(unit_ids)} units\n")
    if fit_history:
        print("  History augmentation: ON")
    print()
    history_transform = make_history_transform() if fit_history else None

    # preserve registry order
    ordered_keys = list(registry.keys())

    # ── single-pass execution (fit once per model) ───────────────────────
    # selected_fit: params-only
    # selected_diag: fit + deep diagnostics
    run_list = [k for k in ordered_keys if (k in selected_fit or k in selected_diag)]
    if run_list:
        n_deep = sum(1 for k in run_list if k in selected_diag)
        print(f"{'='*60}")
        print(f"  Running {len(run_list)} model(s): {len(run_list)-n_deep} params-only, {n_deep} deep diagnostics")
        print(f"  {'force refit' if refit else 'skip cached'}")
        print(f"{'='*60}")

        for idx, name in enumerate(run_list, 1):
            entry = registry[name]
            cov_df, sc = datasets[entry["dataset"]]
            run_name = _run_name(name, fit_history=fit_history)
            formula = make_history_formula(entry["formula"]) if fit_history else entry["formula"]
            deep = (name in selected_diag)
            mode_label = "deep diagnostics" if deep else "params-only"
            csv_stem = os.path.splitext(os.path.basename(model_csv_path(run_name)))[0]

            print(
                f"\n  [{idx}/{len(run_list)}] {run_name} ({mode_label}) "
                f"({len(cov_df):,} bins)...",
                flush=True,
            )

            t0 = time.time()
            result = fit_glm_all_units(
                formula, cov_df, sc, unit_ids,
                model_name=run_name, refit=refit,
                per_unit_transform=history_transform,
                deep_diagnostics=deep,
                diagnostics_base_dir=base_dir,
                diagnostics_model_name=csv_stem,
                diagnostics_covariate_cols=["linear_position", "speed", "trial_progress"],
                diagnostics_categorical_cols=["trial_type", "choice"],
                diagnostics_n_bins=50,
            )
            elapsed = time.time() - t0
            n_ok = result["converged"].sum() if "converged" in result.columns else "?"
            print(f"    {n_ok}/{len(unit_ids)} converged ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print("  All done.")
    print(f"{'='*60}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    selected_fit, selected_diag, refit, fit_history = curses.wrapper(run_selector)
    if selected_fit is None and selected_diag is None:
        print("Cancelled.")
        return

    run_selected(selected_fit, selected_diag, refit, fit_history=fit_history)


if __name__ == "__main__":
    main()
