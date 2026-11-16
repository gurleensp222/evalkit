"""Compare saved runs.

Launch with ``streamlit run scripts/dashboard.py`` after
``pip install -e ".[dashboard]"``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from evalkit.analysis import by_category, failures, load_runs, summary
from evalkit.storage import DEFAULT_RUNS_DIR, list_run_ids


def main() -> None:
    """Draw the run comparison page."""
    st.set_page_config(page_title="evalkit", layout="wide")
    st.title("evalkit runs")
    directory = Path(st.sidebar.text_input("Runs directory", value=str(DEFAULT_RUNS_DIR)))
    run_ids = list_run_ids(directory=directory)
    if not run_ids:
        st.info("No runs in the database yet.")
        return

    chosen = st.multiselect("Runs", run_ids, default=run_ids[:2])
    if not chosen:
        st.info("Select at least one run.")
        return

    frame = load_runs(chosen, directory=directory)
    st.subheader("Summary")
    st.dataframe(summary(frame), use_container_width=True)
    st.subheader("Pass rate by category")
    st.dataframe(by_category(frame), use_container_width=True)
    st.subheader("Failures")
    st.dataframe(failures(frame), use_container_width=True)

    changed = frame.loc[frame["unstable"].astype(bool) | frame["output_varies"].astype(bool)]
    st.subheader("Unstable answers")
    if changed.empty:
        st.write("No task changed its pass result or its answer text.")
        return
    columns = [
        column
        for column in (
            "run_id",
            "task_id",
            "attempt",
            "model",
            "output",
            "passed",
            "unstable",
            "output_varies",
        )
        if column in changed.columns
    ]
    st.dataframe(changed[columns], use_container_width=True)


main()
