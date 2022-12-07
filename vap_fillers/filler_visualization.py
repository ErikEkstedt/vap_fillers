from argparse import ArgumentParser
import streamlit as st

from vap_fillers.utils import load_model
from vap_fillers.main import (
    load_fillers,
    load_filler,
    find_difference,
    find_no_cross,
    plot_filler,
    extract_and_plot_diff_bars,
)


parser = ArgumentParser()
parser.add_argument(
    "--results", type=str, default="results/all_fillers_test_prosody_model.csv"
)
args = parser.parse_args()

df = load_fillers(args.results)


if __name__ == "__main__":
    if "model" not in st.session_state:
        st.session_state.model = load_model()

    if "fillers" not in st.session_state:
        st.session_state.fillers = df
        st.session_state.no_cross = find_no_cross(df)
        st.session_state.global_fig, _ = extract_and_plot_diff_bars(df, plot=False)

        st.session_state.positive = find_difference(df, direction="pos")
        st.session_state.negative = find_difference(df, direction="neg")
        st.session_state.zero = find_difference(df, direction="zero")

        st.session_state.abs_positive = find_difference(
            df, direction="pos", relative=False
        )
        st.session_state.abs_negative = find_difference(
            df, direction="neg", relative=False
        )
        st.session_state.abs_zero = find_difference(
            df, direction="zero", relative=False
        )

    col1, col2 = st.columns(2)

    with col1:
        direction = st.radio(
            "Direction of difference. Positive -> faster shift without filler. (We want positives)",
            options=["pos", "neg", "zero", "no_cross"],
            horizontal=True,
        )
        relative = st.radio(
            "Relative from start of silence. `False` accounts for filler duration i.e. does No-filler make faster shift in absolute terms",
            [True, False],
            horizontal=True,
        )
        smooth = st.radio(
            "Smooth: apply moving average using `N` frames over the prediction probabilities.",
            [0, 5, 10, 15, 20],
            horizontal=True,
        )
    with col2:
        st.pyplot(st.session_state.global_fig)

    if direction == "pos":
        if relative:
            current = st.session_state.positive
        else:
            current = st.session_state.abs_positive
    elif direction == "neg":
        if relative:
            current = st.session_state.negative
        else:
            current = st.session_state.abs_negative
    elif direction == "zero":
        if relative:
            current = st.session_state.zero
        else:
            current = st.session_state.abs_zero
    else:
        current = st.session_state.no_cross

    index = st.number_input(f"Index ({len(current)})", 0, len(current))
    st.title(f"{direction.upper()} (smooth: {smooth})")

    filler = current.iloc[index]
    x, rel_filler_start = load_filler(filler)
    out = st.session_state.model.probs(x.to(st.session_state.model.device))
    fig = plot_filler(
        x,
        out,
        speaker=0 if filler.speaker == "A" else 1,
        rel_filler_start=rel_filler_start,
        filler_dur=filler["duration"],
        smooth=smooth,
    )
    st.pyplot(fig)
    st.subheader("Red line: start of FILLER")
    st.subheader("Red Dashed line: end of FILLER")
