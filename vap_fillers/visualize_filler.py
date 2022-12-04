from os.path import basename, join

import streamlit as st

from glob import glob


ROOT = "results/fillers"
FIG_ROOT = join(ROOT, "figures")

figure_paths = glob(join(FIG_ROOT, "*.png"))
figure_paths.sort()
figs = {basename(path).split("-")[1]: path for path in figure_paths}

sorted_idx = [int(s) for s in list(figs.keys())]
sorted_idx.sort()
# sorted_idx = [str(s).zfill(4) for s in sorted_idx]


if __name__ == "__main__":
    st.title(f"{ROOT}")

    value = st.number_input("filler index", 0, sorted_idx[-1], value=0)
    omit_path = figs[str(value).zfill(4)]
    filler_path = omit_path.replace("_omit.png", ".png")

    st.image(filler_path)
    st.image(omit_path)

    # st.header("BAD")
    # value = st.number_input("filler index", 0, sorted_idx[-1], value=0)
    # omit_path = figs[str(value).zfill(4)]
    # filler_path = omit_path.replace("_omit.png", ".png")
    # st.image(filler_path)
    # st.image(omit_path)
