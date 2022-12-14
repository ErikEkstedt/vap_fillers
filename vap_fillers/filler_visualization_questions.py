import streamlit as st

from argparse import ArgumentParser
from os.path import basename, join
from glob import glob

parser = ArgumentParser()
parser.add_argument("--root", type=str, default="results/images/questions_fillers")
args = parser.parse_args()

# sw2657A-ms98-a-0063_F_sw2657A-ms98-a-0016.png

files = list(set([basename(f).split("_")[0] for f in glob(join(args.root, "*.png"))]))
files.sort()


good = ["sw2510-ms98-a-0031", "sw2168B-ms98-a-0046", "sw2510B-ms98-a-0031"]


def find_images(name):
    fillers = glob(join(args.root, f"{name}*"))
    return {"fillers": fillers}


if __name__ == "__main__":

    st.title("Question + Fillers")
    col1, col2 = st.columns(2)
    col1.write(good)
    name = st.selectbox("Question", files)
    img_paths = find_images(name)
    for fp in img_paths["fillers"]:
        st.image(fp)
