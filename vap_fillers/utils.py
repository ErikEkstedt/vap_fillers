import torch
from vap.model import VAPModel
from textgrids import TextGrid

from vap.utils import everything_deterministic

everything_deterministic()
torch.manual_seed(0)

CHECKPOINT = "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"


def load_model(checkpoint=CHECKPOINT):
    print("Load Model...")
    model = VAPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def wav_path_to_tg_path(wavpath):
    return wavpath.replace("/Stimuli/", "/alignment/").replace(".wav", ".TextGrid")


def pad_silence(waveform, silence=10, sample_rate=16_000, sil_samples=None):
    assert (
        waveform.ndim == 3
    ), f"Expects waveform of shape (B, C, n_samples) but got {waveform.shape}"

    if sil_samples is None:
        sil_samples = int(silence * sample_rate)
    B, C, _ = waveform.size()
    z = torch.zeros((B, C, sil_samples), device=waveform.device)
    return torch.cat([waveform, z], dim=-1)


def read_text_grid(path: str) -> dict:
    grid = TextGrid(path)
    data = {"words": [], "phones": []}
    for word_phones, vals in grid.items():
        for w in vals:
            if w.text == "":
                continue
            # what about words spoken multiple times?
            # if word_phones == 'words':
            #     data[word_phones][w.text] = (w.xmin, w.xmax)
            data[word_phones].append((w.xmin, w.xmax, w.text))
    return data


@torch.no_grad()
def moving_average(x, k=10):
    shape_back = False
    if x.ndim == 2:
        x = x.unsqueeze(1)
        shape_back = True

    # Conv average
    m = torch.nn.Conv1d(
        in_channels=1, out_channels=1, kernel_size=k, bias=False, padding=0
    )
    m.weight.data.fill_(1.0 / k)
    xx = torch.nn.functional.pad(x, pad=(k - 1, 0), mode="replicate")
    y = m(xx)

    if shape_back:
        y = y.squeeze(1)
    return y
