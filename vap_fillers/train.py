import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# from pytorch_lightning.strategies.ddp import DDPStrategy

from vap.model import VAPModel
from vap_fillers.dataset import SpeechEventDataset


# TODO: validation accuracy
# TODO: validation image
# TODO: Possible (probable) loss-masking


class vapGPT_Laughter(pl.LightningModule):
    def __init__(
        self,
        checkpoint="../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
    ):
        super().__init__()
        self.base = VAPModel.load_from_checkpoint(checkpoint)
        self.laughter_layer = nn.Linear(self.base.conf["model"]["ar"]["dim"], 1)
        self.learning_rate = 3e-4

    def freeze_vap(self):
        for p in self.base.parameters():
            p.requires_grad_(False)
        print(f"Froze VAP-projection-model (self.base)!")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.laughter_layer.parameters(),
            lr=self.learning_rate,
        )
        return {"optimizer": opt}

    def loss_fn(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def forward(self, waveform, attention=False):
        assert (
            waveform.shape[1] == 2
        ), f"Expects 2 channels (B, 2, n_samples) got {waveform.shape}"
        x1 = self.base.net.encoder(waveform[:, :1])  # speaker 1
        x2 = self.base.net.encoder(waveform[:, 1:])  # speaker 2
        x1 = self.base.net.projection(x1)
        x2 = self.base.net.projection(x2)

        # Autoregressive
        o1 = self.base.net.ar_channel(x1, attention=attention)  # ["x"]
        o2 = self.base.net.ar_channel(x2, attention=attention)  # ["x"]

        # New task
        x1, x2 = o1["x"], o2["x"]
        y_hat_1 = self.laughter_layer(x1)
        y_hat_2 = self.laughter_layer(x2)
        return torch.cat([y_hat_1, y_hat_2], dim=-1)

    def training_step(self, batch, batch_idx, **kwargs):
        y_hat = self(batch["waveform"])
        y = batch["label"]
        loss = self.loss_fn(y_hat, y)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", loss, batch_size=batch_size, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **kwargs):
        y_hat = self(batch["waveform"])
        y = batch["label"]
        loss = self.loss_fn(y_hat, y)

        p = (y_hat.sigmoid() >= 0.5).float()

        acc = (p == y).float()[y == 1].sum() / (y == 1).nelement()
        batch_size = batch["waveform"].shape[0]
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        self.log("val_acc", acc, batch_size=batch_size, sync_dist=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_dset = SpeechEventDataset(
        path="results/events/laughter_train.csv",
        window_duration=0.5,
        context_min=3,
        duration=10,
    )
    val_dset = SpeechEventDataset(
        path="results/events/laughter_train.csv",
        window_duration=0.5,
        context_min=3,
        duration=10,
    )
    train_dloader = DataLoader(
        train_dset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_dloader = DataLoader(
        val_dset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Non-strict to not throw error for new layers
    checkpoint = "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"
    model = vapGPT_Laughter(checkpoint)
    model.freeze_vap()

    callbacks = [
        ModelCheckpoint(
            mode="min",
            monitor="val_loss",
            auto_insert_metric_name=False,
            filename="Laughter-epoch{epoch}-val_{val_loss:.2f}",
        )
    ]

    trainer = pl.Trainer(accelerator="gpu", strategy="ddp")
    trainer.fit(model, train_dataloaders=train_dloader, val_dataloaders=val_dloader)

    import matplotlib.pyplot as plt
    from vap_fillers.plot_utils import plot_mel_spectrogram

    checkpoint = "lightning_logs/version_4/checkpoints/epoch=3-step=4620.ckpt"
    model = vapGPT_Laughter.load_from_checkpoint(checkpoint)
    model = model.eval()

    batch = next(iter(val_dloader))

    with torch.no_grad():
        y_hat = model(batch["waveform"])
    probs = y_hat.sigmoid()

    x = torch.arange(probs.shape[1]) / 50

    b = 3
    fig, ax = plt.subplots(4, 1, sharex=True)
    plot_mel_spectrogram(y=batch["waveform"][b], ax=[ax[0], ax[2]])
    ax[0].plot(
        x, batch["label"][b, :, 0] * 79, alpha=0.9, label="Laughter A", color="b"
    )
    ax[1].plot(x, probs[b, :, 0], alpha=0.5, label="Laughter A", color="b")
    ax[1].plot(
        x,
        batch["label"][b, :, 0],
        alpha=0.5,
        label="Label A",
        linestyle="dashed",
        color="b",
    )
    ax[2].plot(
        x, batch["label"][b, :, 1] * 79, alpha=0.9, label="Laughter B", color="orange"
    )
    ax[3].plot(x, probs[b, :, 1], alpha=0.5, label="Laughter B", color="orange")
    ax[3].plot(
        x,
        batch["label"][b, :, 1],
        alpha=0.5,
        label="Label B",
        linestyle="dashed",
        color="orange",
    )
    for a in ax:
        a.legend()
        a.set_xlim([0, x[-1]])
    plt.tight_layout()
    plt.show()
