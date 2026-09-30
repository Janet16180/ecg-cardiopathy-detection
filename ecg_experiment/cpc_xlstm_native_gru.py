"""Checkpointable native GRU backend for the matched CPC context study."""

from __future__ import annotations

import torch
from torch import nn

from ecg_experiment.cpc import CPCPretrainer
from ecg_experiment.cpc_xlstm import create_model as original_create_model


class NativeGRU(nn.GRU):
    """Run the unchanged GRU outside cuDNN's opaque dropout RNG backend."""

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the original two-layer GRU with native PyTorch dropout state.

        Parameters
        ----------
        input : torch.Tensor
            CNN tokens of shape [batch, time, 256].
        hx : torch.Tensor | None
            Optional initial GRU hidden state.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            GRU contexts and final hidden state with unchanged dimensions.
        """
        with torch.backends.cudnn.flags(enabled=False):
            return super().forward(input, hx)


def create_model(arm: str, seed: int, device: torch.device | str = "cpu") -> CPCPretrainer:
    """Create the matched CPC arm with a native-backend GRU when requested.

    Parameters
    ----------
    arm : str
        ``gru`` or ``xlstm``.
    seed : int
        Frozen initialization seed shared with the original constructor.
    device : torch.device | str
        Target device after CPU construction and weight transfer.

    Returns
    -------
    CPCPretrainer
        Model with unchanged CNN, head and context weight values.
    """
    model = original_create_model(arm, seed, "cpu")
    if arm == "gru":
        context = model.encoder.context
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2)
            replacement = NativeGRU(
                context.input_size, context.hidden_size, num_layers=context.num_layers,
                bias=context.bias, batch_first=context.batch_first, dropout=context.dropout,
                bidirectional=context.bidirectional,
            )
            replacement.load_state_dict(context.state_dict(), strict=True)
        model.encoder.context = replacement
    return model.to(device)
