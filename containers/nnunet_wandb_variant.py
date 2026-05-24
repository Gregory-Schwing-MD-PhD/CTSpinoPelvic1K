"""
nnU-Net v2 trainer shim — INFERENCE ONLY.

The CTSpinoPelvic1K pseudo-label checkpoints (Dataset803, model folder
`nnUNetTrainerWandB_500ep_LSTVOversample__nnUNetResEncUNetPlans_100G__3d_fullres`)
were trained under a CUSTOM trainer class of that name. At predict time
`nnUNetv2_predict` locates the trainer BY NAME and calls its
`build_network_architecture()` — a @staticmethod inherited from nnUNetTrainer
that builds the network purely from the plans — *before* loading the
checkpoint weights. The trainer is never instantiated for inference, so
`__init__` never runs and none of the training-time machinery (W&B logging,
LSTV queue oversampling, the patch-bias dataloader, CE reweighting) is
exercised. Only the class NAME and its inherited architecture builder matter.

So this ~10-line shim reproduces inference EXACTLY — and lets
slurm/pseudolabel.sh bind a file that lives in THIS repo, instead of depending
on the spinesurg-ct-nnunet training repo being checked out alongside. If the
architecture ever diverged, checkpoint loading would fail with a state_dict
size mismatch (loud), never produce subtly different masks (silent).

The FULL training trainer (2700 lines, the source of the trained weights)
lives at spinesurg-ct-nnunet/tools/nnunet_wandb_variant.py — use that to
retrain, not this.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandB_500ep_LSTVOversample(nnUNetTrainer):
    """Inference shim — identical network to the trained model, since
    build_network_architecture is inherited from nnUNetTrainer and reads the
    plans. Only the class name has to match what the checkpoint recorded."""
    pass
