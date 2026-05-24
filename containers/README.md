# containers/

`.sif` files produced by `slurm/hpc_pull.sh` land here:

- `ctspinopelvic1k.sif` — lean image, used by Stages 1-3 and all utility scripts
- `ctspinopelvic1k-ts.sif` — CUDA + TotalSegmentator image, used by Stage 4 only

Also here (tracked in git, not pulled):

- `nnunet_wandb_variant.py` — a ~10-line nnU-Net trainer **inference shim**.
  `slurm/pseudolabel.sh` binds it into the nnU-Net container so `nnUNetv2_predict`
  can resolve the `nnUNetTrainerWandB_500ep_LSTVOversample` class the Dataset803
  checkpoints were trained under. It reproduces inference exactly (the trainer is
  never instantiated for predict) without depending on the `spinesurg-ct-nnunet`
  training repo being checked out. See the file's docstring.

To populate:

```bash
# On HPC:
sbatch slurm/hpc_pull.sh
# or, interactively on a login node:
make hpc-pull-now
```

To rebuild upstream images from the Dockerfiles (run on a workstation that has Docker + docker login, NOT on HPC):

```bash
DOCKERHUB_USER=yourhubuser make docker-push
```
