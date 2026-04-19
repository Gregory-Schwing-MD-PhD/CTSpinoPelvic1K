# containers/

`.sif` files produced by `slurm/hpc_pull.sh` land here:

- `ctspinopelvic1k.sif` — lean image, used by Stages 1-3 and all utility scripts
- `ctspinopelvic1k-ts.sif` — CUDA + TotalSegmentator image, used by Stage 4 only

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
