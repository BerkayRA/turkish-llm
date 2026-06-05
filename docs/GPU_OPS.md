# Running on the GPU cluster (NVIDIA Base Command Manager / Bright + Slurm)

How to run the tokenizer A/B (and later the full model) on a remote GPU box or
cluster. NVIDIA **Base Command Manager** (BCM, formerly Bright Cluster
Manager) provisions the cluster and ships a workload manager — almost always
**Slurm**. So the workflow is: get on the **head/login node**, stage data to
**shared storage**, set up the **environment**, then submit **Slurm** jobs that
run on the **GPU compute nodes**.

> These are the standard BCM+Slurm patterns. Your cluster's exact partition
> names, module names, and filesystem paths differ — the **Discover** steps
> below print your actual config; fill the placeholders (`<...>`) from those.

---

## 0. Mental model

- **Login/head node:** where you SSH in, edit code, submit jobs. **Do not train here** — no/shared GPUs, and it's everyone's front door.
- **Compute nodes:** the GPU machines. You reach them only via Slurm (`srun`/`sbatch`), never by training on the login node.
- **Shared filesystem:** a network mount (often `/home`, plus fast scratch like `/scratch`, `/lustre`, or `/cm/shared`) visible on all nodes. Stage the repo + data here so compute nodes can read them.
- **BCM specifics:** `module` (Lmod/environment-modules) for CUDA/compilers; `cmsh` (the Bright/BCM cluster shell) to inspect nodes/partitions; Slurm for jobs.

---

## 1. Get on the cluster & discover its layout

```bash
ssh <user>@<head-node>          # the login node

# --- Discover (run these first; they tell you the real config) ---
sinfo -s                         # partitions, node states, GPU node groups
sinfo -o '%P %G %N %m %c'        # per-partition GRES (gpus), nodes, mem, cpus
scontrol show partition          # partition limits / defaults
module avail 2>&1 | less         # available modules (cuda, cudnn, gcc, anaconda...)
sacctmgr show assoc user=$USER format=Account,Partition,QOS  # what you may submit to
df -h /home /scratch /cm/shared 2>/dev/null   # where the shared/fast storage is
nvidia-smi -L 2>/dev/null || echo "no GPU on login node (expected)"

# --- BCM cluster shell (optional, read-only inspection) ---
cmsh -c 'device list'            # nodes and categories
cmsh -c 'jobqueue list'          # Slurm queues as BCM sees them
```

Write down: the **GPU partition** name (e.g. `gpu`, `a100`, `defq`), the
**GRES** string (e.g. `gpu:a100:8`), and the **fast shared path** for data
(e.g. `/scratch/$USER`). Everything below uses those.

---

## 2. One-time environment setup (on the login node, on shared storage)

Put the project on shared storage so compute nodes see it. The big data
(`data/`, `cache/`, the corpus) is **gitignored** and must be transferred
out-of-band (§3) — git only carries the code.

```bash
cd /scratch/$USER                      # or your shared/fast path
git clone --recurse-submodules https://github.com/BerkayRA/turkish-llm
cd turkish-llm

# CUDA toolchain via modules (names from `module avail`)
module load cuda/<ver> cudnn/<ver>     # e.g. cuda/12.4

# Python env — conda (often preinstalled on BCM) or venv
module load anaconda3 2>/dev/null || true
conda create -y -p ./.cendaenv python=3.12 && conda activate ./.cendaenv
#   (or: python3 -m venv .venv && . .venv/bin/activate)

pip install -r requirements.txt        # sentencepiece, tokenizers
# Torch built for the cluster's CUDA (match the module's CUDA major/minor):
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # run under srun, not login
```

Verify the GPU build **on a compute node**, not the login node:
```bash
srun -p <gpu-partition> --gres=gpu:1 --pty python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 3. Stage data to the cluster (the gitignored artifacts)

The corpus, normalized splits, and the token-ID cache are **not in git**.
Transfer them from your workstation to cluster shared storage with `rsync`
(resumable, only-changed):

```bash
# from your local machine (turkish-llm dir):
rsync -avP data/    <user>@<head-node>:/scratch/$USER/turkish-llm/data/
rsync -avP models/  <user>@<head-node>:/scratch/$USER/turkish-llm/models/
rsync -avP cache/   <user>@<head-node>:/scratch/$USER/turkish-llm/cache/   # if already built
```

Prefer building the **cache on the cluster** (CPU nodes, §6) rather than
shipping it — the `.ids.npy` memmaps can be large. Ship only the small inputs
(`corpus.norm.txt`, tokenizer models) and generate the cache there.

---

## 4. Interactive sanity run (before batch)

Grab one GPU interactively to smoke-test the training entrypoint:
```bash
srun -p <gpu-partition> --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 00:30:00 --pty bash
# now on a GPU node:
conda activate ./.cendaenv   # or . .venv/bin/activate
python train_ab.py --arm U32 --config A --max-steps 50 --smoke   # tiny dry run
exit
```
Confirms data paths, GPU visibility, and that a few steps run before you queue a long job.

---

## 5. Batch training jobs (Slurm `sbatch`)

### 5.1 Single-GPU job
`scripts/train_one.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=ab-U32
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out          # %x=jobname %j=jobid

set -euo pipefail
module load cuda/<ver>
source /scratch/$USER/turkish-llm/.venv/bin/activate   # or conda activate
cd /scratch/$USER/turkish-llm

srun python train_ab.py \
  --arm "${ARM:-U32}" --config A --seed "${SEED:-0}" \
  --protocol match_text --flop-budget 1.32e18 \
  --data data --cache cache --out checkpoints
```
Submit (parameterize via env so one script runs every arm/seed):
```bash
mkdir -p logs
sbatch --export=ALL,ARM=U32,SEED=0 scripts/train_one.sbatch
sbatch --export=ALL,ARM=M8,SEED=0  scripts/train_one.sbatch
```

### 5.2 Multi-GPU on one node (DDP via torchrun) — for Config B / faster runs
```bash
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
...
srun torchrun --standalone --nproc_per_node=8 train_ab.py --arm M8 --config B ...
```
`train_ab.py` must init DDP (`torch.distributed.init_process_group("nccl")`,
shard the dataset by `RANK`/`WORLD_SIZE`, wrap the model in `DistributedDataParallel`).

### 5.3 Multi-node DDP (large runs)
```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
srun torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train_ab.py --arm M8 --config B ...
```
NCCL over the cluster fabric: if there's InfiniBand, set `NCCL_IB_HCA`
appropriately; otherwise `export NCCL_SOCKET_IFNAME=<iface>` (find with `ip a`
on a compute node). `export NCCL_DEBUG=INFO` when debugging hangs.

### 5.4 Parallelize the A/B sweep with a job array
Run all (arm × seed) cells as one array:
```bash
# scripts/sweep.sbatch  — %a indexes a CELL line in scripts/cells.txt
#SBATCH --array=0-5%3        # 6 cells, max 3 concurrent
CELL=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scripts/cells.txt)   # e.g. "U32 0"
read ARM SEED <<< "$CELL"
srun python train_ab.py --arm "$ARM" --seed "$SEED" ...
```

---

## 6. The analyzer pre-tokenization (CPU-bound) on the cluster

The morpheme-BPE cache (`tokenize_corpus.py` / `segment_morphemes.py`) is
CPU-heavy, not GPU. Run it on **CPU partitions** as a job array that shards the
corpus by line range, then concatenate:
```bash
#SBATCH --partition=<cpu-partition>
#SBATCH --cpus-per-task=64
#SBATCH --array=0-15
srun python segment_morphemes.py data/corpus.norm.txt \
  -o cache/M8/shard_${SLURM_ARRAY_TASK_ID}.morph.txt \
  --shard ${SLURM_ARRAY_TASK_ID} --num-shards 16 --workers 64
```
(`segment_morphemes.py` would need a `--shard/--num-shards` flag — a small
addition.) This turns the ~21 CPU-day single-box estimate into hours across
the fleet. Cache once; training reads the `.ids.npy` memmaps.

---

## 7. Monitor, checkpoint, retrieve

```bash
squeue -u $USER                      # your queued/running jobs
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40   # accounting
tail -f logs/ab-U32-<jobid>.out      # live training log
srun --jobid <jobid> --pty nvidia-smi    # peek at GPU util of a running job
scancel <jobid>                      # kill
```
- **Checkpoints** land in `checkpoints/<arm>/<seed>/` on shared storage. The trainer should checkpoint every K steps and at the FLOP-matched stop so a pre-empted job resumes (`--resume`).
- **Wall-time limits:** jobs are killed at `--time`. Make training **resumable** and chain with `sbatch --dependency=afterany:<jobid>` for long runs, or use `--requeue`.
- **Pull results back** to your workstation:
```bash
rsync -avP <user>@<head-node>:/scratch/$USER/turkish-llm/reports/ ./reports/
rsync -avP <user>@<head-node>:/scratch/$USER/turkish-llm/checkpoints/<arm>/ ./checkpoints/<arm>/
```
`reports/` is the version-controlled audit trail — commit the `ab_*`/`bpb_*`/`probe_*` reports after pulling them.

---

## 8. Etiquette & gotchas

- Never run compute on the **login node**; always `srun`/`sbatch`.
- Request only the GPUs/CPUs/mem you need; oversized requests sit in queue.
- Keep big data on **scratch**, not `/home` (quota + slow); scratch may be purged — back up checkpoints.
- Match **torch's CUDA** to the loaded `cuda` module major version.
- **BCM tip:** if `module`/`sinfo` aren't found, source the profile (`source /etc/profile.d/modules.sh`) or check `cmsh -c 'jobqueue list'` to confirm Slurm is the active WLM (some BCM installs use PBS Pro — then translate `sbatch`→`qsub`, `squeue`→`qstat`).
- Set `export HF_HOME=/scratch/$USER/.hf` and `TRANSFORMERS_OFFLINE=1` if pulling any HF assets, to keep them off `/home`.

---

## 9. Minimal end-to-end (once code exists)

```bash
# on cluster shared storage, env active:
python tokenize_corpus.py --arm U32 --split all     # fast
sbatch scripts/segment_array.sbatch                 # M8 analyzer cache (CPU array)
python tokenize_corpus.py --arm M8 --split all      # apply merges -> ids (after segmentation)
sbatch --export=ALL,ARM=U32,SEED=0 scripts/train_one.sbatch
sbatch --export=ALL,ARM=M8,SEED=0  scripts/train_one.sbatch
# after training:
python bpb_eval.py   --arm U32 --ckpt checkpoints/U32/0/flopmatched.pt
python probe_eval.py --scorer model --ckpt checkpoints/M8/0/flopmatched.pt
```
The `scripts/*.sbatch` and the `--shard`/`--resume`/`--scorer model` flags are
part of the Phase 0–2 build (see `docs/EXPERIMENT_AB.md` §6).
