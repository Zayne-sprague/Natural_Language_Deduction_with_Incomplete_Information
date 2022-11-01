from pathlib import Path

ROOT_PROJECT_DIR = Path(__file__).resolve().parent.parent

ENT_CHECKPOINTS_DIR: Path = ROOT_PROJECT_DIR / 'checkpoints'
ENT_DATASET_DIR = ROOT_PROJECT_DIR / 'datasets'
ENT_OUT_DIR = ROOT_PROJECT_DIR / 'eval_out'
FP_DATASET_ARTIFACTS_DIR = ROOT_PROJECT_DIR / 'fp_dataset_artifacts'
ENT_OUTS_DIR = ROOT_PROJECT_DIR / 'outs'


ENT_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
ENT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
ENT_OUTS_DIR.mkdir(parents=True, exist_ok=True)
