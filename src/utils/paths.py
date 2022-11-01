from pathlib import Path

ROOT_FOLDER = Path(__file__).parent.parent.parent.resolve()
SRC_FOLDER = ROOT_FOLDER / 'src'
DATA_FOLDER = ROOT_FOLDER / 'data'
DATA_STEP_FOLDER = DATA_FOLDER / 'step'
DATA_FULL_FOLDER = DATA_FOLDER / 'full'
ENTAILMENT_BANK_FOLDER = DATA_FOLDER / 'full/entailmentbank'
ENWN = DATA_FOLDER / 'full/enwn'

OUTPUT_FOLDER = ROOT_FOLDER / 'output'
CONFIGS_FOLDER = ROOT_FOLDER / 'configs'
TRAINED_MODELS_FOLDER = ROOT_FOLDER / 'trained_models'

SEARCH_FOLDER = SRC_FOLDER / 'search'
SCRIPTS_FOLDER = SEARCH_FOLDER / 'scripts'
