from typing import List, Tuple, Dict, TYPE_CHECKING
import torch
import transformers

from search.tree.tree import Tree
from search.heuristics.heuristic import Heuristic
from utils.paths import TRAINED_MODELS_FOLDER

if TYPE_CHECKING:
    from search.fringe.fringe_item import Step


def chunks(it, n):
    curr = []
    for x in it:
        curr.append(x)
        if len(curr) == n:
            yield curr
            curr = []
    if len(curr) > 0:
        yield curr


class CalibratorHeuristic:

    def __init__(self, model_name, device, goal_conditioned=False):
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(TRAINED_MODELS_FOLDER / model_name)
        self.model.eval()

        if 'cuda' in device and not torch.cuda.is_available():
            device = 'cpu'

        self.model.to(device)

        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(TRAINED_MODELS_FOLDER / model_name)
        self.goal_conditioned = goal_conditioned
        self.batch_size = 4

    def score_steps(
            self,
            tree: Tree,
            steps: List['Step'],
            *args,
            **kwargs
    ):
        step_inputs = [" ".join([tree.get_step_value(x) for x in step.inputs]) for step in steps]
        batches = [
            self.tokenizer(
                chunk_inputs,
                text_pair=[tree.goal]*len(chunk_inputs) if self.goal_conditioned else None,
                truncation=True, padding=True, return_tensors='pt'
            ) for chunk_inputs in chunks(step_inputs, self.batch_size)
        ]

        scores = []

        # with torch.no_grad():
        for batch in batches:
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            logits = self.model(**batch, return_dict=True).logits
            for batch_step in range(logits.shape[0]):
                scores.append(logits[batch_step, 1])
            del logits

        scores = [x.item() for x in scores]

        return scores


class StepTypeModeled(Heuristic):
    """
    Uses a torch model that takes text input and produces a score that can be ranked.
    """

    def __init__(self, torch_device: str = 'cpu', force_new_instance: bool = False, forward_name: str = 'forward_v3_gc', abductive_name: str = 'abductive_gc'):
        super().__init__()

        if hasattr(self, 'instantiated'):
            return

        self.abductive_heuristic_model = CalibratorHeuristic(forward_name, torch_device, goal_conditioned=False)
        self.forward_heuristic_model = CalibratorHeuristic(abductive_name, torch_device, goal_conditioned=True)
        self.instantiated = True

    def __new__(
            cls,
            forward_name: str = 'forward_v3_gc',
            abductive_name: str = 'abductive_gc',
            torch_device: str = 'cpu',
            force_new_instance: bool = False
    ):
        """ creates a singleton object, if it is not created,
        or else returns the previous singleton object"""

        instance_name = f'instance_{forward_name}_{abductive_name}__{torch_device}'
        if force_new_instance:
            return super(StepTypeModeled, cls).__new__(cls)

        if not hasattr(cls, instance_name):
            setattr(cls, instance_name, super(StepTypeModeled, cls).__new__(cls))
        return getattr(cls, instance_name)


    def score_steps(
            self,
            tree: Tree,
            steps: List['Step'],
            *args,
            **kwargs
    ) -> List[float]:

        abductive_steps = []
        abductive_indices = []

        forward_steps = []
        forward_indices = []
        for idx, step in enumerate(steps):
            step_type = step.type.name

            if step_type == 'abductive':
                abductive_steps.append(step)
                abductive_indices.append(idx)
            elif step_type == 'forward':
                forward_steps.append(step)
                forward_indices.append(idx)
            else:
                raise Exception("Unknown step type given to the steptype_modeled heuristic!")

        a_scores = self.abductive_heuristic_model.score_steps(tree, abductive_steps)
        f_scores = self.forward_heuristic_model.score_steps(tree, forward_steps)

        score_and_idx = zip([*a_scores, *f_scores], [*abductive_indices, *forward_indices])
        score_and_idx = list(sorted(score_and_idx, key=lambda x: x[1]))
        scores = [x[0] for x in score_and_idx]
        return scores
