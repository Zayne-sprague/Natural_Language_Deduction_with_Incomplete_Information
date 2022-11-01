from typing import List, Dict


class StepGeneration:
    """Class to help with the structure of intermediate steps."""

    inputs: List[str]
    output: str
    scores: Dict[str, any]
    annotations: Dict[str, any]

    def __init__(
            self,
            inputs: List[str],
            output: str = None,
            scores: Dict[str, any] = None,
            annotations: Dict[str, any] = None,
            tags: Dict[str, any] = None,
    ):
        """
        :param inputs: Inputs used to generate the output
        :param output: The output or generation of the given inputs
        :param scores: Stores the scores of the step used usually for evaluation (not for searching)
        :param annotations: Stores the manually created annotations for a step generation
        :param tags: Helpful tags that can be created for a step generation (i.e. {step_type: 'forward'} etc.)
        """

        self.inputs = inputs
        self.output = output
        self.scores = scores if scores else {}
        self.annotations = annotations if annotations else {}
        self.tags = tags if tags else {}

    def to_json(self) -> Dict[str, any]:
        """Helper to convert intermediates into their json structure"""

        out = {'output': self.output, 'inputs': self.inputs}
        if len(self.scores) > 0:
            out["scores"] = self.scores
        if len(self.annotations) > 0:
            out['annotations'] = self.annotations
        if len(self.tags) > 0:
            out['tags'] = self.tags

        return out

    @classmethod
    def from_json(cls, ex: Dict[str, any]) -> 'Intermediate':
        """Helper to make an intermediate instance from their json structure"""

        inputs = ex['inputs']
        output = ex['output']
        scores = ex.get("scores", None)
        annotations = ex.get("annotations", None)
        tags = ex.get("tags", None)
        instance = cls(inputs, output, scores, annotations, tags)
        return instance

    def __repr__(self):
        return self.output
