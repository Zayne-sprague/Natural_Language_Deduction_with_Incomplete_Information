from typing import List, Dict, Iterable, Tuple, ClassVar, Union
from copy import deepcopy
from functools import partial

from search.tree.step_generation import StepGeneration
from utils.search_utils import normalize


class Tree:
    goal: str

    premises: List[str]

    hypotheses: List[StepGeneration]
    intermediates: List[StepGeneration]

    missing_premises: List[str]  # Intentionally removed or masked premises

    __original_premises__: List[str]  # List of original premises (self.premises + self.missing_premises)

    distractor_premises: List[str]

    input: List[str]
    output: List[str]

    def __init__(
            self,
            goal: str,
            premises: List[str] = (),
            hypotheses: List[StepGeneration] = (),
            intermediates: List[StepGeneration] = ()
    ):
        """
        This class stores generations that can be combined or represented as a "natural language proof" towards a goal
        statement.

        :param goal: The statement you are trying to entail or prove
        :param premises: List of the string premises for the tree (no particular order)
        :param hypotheses: Abductions that have been done starting with the Goal Node (G - P0 -> Hypothesis 1 etc.)
        :param intermediates: Forward step generations given the premises (P0 + P1 -> Intermediate 1 etc.)
        """

        self.goal = goal

        self.premises = premises
        self.__original_premises__ = deepcopy(premises)

        self.hypotheses = hypotheses
        self.intermediates = intermediates

        self.missing_premises = []
        self.distractor_premises = []

    # TODO - these properties should be reversed --- normalize the raw data as it changes and cache the normalized vals.
    @property
    def normalized_premises(self):
        return [normalize(x) for x in self.premises]

    @property
    def normalized_intermediates(self):
        return [normalize(x.output) for x in self.intermediates]

    @property
    def normalized_hypotheses(self):
        return [normalize(x.output) for x in self.hypotheses]

    @property
    def normalized_missing_premises(self):
        return [normalize(x) for x in self.missing_premises]

    @property
    def normalized_goal(self):
        return normalize(self.goal)

    @classmethod
    def from_json(cls, tree_json: Dict[str, any]) -> ClassVar['Tree']:
        """Constructor for building trees from a json structure"""

        if 'goal' not in tree_json:
            return cls.from_canonical_json(tree_json)

        goal = tree_json['goal']
        hypotheses = [StepGeneration.from_json(x) for x in tree_json['hypotheses']]
        premises = tree_json['premises']
        intermediates = [StepGeneration.from_json(x) for x in tree_json['intermediates']]

        tree = cls(goal=goal, premises=premises, hypotheses=hypotheses, intermediates=intermediates)

        if 'missing_premises' in tree_json:
            tree.missing_premises = tree_json['missing_premises']
        if 'original_premises' in tree_json:
            tree.__original_premises__ = tree_json['original_premises']
        if 'distractor_premises' in tree_json:
            tree.distractor_premises = tree_json['distractor_premises']

        return tree

    @classmethod
    def from_canonical_json(cls, canonical_json: Dict[str, any]) -> ClassVar['Tree']:
        """
        Canonical tree formats have 1 hypothesis, multiple premises, and multiple intermediates.

        This datastructure, however, would call that one hypothesis a Goal and the first Hypothesis in the hypotheses
        array.
        """

        goal = canonical_json['hypothesis']
        hypotheses = [goal]
        premises = canonical_json['premises']
        intermediates = [StepGeneration.from_json(x) for x in canonical_json['intermediates']]

        return cls(goal=goal, premises=premises, hypotheses=hypotheses, intermediates=intermediates)

    def to_json(self) -> Dict[str, any]:
        """Serializer function to convert this class instance into primitive objects"""

        if len(self.hypotheses) > 0 and isinstance(self.hypotheses[0], str):
            # If we are in the canonical variation of the tree class, return the canonical version of the to_json func.
            return self.to_canonical_json()

        return {
            'goal': self.goal,
            'hypotheses': [x.to_json() for x in self.hypotheses],
            'premises': self.premises,
            'intermediates': [x.to_json() for x in self.intermediates],
            'missing_premises': self.missing_premises,
            'original_premises': self.__original_premises__,
            'distractor_premises': self.distractor_premises,
        }

    def to_canonical_json(self) -> Dict[str, any]:
        """Helper for converting to the original 'tree' format"""

        return {
            'hypothesis': self.goal,
            'premises': self.premises,
            'intermediates': [x.to_json() for x in self.intermediates]
        }

    def __len__(self):
        return len(self.intermediates) + len(self.hypotheses)

    def mask_premise(self, idx: int):
        """Remove a premise from the premises list and place it in the missing_premises list."""

        # TODO - make this reversible
        self.missing_premises.append(self.premises[idx])
        self.premises.remove(self.premises[idx])

        # Update all the intermediate steps to reflect the new masked tag.
        for intermediate_idx, step in enumerate(self.intermediates):
            for input_idx, inp in enumerate(step.inputs):
                inp_key = self.get_step_key(inp)
                if inp_key == 'p':
                    premise_idx = int(inp[1:])
                    if premise_idx == idx:
                        step.inputs[input_idx] = f'm{len(self.missing_premises) - 1}'
                        self.intermediates[intermediate_idx] = step
                    elif premise_idx > idx:
                        step.inputs[input_idx] = f'p{premise_idx - 1}'
                        self.intermediates[intermediate_idx] = step

    def bridge_intermediate_to_hypotheses(self, intermediate_idx: int, hypothesis_idx: int):
        """
        This function will bridge the intermediate at the specified index with the hypothesis at the specified
        index, then for every hypothesis used as an input for the given hypothesis index will be turned into an
        intermediate.  The result of this function is an end to end path from the intermediate index to the goal
        statement TODO - this results in your hypotheses array & non-essential intermediates being wiped

        :param intermediate_idx: The intermediate index that equals the hypothesis
        :param hypothesis_idx: the hypothesis index that equals the intermediate
        """
        assert len(self.intermediates) > intermediate_idx and len(self.hypotheses) > hypothesis_idx, \
            'Intermediate and/or Hypothesis idx out of bounds.'

        self.__bridge_intermediate_to_hypotheses__(intermediate_idx, hypothesis_idx)

        self.reduce_to_intermediate_subtree(len(self.intermediates) - 1)

    def __bridge_intermediate_to_hypotheses__(self, intermediate_idx: int, hypothesis_idx: int):
        """Helper recursive function for the bridge_intermediate_to_hypotheses function."""

        hypothesis = self.hypotheses[hypothesis_idx]

        new_step = StepGeneration(
            output=self.get_step_value(hypothesis.inputs[-1]),
            inputs=[*hypothesis.inputs[0:-1], f'i{intermediate_idx}'],
            tags={'step_type': 'bridge'},
        )

        self.intermediates.append(new_step)

        if self.get_step_key(hypothesis.inputs[-1]) == 'h':
            self.__bridge_intermediate_to_hypotheses__(len(self.intermediates) - 1, int(hypothesis.inputs[-1][1:]))

    def hypothesis_to_intermediates(self, hypothesis_idx: int) -> 'Tree':
        """
        Given a hypothesis, convert it to a "generated premise" and then resolve the rest of the tree so that all the
        hypothesis inputs are forward step generations towards the goal.  This will return a reduced the tree which only
        contains a set of intermediates that lead directly to the goal + 1 additional premise that is flagged as a
        generated premise.

        :param hypothesis_idx: The hypothesis we want to convert to a premise
        :return: A reduced tree with a set of intermediates that lead to the goal and a new generated premise from the
            given hypothesis index.
        """

        tree_copy = deepcopy(self)

        hypothesis = tree_copy.get_step(f'h{hypothesis_idx}')
        tree_copy.premises.append(hypothesis.output)
        new_step = StepGeneration(
            inputs=[*hypothesis.inputs[:-1], f'p{len(tree_copy.premises) - 1}'],
            output=tree_copy.get_step_value(hypothesis.inputs[-1])
        )

        tree_copy.intermediates.append(new_step)

        if 'h' in hypothesis.inputs[-1]:
            tree_copy.__bridge_intermediate_to_hypotheses__(len(tree_copy.intermediates) - 1, int(hypothesis.inputs[-1][1:]))

        tree_copy.reduce_to_intermediate_subtree(len(tree_copy.intermediates) - 1)
        return tree_copy

    def get_subtree(self, root_intermediate_idx: int, depth: int = -1) -> List[StepGeneration]:
        """
        Given a root intermediate index, find all the intermediate step generations below the given index in order
        and return them in an array.  It will also relabel each intermediate so that the subtree returned is exactly
        like a normal Tree object.

        :param root_intermediate_idx: The top level intermediate you want the subtree for
        """

        intermediate_keys = list(set(self.__get_subtree__(root_intermediate_idx)))
        if depth > -1:
            intermediate_keys = intermediate_keys[0:min(depth+1, len(intermediate_keys))]

        intermediate_indices = [int(x[1:]) for x in intermediate_keys]

        intermediates = sorted(zip(intermediate_keys, intermediate_indices), key=lambda x: x[1])

        subtree = []
        intermediate_map = {}

        for (intermediate_key, idx) in intermediates:
            new_intermediate_index = intermediate_map.get(idx, len(subtree))
            intermediate_map[idx] = new_intermediate_index

            intermediate = deepcopy(self.intermediates[idx])
            for input_idx, x in enumerate(intermediate.inputs):
                if self.get_step_key(x) == 'i':
                    intermediate.inputs[input_idx] = f'i{intermediate_map[int(x[1:])]}'

            subtree.append(intermediate)
        return subtree

    def __get_subtree__(self, intermediate: int) -> List[str]:
        """
        Helper function that gets the subtree of intermediate step_keys ([i4, i3, i0, i1]) etc.

        This can contain duplicates, no filtering is applied -- it's up to the callee to interpret the subtree.
        """

        root_intermediate = self.intermediates[intermediate]

        subtree = []
        for x in root_intermediate.inputs:
            step_key = self.get_step_key(x)

            if step_key == 'i':
                subtree.extend(self.__get_subtree__(int(x[1:])))


        return [f'i{intermediate}', *subtree]

    def get_depth(self):
        last_int = self.intermediates[-1]

        return self.__get_depth__(last_int)

    def __get_depth__(self, step: StepGeneration):
        arg_depths = []
        for arg in step.inputs:
            if 'i' in arg:
                arg_depths.append(self.__get_depth__(self.get_step(arg)))
            else:
                arg_depths.append(0)

        return 1 + max(arg_depths)


    def slice(self, root_intermediate: int, depth: int = -1) -> 'Tree':
        root = self.get_step(f'i{root_intermediate}')
        new_tree = Tree(goal=root.output, premises=[], intermediates=[])

        self.__slice__(new_tree, root, depth, 0)
        return new_tree

    def __slice__(self, tree: 'Tree', step: StepGeneration, depth: int, current_depth: int = 0):
        if depth == current_depth:
            intermediates = [x.output for x in tree.intermediates]
            if step.output in intermediates:
                return f'i{intermediates.index(step.output)}'

            new_args = []
            for arg in step.inputs:
                val = self.get_step_value(arg)

                if val not in tree.premises:
                    new_args.append(f'p{len(tree.premises)}')
                    tree.premises.append(val)
                else:
                    new_args.append(f'p{tree.premises.index(val)}')

            tree.intermediates.append(StepGeneration(inputs=new_args, output=step.output))
            return f'i{len(tree.intermediates) - 1}'

        intermediates = [x.output for x in tree.intermediates]
        if step.output in intermediates:
            return f'i{intermediates.index(step.output)}'

        new_args = []
        for arg in step.inputs:

            if 'i' in arg:
                new_args.append(self.__slice__(tree, self.get_step(arg), depth, current_depth+1))
            elif 'p' in arg or 'm' in arg:
                if self.get_step_value(arg) not in tree.premises:
                    tree.premises.append(self.get_step_value(arg))
                    new_args.append(f'p{len(tree.premises) - 1}')
                else:
                    new_args.append(f'p{tree.premises.index(self.get_step_value(arg))}')

        tree.intermediates.append(StepGeneration(inputs=new_args, output=step.output))
        return f'i{len(tree.intermediates) - 1}'

    def __build_from_intermediate__(
            self,
            intermediate: List[Union[str, List]],
            tree: 'Tree'
    ) -> List[StepGeneration]:
        root = intermediate[0]

        steps = []
        inputs = []

        for arg in intermediate[1:]:
            if isinstance(arg, list):
                steps.extend(self.__build_from_intermediate__(arg, tree))
                inputs.append(f'i{len(steps) - 1}')
            else:
                val = self.get_step_value(arg)
                if val not in tree.premises:
                    tree.premises.append(val)
                inputs.append(f'p{tree.premises.index(val)}')

        original_intermediate = self.intermediates[int(root[1:])]
        step = StepGeneration(
            inputs=inputs,
            output=original_intermediate.output,
        )
        steps.append(step)

        return steps

    def reduce_to_intermediate_subtree(self, intermediate_idx: int):
        """
        Makes the current tree only have steps that lead to the given intermediate_idx, everything else is removed.

        :param intermediate_idx: Index of the intermediate the tree should put as it's root
        :return:
        """

        # Get the subtree of intermediate steps from the step that matched the goal
        subtree = self.get_subtree(intermediate_idx)

        # Set all the intermediates of the tree to only those that lead to directly matching the goal
        self.intermediates = subtree

        # For each intermediate, swap hypotheses to premises.
        for iidx, intermediate in enumerate(self.intermediates):
            inputs = intermediate.inputs

            for input_idx, input in enumerate(inputs):
                if 'h' in input:
                    val = self.get_step_value(input)
                    self.intermediates[iidx].inputs[input_idx] = f'p{len(self.premises)}'
                    self.premises.append(val)

        # Remove the hypotheses.
        self.hypotheses = []

    def get_step_value(self, step: str) -> str:

        key = self.get_step_key(step)
        if key == 'g':
            return self.goal

        idx = int(step[1:])
        if key == 'h':
            return self.hypotheses[idx].output
        if key == 'p':
            return self.premises[idx]
        if key == 'i':
            return self.intermediates[idx].output
        if key == 'm':
            return self.missing_premises[idx]

    def get_step(self, step: str) -> Union[StepGeneration, str]:

        key = self.get_step_key(step)
        if key == 'g':
            return self.goal

        idx = int(step[1:])
        if key == 'h':
            return self.hypotheses[idx]
        if key == 'p':
            return self.premises[idx]
        if key == 'i':
            return self.intermediates[idx]
        if key == 'm':
            return self.missing_premises[idx]

    def get_normalized_step_value(self, step: str) -> str:
        key = self.get_step_key(step)
        if key == 'g':
            return self.normalized_goal

        idx = int(step[1:])
        if key == 'h':
            return self.normalized_hypotheses[idx].output
        if key == 'p':
            return self.normalized_premises[idx]
        if key == 'i':
            return self.normalized_intermediates[idx].output
        if key == 'm':
            return self.missing_premises[idx]

    def get_step_key(self, step: str) -> str:
        return step[0]
