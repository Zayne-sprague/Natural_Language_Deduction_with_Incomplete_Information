import json
import argparse
import random
from itertools import combinations


def get_term_text(ex, inp):
    i = int(inp[1:], base=10)
    k = inp[0]
    if k == "p":
        return ex["premises"][i]
    elif k == "i":
        return ex["intermediates"][i]["output"]
    raise RuntimeError("Unrecognized step input key: " + repr(k))


def get_random_term(dataset, rand):
    ex = rand.choice(dataset)
    i = rand.randrange(len(ex["premises"]) + len(ex["intermediates"]))
    if i >= len(ex["premises"]):
        return ex["intermediates"][i - len(ex["premises"])]["output"]
    else:
        return ex["premises"][i]


def generate_data(input_data, goal_conditioned=False, indirect_goals=False, seed=0):
    rand = random.Random(seed)
    for ex in input_data:
        target_descendants = {}
        all_terms = set(ex["premises"])
        all_terms.update(step["output"] for step in ex["intermediates"])
        for i, step in enumerate(ex["intermediates"]):
            k = f"i{i}"
            descendants = target_descendants.get(k)
            if descendants is None:
                target_descendants[k] = descendants = set()
            for inp in step["inputs"]:
                descendants.add(get_term_text(ex, inp))
                if inp.startswith("i"):
                    descendants.update(target_descendants[inp])
        for i, step in enumerate(ex["intermediates"]):
            k = f"i{i}"
            step_output = step["output"]
            descendants = target_descendants[k]
            successors = [step_output]
            for j in range(i + 1, len(ex["intermediates"])):
                k2 = f"i{j}"
                if step_output in target_descendants[k2]:
                    successors.append(ex["intermediates"][j]["output"])

            true_inputs = [get_term_text(ex, inp) for inp in step["inputs"]]
            rand.shuffle(true_inputs)
            pos_ex = {"input_text": " ".join(true_inputs), "label": 1}

            negative_input_options = all_terms - descendants
            negative_input_options.difference_update(successors)
            negative_input_count = rand.randint(1, len(true_inputs))
            while len(negative_input_options) < negative_input_count:
                # WARNING: this is a kludge - spins forever if len(input_data) == 1
                random_term = get_random_term(input_data, rand)
                if random_term not in descendants:
                    negative_input_options.add(random_term)
            negative_inputs = rand.sample(
                list(descendants)
                if goal_conditioned and indirect_goals
                else true_inputs,
                len(true_inputs) - negative_input_count,
            )
            negative_inputs.extend(
                rand.sample(negative_input_options, negative_input_count)
            )
            rand.shuffle(negative_inputs)
            neg_ex = {"input_text": " ".join(negative_inputs), "label": 0}

            if goal_conditioned:
                if indirect_goals:
                    pos_ex["goal"] = rand.choice(successors)
                else:
                    pos_ex["goal"] = step_output
                neg_ex["goal"] = step_output

            yield pos_ex
            yield neg_ex


# TODO - for negative and positive examples just replace one of the input sentences with the output sentence and set the
#   the output sentence as the other input.
# TODO - maybe refactor this  p0 + p1 is a forward positive example, p0 + i0 is a abductive positive example
#   if p0 is never results from i10 then it's a negative example (i.e. from a different tree) premise[sep]conclusion
#   anything outside that subtree is a good negative example.
# TODO - Forward step: x1 + x2 -> y | Abductive step: x2 <- y - x1 or y / x1
# TODO - separate model for the Abductive heuristic. (We could think about training a joint model, might not be useful)
def generate_abductive_data(
    input_data, goal_conditioned=False, indirect_goals=False, seed=0
):
    rand = random.Random(seed)
    for ex in input_data:
        target_descendants = {}
        all_terms = set(ex["premises"])
        all_terms.update(step["output"] for step in ex["intermediates"])
        for i, step in enumerate(ex["intermediates"]):
            k = f"i{i}"
            descendants = target_descendants.get(k)
            if descendants is None:
                target_descendants[k] = descendants = set()
            for inp in step["inputs"]:
                descendants.add(get_term_text(ex, inp))
                if inp.startswith("i"):
                    descendants.update(target_descendants[inp])
        for i, step in enumerate(ex["intermediates"]):
            k = f"i{i}"
            step_output = step["output"]
            descendants = target_descendants[k]
            successors = [step_output]
            for j in range(i + 1, len(ex["intermediates"])):
                k2 = f"i{j}"
                if step_output in target_descendants[k2]:
                    successors.append(ex["intermediates"][j]["output"])

            true_inputs = [get_term_text(ex, inp) for inp in step["inputs"]]
            rand.shuffle(true_inputs)

            if len(true_inputs) > 1:
                positive_inputs = [
                    [*list(set(true_inputs) - {x}), step_output] for x in true_inputs
                ]
            else:
                positive_inputs = [[*true_inputs, step_output]]

            pos_exs = [{"input_text": " ".join(x), "label": 1} for x in positive_inputs]

            if goal_conditioned:
                for i in range(len(pos_exs)):
                    if indirect_goals:
                        pos_exs[i]["goal"] = rand.choice(successors)
                    else:
                        pos_exs[i]["goal"] = step_output

            negative_input_options = all_terms - descendants
            negative_input_options.difference_update(successors)
            negative_input_count = rand.randint(1, max(1, len(true_inputs) - 1))
            while len(negative_input_options) < negative_input_count:
                # WARNING: this is a kludge - spins forever if len(input_data) == 1
                random_term = get_random_term(input_data, rand)
                if random_term not in descendants:
                    negative_input_options.add(random_term)

            negative_inputs = rand.sample(
                list(descendants)
                if goal_conditioned and indirect_goals
                else true_inputs,
                max([(len(true_inputs) - 1) - negative_input_count, 0]),
            )

            # negative_inputs = []
            neg_samples = list(
                combinations(negative_input_options, negative_input_count)
            )

            neg_inputs = [[*list(negative_inputs), *x] for x in neg_samples]
            for i in range(len(neg_inputs)):
                rand.shuffle(neg_inputs[i])

            neg_exs = [
                {"input_text": " ".join([*x, step_output]), "label": 0}
                for x in neg_inputs
            ]

            if goal_conditioned:
                for i in range(len(neg_exs)):
                    neg_exs[i]["goal"] = step_output

            for i in range(min([len(pos_exs), len(neg_exs)])):
                pos_ex = pos_exs[i]
                neg_ex = neg_exs[i]

                if len(pos_ex["input_text"].split(".")) != len(
                    neg_ex["input_text"].split(".")
                ):
                    print("hi")

                yield pos_ex
                yield neg_ex
