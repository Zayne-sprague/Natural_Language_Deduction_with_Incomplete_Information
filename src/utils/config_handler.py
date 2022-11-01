from yaml import safe_load
from pathlib import Path
from argparse import Namespace
from typing import List, Union


def merge_yaml_and_namespace(
        yaml: Path,
        args: Namespace,
        scopes: Union[List[str], str] = None,
        default_scope: str = None,
        favor_namespace: bool = True,
) -> Namespace:
    """
    Given a path to a yaml config file and a Namespace, we can load the Yaml file and merge its parameters
    with the Namespace (favoring one or the other if a param is specified in both.)

    Warning: Favoring the yaml will ignore all command line arguments given in the Namespace that are also specified
    in the yaml file, AND favoring the namespace will wipe out any Yaml params (so make sure you don't have
    defaults set in the argparser arguments!)

    :param yaml: Path to the yaml file we want to load in
    :param args: Namespace with the arguments we want to merge
    :param scopes: The dot path to the scope of parameters in the Yaml file, ex: abductive_only.search
    :param default_scope: The path to the default scope (usually set by a script and not from an CLI arg.)
    :param favor_namespace: If a param is found in both the namespace and the yaml file, take the Namespaces value.
    """

    with yaml.open('r') as file:
        config: Dict[str, any] = safe_load(file)

    # Set up default parameters
    parameters: Dict[str, any] = {}
    if default_scope:
        path = default_scope.split(".")
        scoped_parameters: Dict[str, any] = config
        for p in path:
            scoped_parameters = scoped_parameters.get(p, {})
        parameters.update({**scoped_parameters})

    # Set up parameters per given scope
    if scopes:
        for scope in scopes:
            path = scope.split(".")

            scoped_parameters: Dict[str, any] = config
            for p in path:
                scoped_parameters = scoped_parameters.get(p, {})
            parameters.update({**scoped_parameters})

    # Update the namespace with the new parameters.
    for key, value in parameters.items():
        try:
            defined_in_namespace = args.__getattribute__(key) is not None
            if defined_in_namespace and favor_namespace:
                continue
        except AttributeError as e:
            pass

        args.__setattr__(key, value)

    return args
