import argparse
from dataclasses import dataclass
from typing import Tuple
import sys
import json

import numpy as np
import pandas as pd

from median_method import MedianMethod


@dataclass
class CustomParameters:
    neighbourhood_size: int = 100
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)

def read_csv_in_batches(filepath, batch_size):
    iterator = pd.read_csv(filepath, chunksize=batch_size)
    
    for batch in iterator:
        yield batch["value"].values


def execute(config):
    set_random_state(config)
    mm = MedianMethod(neighbourhood_size=config.customParameters.neighbourhood_size)

    for batch in read_csv_in_batches(config.dataInput, 1000):
        scores = mm.fit_predict(batch)
        with open(config.dataOutput, 'a') as out_f:
            np.savetxt(out_f, scores)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()

    if config.executionType == "train":
        print("Nothing to train, finished!")
        exit(0)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected 'execute'!")
