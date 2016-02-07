import os
from _operator import attrgetter

import numpy as np

import parser
from config import Config


class Hyperparams(object):
    def __init__(self, learning_rate, decay_factor):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.score = -float("inf")

    def run(self):
        assert Config().args.train and Config().args.passages or Config().args.folds, \
            "insufficient parameters given to parser"
        print("Running with %s" % self)
        Config().learning_rate = self.learning_rate
        Config().decay_factor = self.decay_factor
        self.score = parser.main()
        assert self.score is not None, "parser failed to produce score"

    def __str__(self):
        ret = "learning rate: %.3f" % self.learning_rate
        ret += ", decay factor: %.3f" % self.decay_factor
        if self.score > -float("inf"):
            ret += ", score: %.3f" % self.score
        return ret

    def print(self, file):
        print(", ".join("%.3f" % p for p in
                        [self.learning_rate, self.decay_factor, self.score]),
              file=file)

    @staticmethod
    def print_title(file):
        print("learning rate, decay factor, score", file=file)


def main():
    out_file = os.environ.get("HYPERPARAMS_FILE", "hyperparams.csv")
    num = int(os.environ.get("HYPERPARAMS_NUM", 30))
    dims = np.tile(np.sqrt(num + 1), 2)
    hyperparams = list(set(
        Hyperparams(learning_rate, decay_factor)
        for learning_rate, decay_factor in np.round(0.001 + np.random.exponential(0.8, dims), 3)
    ))
    print("\n".join(["All hyperparam combinations to try: "] +
                    [str(h) for h in hyperparams]))
    print("Saving results to '%s'" % out_file)
    with open(out_file, "w") as f:
        Hyperparams.print_title(f)
    for hyperparam in hyperparams:
        hyperparam.run()
        with open(out_file, "a") as f:
            hyperparam.print(f)
        best = max(hyperparams, key=attrgetter("score"))
        print("Best hyperparams: %s" % best)


if __name__ == "__main__":
    main()
