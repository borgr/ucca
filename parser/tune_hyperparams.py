import sys

import numpy as np

import parser
from config import Config

if __name__ == "__main__":
    out_file = sys.argv.pop(1)
    n = int(sys.argv.pop(1))
    learning_rates = list(set(np.round(0.001 + np.random.exponential(0.8, n), 3)))
    print("All learning rates to try: " + ",".join(
            "%.3f" % learning_rate for learning_rate in sorted(learning_rates)))
    scores = []
    for learning_rate in learning_rates:
        print("Running with learning rate of %.3f" % learning_rate)
        Config().learning_rate = learning_rate
        score = None
        while score is None:
            # noinspection PyBroadException
            try:
                score = parser.main()
            except Exception as e:
                print(e)
        scores.append(score)
        best = np.argmax(scores)
        print("Best learning rate: %f (F1=%f)" % (learning_rates[best], scores[best]))
        with open(out_file, mode="a"):
            print([learning_rate, score], sep=",", file=out_file)
