import argparse


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance


class Config(object, metaclass=Singleton):
    def __init__(self):
        argparser = argparse.ArgumentParser(description="""Transition-based parser for UCCA.""")
        argparser.add_argument("passages", nargs="*", default=[],
                               help="passage files/directories to test on/parse")
        argparser.add_argument("-t", "--train", nargs="+", default=[],
                               help="passage files/directories to train on")
        argparser.add_argument("-d", "--dev", nargs="+", default=[],
                               help="passage files/directories to tune on")
        argparser.add_argument("-m", "--model", default=None, help="model file to load/save")
        argparser.add_argument("-o", "--outdir", default=".", help="output directory")
        argparser.add_argument("-p", "--prefix", default="ucca_passage", help="output filename prefix")
        argparser.add_argument("-L", "--log", default="parser.log", help="output log file")
        argparser.add_argument("-I", "--iterations", type=int, default=1,
                               help="number of training iterations")
        argparser.add_argument("-b", "--binary", action="store_true", default=False,
                               help="read and write passages in Pickle binary format, not XML")
        argparser.add_argument("-v", "--verbose", action="store_true", default=False,
                               help="display detailed information while parsing")
        argparser.add_argument("-s", "--sentences", action="store_true", default=False,
                               help="separate passages to sentences and parse each one separately")
        argparser.add_argument("-a", "--paragraphs", action="store_true", default=False,
                               help="separate passages to paragraphs and parse each one separately")
        argparser.add_argument("-r", "--learningrate", type=float, default=1.0,
                               help="learning rate for the model weight updates")
        argparser.add_argument("-u", "--minupdate", type=int, default=5,
                               help="minimum updates a feature must have before being used")
        argparser.add_argument("-l", "--checkloops", action="store_true", default=False,
                               help="check for infinite loops")
        argparser.add_argument("-V", "--verify", action="store_true", default=False,
                               help="verify oracle successfully reproduces the passage")
        argparser.add_argument("-c", "--compoundswap", action="store_true", default=False,
                               help="enable compound swap")
        argparser.add_argument("-S", "--maxswap", type=int, default=11,
                               help="maximum distance for compound swap")
        argparser.add_argument("-N", "--maxnodes", type=float, default=3.0,
                               help="maximum ratio between non-terminal to terminal nodes")
        argparser.add_argument("-M", "--multiedge", action="store_true", default=False,
                               help="allow multiple edges between the same nodes (with different tags)")
        self.args = argparser.parse_args()

        self.iterations = self.args.iterations

        self.verbose = self.args.verbose
        self.line_end = "\n" if self.verbose else " "  # show all in one line unless verbose
        self._log_file = None

        self.sentences = self.args.sentences
        self.paragraphs = self.args.paragraphs
        assert not (self.sentences and self.paragraphs),\
            "At most one of --sentences and --paragraphs may be specified"
        self.split = self.sentences or self.paragraphs
        self.learning_rate = self.args.learningrate
        self.min_update = self.args.minupdate
        self.check_loops = self.args.checkloops
        self.verify = self.args.verify

        self.compound_swap = self.args.compoundswap
        self.max_swap = self.args.maxswap

        self.max_nodes_ratio = self.args.maxnodes
        self.multiple_edges = self.args.multiedge

    def log(self, message):
        if self._log_file is None:
            self._log_file = open(self.args.log, "w")
        print(message, file=self._log_file, flush=True)
        if self.verbose:
            print(message)

    def close(self):
        if self._log_file is not None:
            self._log_file.close()

    def __str__(self):
        return " ".join("--%s=%s" % item for item in vars(self.args).items())
