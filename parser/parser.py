import os
import random
import time
from itertools import groupby
from random import shuffle
from xml.etree.ElementTree import ParseError

from nltk import pos_tag

from action import Action
from averaged_perceptron import AveragedPerceptron
from config import Config
from features import FeatureExtractor
from oracle import Oracle
from state import State
from ucca import core, layer0, convert, ioutil, diffutil
from ucca.evaluation import evaluate, print_aggregate, average_f1


class ParserException(Exception):
    pass


class Parser(object):

    """
    Main class to implement transition-based UCCA parser
    """
    def __init__(self, model_file=None):
        self.state = None  # State object created at each parse
        self.oracle = None  # Oracle object created at each parse
        self.scores = None  # dict of action IDs -> model scores at each action
        self.action_count = 0
        self.correct_count = 0
        self.total_actions = 0
        self.total_correct = 0

        self.model = AveragedPerceptron(len(Action.get_all_actions()),
                                        min_update=Config().min_update)
        self.model_file = model_file
        self.feature_extractor = FeatureExtractor()

    def train(self, passages, dev=None, iterations=1):
        """
        Train parser on given passages
        :param passages: iterable of Passage objects to train on
        :param dev: iterable of Passage objects to tune on
        :param iterations: number of iterations to perform
        :return: trained model
        """
        if not passages:
            if self.model_file is not None:  # Nothing to train on; pre-trained model given
                self.model.load(self.model_file)
            return self.model

        best_score = 0
        best_model = None
        print("Training %d iterations" % iterations)
        for iteration in range(1, iterations + 1):
            print("Iteration %d: " % iteration)
            passages = [(passage, passage_id) for _, passage, passage_id in
                        self.parse(passages, mode="train")]
            shuffle(passages)
            if dev:
                print("Evaluating on dev passages")
                dev, scores = zip(*[((passage, passage_id),
                                     evaluate(predicted_passage, passage,
                                              verbose=False, units=False, errors=False))
                                    for predicted_passage, passage, passage_id in
                                    self.parse(dev, mode="dev")])
                score = average_f1(scores)
                print("Average F1 score on dev: %.3f" % score)

                if score > best_score:
                    print("Better than previous best score (%.3f)" % best_score)
                    best_score = score
                    best_model = self.model.average()
                    if self.model_file is not None:  # Save trained model
                        best_model.save(self.model_file)
                else:
                    print("Not better than previous best score (%.3f)" % best_score)
            print()
        print("Trained %d iterations" % iterations)

        self.model = best_model
        return self.model

    def parse(self, passages, mode="test"):
        """
        Parse given passages
        :param passages: iterable of pairs of (passage, passage ID), where passage may be:
                         either Passage object, or list of lists of tokens
        :param mode: "train", "test" or "dev".
                     If "train", use oracle to train on given passages.
                     Otherwise, just parse with classifier.
        :return: generator of triplets of (parsed passage, original passage, passage ID)
        """
        train = (mode == "train")
        passage_word = "sentence" if Config().sentences else \
                       "paragraph" if Config().paragraphs else \
                       "passage"
        assert train or mode in ("test", "dev"), "Invalid parse mode: %s" % mode
        self.total_actions = 0
        self.total_correct = 0
        total_duration = 0
        total_tokens = 0
        num_passages = 0
        for passage, passage_id in passages:
            print("%s %-7s" % (passage_word, passage_id), end=Config().line_end, flush=True)
            started = time.time()
            self.action_count = 0
            self.correct_count = 0
            assert not train or isinstance(passage, core.Passage), "Cannot train on unannotated passage"
            self.state = State(passage, passage_id, callback=self.pos_tag)
            history = set()
            self.oracle = Oracle(passage) if isinstance(passage, core.Passage) else None
            failed = False
            try:
                self.parse_passage(history, train)  # This is where the actual parsing takes place
            except ParserException as e:
                if train:
                    raise
                Config().log("%s %-7s:\n%s" % (passage_word, passage_id, e))
                print("failed")
                failed = True
            predicted_passage = passage
            if not train or Config().verify:
                predicted_passage = self.state.create_passage(assert_proper=Config().verify)
            duration = time.time() - started
            total_duration += duration
            if not failed:
                if self.oracle:  # passage is a Passage object, and we have an oracle to verify by
                    if Config().verify:
                        self.verify_passage(passage, predicted_passage, train)
                    print("accuracy: %.3f (%d/%d)" %
                          (self.correct_count/self.action_count, self.correct_count, self.action_count)
                          if self.action_count else "No actions done", end=Config().line_end)
                num_tokens = len(passage.layer(layer0.LAYER_ID).all) if self.oracle else sum(map(len, passage))
                total_tokens += num_tokens
                print("time: %0.3fs (%d tokens/second)" % (duration, num_tokens / duration),
                      end=Config().line_end + "\n", flush=True)
            self.total_correct += self.correct_count
            self.total_actions += self.action_count
            num_passages += 1
            yield predicted_passage, passage, passage_id

        if num_passages > 1:
            print("Parsed %d %ss" % (num_passages, passage_word))
            if self.oracle and self.total_actions:
                print("Overall %s accuracy: %.3f (%d/%d)" %
                      (mode,
                       self.total_correct / self.total_actions, self.total_correct, self.total_actions))
            print("Total time: %.3fs (average time/%s: %.3fs, average tokens/second: %d)" % (
                total_duration, passage_word, total_duration / num_passages,
                total_tokens / total_duration), flush=True)

    def parse_passage(self, history=None, train=False):
        """
        Internal method to parse a single passage
        :param history: set of hashed states in the parser's history, if loop checking is enabled
        :param train: use oracle to train on given passages, or just parse with classifier?
        """
        if Config().verbose:
            print("  initial state: %s" % self.state)
        while True:
            if Config().check_loops and history is not None:
                self.check_loop(history, train)

            true_actions = []
            if self.oracle is not None:
                try:
                    true_actions = self.oracle.get_actions(self.state)
                except (AttributeError, AssertionError) as e:
                    if train:
                        raise ParserException("Error in oracle during training") from e

            features = self.feature_extractor.extract_features(self.state)
            predicted_action = self.predict_action(features, true_actions)
            action = predicted_action
            if not true_actions:
                true_actions = "?"
            elif predicted_action in true_actions:
                self.correct_count += 1
            elif train:
                best_true_action_id = max([true_action.id for true_action in true_actions],
                                          key=self.scores.get) if len(true_actions) > 1 \
                    else true_actions[0].id
                self.model.update(features, predicted_action.id, best_true_action_id,
                                  Config().learning_rate)
                action = random.choice(true_actions)
            self.action_count += 1
            self.state.transition(action)
            if Config().verbose:
                if self.oracle is None:
                    print("  action: %-15s %s" % (action, self.state))
                else:
                    print("  predicted: %-15s true: %-15s taken: %-15s %s" % (
                        predicted_action, "|".join(str(true_action) for true_action in true_actions),
                        action, self.state))
                for line in self.state.log:
                    print("    " + line)
            if self.state.finished:
                return  # action is FINISH

    def check_loop(self, history, train):
        """
        Check if the current state has already occurred, indicating a loop
        :param history: set of hashed states in the parser's history
        :param train: whether to print the oracle in case of an assertion error
        """
        h = hash(self.state)
        assert h not in history, "\n".join(["Transition loop", self.state.str("\n")] +
                                           [self.oracle.str("\n")] if train else [])
        history.add(h)

    def predict_action(self, features, true_actions=None):
        """
        Choose action based on classifier
        :param features: extracted feature values
        :param true_actions: from the oracle, to copy orig_node if the same action is selected
        :return: valid action with maximum probability according to classifier
        """
        self.scores = self.model.score(features)  # Returns dict of id -> score
        best_action = self.select_action(max(self.scores, key=self.scores.get), true_actions)
        if self.state.is_valid(best_action):
            return best_action
        # Usually the best action is valid, so max is enough to choose it in O(n) time
        # Otherwise, sort all the other scores to choose the best valid one in O(n lg n)
        sorted_ids = reversed(sorted(self.scores, key=self.scores.get))
        actions = (self.select_action(i, true_actions) for i in sorted_ids)
        try:
            return next(action for action in actions if self.state.is_valid(action))
        except StopIteration as e:
            raise ParserException("No valid actions available\n"
                                  "True actions: %s" % true_actions) from e

    @staticmethod
    def select_action(i, true_actions):
        action = Action.by_id(i)
        try:
            return next(true_action for true_action in true_actions if action == true_action)
        except StopIteration:
            return action

    @staticmethod
    def verify_passage(passage, predicted_passage, show_diff):
        """
        Compare predicted passage to true passage and die if they differ
        :param passage: true passage
        :param predicted_passage: predicted passage to compare
        :param show_diff: if passages differ, show the difference between them?
                          Depends on predicted_passage having the original node IDs annotated
                          in the "remarks" field for each node.
        """
        assert passage.equals(predicted_passage), "Failed to produce true passage" + \
                                                  (diffutil.diff_passages(
                                                          passage, predicted_passage) if show_diff else "")

    @staticmethod
    def pos_tag(state):
        """
        Function to pass to State to POS tag the tokens when created
        :param state: State object to modify
        """
        tokens = [token for tokens in state.tokens for token in tokens]
        tokens, tags = zip(*pos_tag(tokens))
        if Config().verbose:
            print(" ".join("%s/%s" % (token, tag) for (token, tag) in zip(tokens, tags)))
        for node, tag in zip(state.nodes, tags):
            node.pos_tag = tag

    @staticmethod
    def read_passage(passage):
        """
        Read a passage given in any format
        :param passage: either a core.Passage, a file, or a list of list of strings (paragraphs, words)
        :return: a core.Passage and its ID if given a Passage or file, or else the given list of lists
        """
        if isinstance(passage, core.Passage):
            passage_id = passage.ID
        elif os.path.exists(passage):  # a file
            try:
                passage = ioutil.file2passage(passage)  # XML or binary format
                passage_id = passage.ID
            except (IOError, ParseError):
                passage_id = os.path.splitext(os.path.basename(passage))[0]
                with open(passage) as text_file:  # simple text file
                    lines = (line.strip() for line in text_file.readlines())
                    passage = [[token for line in group for token in line.split()]
                               for is_sep, group in groupby(lines, lambda x: not x)
                               if not is_sep]
        else:
            raise IOError("File not found: %s" % passage)
        return passage, passage_id


def read_passages(files):
    for file in files:
        passage, i = Parser.read_passage(file)
        if Config().split:
            segments = convert.split2segments(passage, is_sentences=Config().sentences,
                                              remarks=True)
            for j, segment in enumerate(segments):
                yield (segment, "%s_%d" % (i, j))
        else:
            yield (passage, i)


def read_files_and_dirs(files):
    """
    :param files: a list of files and/or directories to look in
    :return: passages from all files given, plus any files directly under any directory given
    """
    files += [os.path.join(d, f) for d in files if os.path.isdir(d) for f in os.listdir(d)]
    files = [f for f in files if not os.path.isdir(f)]
    return read_passages(files) if files else ()


def write_passage(passage, outdir, prefix, binary, verbose):
    suffix = ".pickle" if binary else ".xml"
    outfile = outdir + os.path.sep + prefix + passage.ID + suffix
    if verbose:
        print("Writing passage '%s'..." % outfile)
    ioutil.passage2file(passage, outfile, binary=binary)


def main():
    args = Config().args
    print("Running parser with %s" % Config())
    parser = Parser(args.model)
    train_passages = read_files_and_dirs(args.train)
    dev_passages = read_files_and_dirs(args.dev)
    parser.train(train_passages, dev=dev_passages, iterations=args.iterations)
    if args.passages:
        if args.train:
            print("Evaluating on test passages")
        scores = []
        test_passages = read_files_and_dirs(args.passages)
        for guessed_passage, ref_passage, _ in parser.parse(test_passages):
            if isinstance(ref_passage, core.Passage):
                scores.append(evaluate(guessed_passage, ref_passage,
                                       verbose=args.verbose and guessed_passage is not None))
            if guessed_passage is not None:
                write_passage(guessed_passage, args.outdir, args.prefix, args.binary, args.verbose)
        if scores:
            f1 = average_f1(scores)
            print()
            print("Average F1 score on test: %.3f" % f1)
            print("Aggregated scores:")
            print()
            print_aggregate(scores)
            return f1


if __name__ == "__main__":
    main()
    Config().close()
