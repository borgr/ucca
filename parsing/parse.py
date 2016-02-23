import os
import random
import time
from random import shuffle

from nltk import pos_tag

from parsing.action import Actions
from parsing.averaged_perceptron import AveragedPerceptron
from parsing.config import Config
from parsing.features import FeatureExtractor
from parsing.oracle import Oracle
from parsing.state import State
from parsing.util import read_files_and_dirs, write_passage, save, load
from ucca import core, diffutil, evaluation, layer0, layer1


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

        self.model = AveragedPerceptron(len(Actions().all),
                                        min_update=Config().min_update)
        self.model_file = model_file
        self.feature_extractor = FeatureExtractor()

        self.learning_rate = Config().learning_rate
        self.decay_factor = Config().decay_factor

        # Used in verify_passage to optionally ignore a mismatch in linkage nodes
        self.ignore_node = lambda n: n.tag == layer1.NodeTags.Linkage if Config().no_linkage else None

        self.state_hash_history = None  # For loop checking

    def train(self, passages, dev=None, iterations=1):
        """
        Train parser on given passages
        :param passages: iterable of passages to train on
        :param dev: iterable of passages to tune on
        :param iterations: number of iterations to perform
        :return: trained model
        """
        if not passages:
            if self.model_file is not None:  # Nothing to train on; pre-trained model given
                d = load(self.model_file)
                self.model.load(d["model"])
                Actions.all = d["actions"]
            return self.model

        best_score = 0
        best_model = None
        save_model = True
        with open(Config().dev_scores, "w") as f:
            print(",".join(["iteration"] + evaluation.Scores.field_titles()), file=f)
        for iteration in range(iterations):
            print("Training iteration %d of %d: " % (iteration + 1, iterations))
            passages = [passage for _, passage in self.parse(passages, mode="train")]
            self.learning_rate *= self.decay_factor
            shuffle(passages)
            if dev:
                print("Evaluating on dev passages")
                dev, scores = zip(*[(passage, evaluation.evaluate(
                    predicted_passage, passage, verbose=False, units=False, errors=False))
                                    for predicted_passage, passage in
                                    self.parse(dev, mode="dev")])
                scores = evaluation.Scores.aggregate(scores)
                score = scores.average_unlabeled_f1()
                print("Average unlabeled F1 score on dev: %.3f" % score)
                with open(Config().dev_scores, "a") as f:
                    print(",".join([str(iteration)] + scores.fields()), file=f)
                if score >= best_score:
                    print("Better than previous best score (%.3f)" % best_score)
                    best_score = score
                    save_model = True
                else:
                    print("Not better than previous best score (%.3f)" % best_score)
                    save_model = False

            if save_model or best_model is None:
                best_model = self.model.average()
                if self.model_file is not None:
                    save(self.model_file, {"model": best_model.save(),
                                           "actions": Actions().all})

        print("Trained %d iterations" % iterations)

        self.model = best_model
        return self.model

    def parse(self, passages, mode="test"):
        """
        Parse given passages
        :param passages: iterable of passages to parse
        :param mode: "train", "test" or "dev".
                     If "train", use oracle to train on given passages.
                     Otherwise, just parse with classifier.
        :return: generator of pairs of (parsed passage, original passage)
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
        for passage in passages:
            print("%s %-7s" % (passage_word, passage.ID), end=Config().line_end, flush=True)
            started = time.time()
            self.action_count = 0
            self.correct_count = 0
            assert not train or isinstance(passage, core.Passage), "Cannot train on unannotated passage"
            self.state = State(passage, callback=self.pos_tag)
            self.state_hash_history = set()
            self.oracle = Oracle(passage) if isinstance(passage, core.Passage) else None
            failed = False
            try:
                self.parse_passage(train)  # This is where the actual parsing takes place
            except ParserException as e:
                if train:
                    raise
                Config().log("%s %s: %s" % (passage_word, passage.ID, e))
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
            yield predicted_passage, passage

        if num_passages > 1:
            print("Parsed %d %ss" % (num_passages, passage_word))
            if self.oracle and self.total_actions:
                print("Overall %s accuracy: %.3f (%d/%d)" %
                      (mode,
                       self.total_correct / self.total_actions, self.total_correct, self.total_actions))
            print("Total time: %.3fs (average time/%s: %.3fs, average tokens/second: %d)" % (
                total_duration, passage_word, total_duration / num_passages,
                total_tokens / total_duration), flush=True)

    def parse_passage(self, train=False):
        """
        Internal method to parse a single passage
        :param train: use oracle to train on given passages, or just parse with classifier?
        """
        if Config().verbose:
            print("  initial state: %s" % self.state)
        while True:
            if Config().check_loops:
                self.check_loop(print_oracle=train)

            true_actions = []
            if self.oracle is not None:
                try:
                    true_actions = self.oracle.get_actions(self.state)
                except (AttributeError, AssertionError) as e:
                    if train:
                        raise ParserException("Error in oracle during training") from e

            features = self.feature_extractor.extract_features(self.state)
            predicted_action = self.predict_action(features, true_actions)  # sets self.scores
            action = predicted_action
            if not true_actions:
                true_actions = "?"
            elif predicted_action in true_actions:
                self.correct_count += 1
            elif train:
                best_true_action_id = max([true_action.id for true_action in true_actions],
                                          key=self.scores.get) if len(true_actions) > 1 \
                    else true_actions[0].id
                rate = self.learning_rate
                if Actions().by_id(best_true_action_id).is_swap:
                    rate *= Config().importance
                self.model.update(features, predicted_action.id, best_true_action_id, rate)
                action = random.choice(true_actions)
            self.action_count += 1
            try:
                self.state.transition(action)
            except AssertionError as e:
                raise ParserException("Invalid transition: %s" % action) from e
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

    def check_loop(self, print_oracle):
        """
        Check if the current state has already occurred, indicating a loop
        :param print_oracle: whether to print the oracle in case of an assertion error
        """
        h = hash(self.state)
        assert h not in self.state_hash_history,\
            "\n".join(["Transition loop", self.state.str("\n")] +
                      [self.oracle.str("\n")] if print_oracle else ())
        self.state_hash_history.add(h)

    def predict_action(self, features, true_actions=None):
        """
        Choose action based on classifier
        :param features: extracted feature values
        :param true_actions: from the oracle, to copy orig_node if the same action is selected
        :return: valid action with maximum probability according to classifier
        """
        self.scores = self.model.score(features)  # Returns dict of id -> score
        if true_actions is not None:
            self.scores.update({a.id: float("-inf") for a in true_actions if a.id not in self.scores})
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
            raise ParserException("No valid actions available\n" +
                                  ("True actions: %s" % true_actions if true_actions
                                   else self.oracle.log if self.oracle is not None
                                   else "")) from e

    @staticmethod
    def select_action(i, true_actions):
        action = Actions().by_id(i)
        try:
            return next(true_action for true_action in true_actions if action == true_action)
        except StopIteration:
            return action

    def verify_passage(self, passage, predicted_passage, show_diff):
        """
        Compare predicted passage to true passage and die if they differ
        :param passage: true passage
        :param predicted_passage: predicted passage to compare
        :param show_diff: if passages differ, show the difference between them?
                          Depends on predicted_passage having the original node IDs annotated
                          in the "remarks" field for each node.
        """
        assert passage.equals(predicted_passage, ignore_node=self.ignore_node),\
            "Failed to produce true passage" + \
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


def train_test(train_passages, dev_passages, test_passages, args, model_suffix=""):
    scores = None
    model_file = args.model
    if model_file is not None:
        model_base, model_ext = os.path.splitext(model_file)
        model_file = model_base + model_suffix + model_ext
    p = Parser(model_file)
    p.train(train_passages, dev=dev_passages, iterations=args.iterations)
    if test_passages:
        if args.train or args.folds:
            print("Evaluating on test passages")
        passage_scores = []
        for guessed_passage, ref_passage in p.parse(test_passages):
            if args.evaluate or train_passages:
                passage_scores.append(evaluation.evaluate(
                    guessed_passage, ref_passage,
                    verbose=args.verbose and guessed_passage is not None))
            if guessed_passage is not None:
                write_passage(guessed_passage, args)
        if passage_scores:
            scores = evaluation.Scores.aggregate(passage_scores)
            print("\nAverage F1 score on test: %.3f" % scores.average_unlabeled_f1())
            print("Aggregated scores:")
            scores.print()
    return scores


def main():
    args = Config().args
    print("Running parser with %s" % Config())
    scores = None
    if args.folds is not None:
        k = args.folds
        fold_scores = []
        all_passages = list(read_files_and_dirs(args.passages))
        assert len(all_passages) >= k,\
            "%d folds are not possible with only %d passages" % (k, len(all_passages))
        shuffle(all_passages)
        folds = [all_passages[i::k] for i in range(k)]
        for i in range(k):
            print("Fold %d of %d:" % (i + 1, k))
            dev_passages = folds[i]
            test_passages = folds[(i+1) % k]
            train_passages = [passage for fold in folds
                              if fold is not dev_passages and fold is not test_passages
                              for passage in fold]
            s = train_test(train_passages, dev_passages, test_passages, args, "_%d" % i)
            if s is not None:
                fold_scores.append(s)
        if fold_scores:
            scores = evaluation.Scores.aggregate(fold_scores)
            print("Average unlabeled test F1 score for each fold: " + ", ".join(
                "%.3f" % s.average_unlabeled_f1() for s in fold_scores))
            print("Aggregated scores across folds:\n")
            scores.print()
    else:  # Simple train/dev/test by given arguments
        train_passages, dev_passages, test_passages = [read_files_and_dirs(arg) for arg in
                                                       (args.train, args.dev, args.passages)]
        scores = train_test(train_passages, dev_passages, test_passages, args)
    return scores


if __name__ == "__main__":
    main()
    Config().close()
