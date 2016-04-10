class Classifier(object):
    """
    Interface for classifier used by the parser.
    """

    def __init__(self, labels=None, weights=None):
        """
        :param labels: a list of labels that can be updated later to add a new label
        :param weights: if given, copy the weights (from a trained model)
        """
        self.labels = labels or []
        self._init_num_labels = len(self.labels)
        self._num_labels = self._init_num_labels
        self.is_frozen = weights is not None

    @property
    def num_labels(self):
        return len(self.labels)

    def score(self, features):
        raise NotImplementedError()

    def update(self, features, pred, true, learning_rate=1):
        assert not self.is_frozen, "Cannot update a frozen model"
        self._update_num_labels()

    def _update_num_labels(self):
        """
        self.num_labels is updated automatically when a label is added to self.labels,
        but we need to update the weights whenever that happens
        """
        if self.num_labels > self._num_labels:
            self._num_labels = self.num_labels
            self.resize()

    def resize(self):
        raise NotImplementedError()

    def finalize(self, *args, **kwargs):
        assert not self.is_frozen, "Cannot freeze a frozen model"
        self._update_num_labels()

    def save(self, filename, io):
        raise NotImplementedError()

    def load(self, filename, io):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def write(self, filename, sep="\t"):
        raise NotImplementedError()
