from features.feature_extractor import FeatureExtractor

FEATURE_TEMPLATES = (
    # unigrams (Zhang and Clark 2009):
    "s0te", "s0we", "s1te", "s1we", "s2te", "s2we", "s3te", "s3we",
    "b0wt", "b1wt", "b2wt", "b3wt",
    "s0lwe", "s0rwe", "s0uwe", "s1lwe", "s1rwe", "s1uwe",
    # bigrams (Zhang and Clark 2009):
    "s0ws1w", "s0ws1e", "s0es1w", "s0es1e", "s0wb0w", "s0wb0t",
    "s0eb0w", "s0eb0t", "s1wb0w", "s1wb0t", "s1eb0w", "s1eb0t",
    "b0wb1w", "b0wb1t", "b0tb1w", "b0tb1t",
    # trigrams (Zhang and Clark 2009):
    "s0es1es2w", "s0es1es2e", "s0es1es2e", "s0es1eb0w", "s0es1eb0t",
    "s0es1wb0w", "s0es1wb0t", "s0ws1es2e", "s0ws1eb0t",
    # extended (Zhu et al. 2013):
    "s0llwe", "s0lrwe", "s0luwe", "s0rlwe", "s0rrwe",
    "s0ruwe", "s0ulwe", "s0urwe", "s0uuwe", "s1llwe",
    "s1lrwe", "s1luwe", "s1rlwe", "s1rrwe", "s1ruwe",
    # parents:
    "s0Uwe", "s1Uwe", "b0Uwe",
    # separator (Zhu et al. 2013):
    "s0wp", "s0wep", "s0wq", "s0weq", "s0es1ep", "s0es1eq",
    "s1wp", "s1wep", "s1wq", "s1weq",
    # disco, unigrams (Maier 2015):
    "s0xwe", "s1xwe", "s2xwe", "s3xwe",
    "s0xte", "s1xte", "s2xte", "s3xte",
    "s0xy", "s1xy", "s2xy", "s3xy",
    # disco, bigrams (Maier 2015):
    "s0xs1e", "s0xs1w", "s0xs1x", "s0ws1x", "s0es1x",
    "s0xs2e", "s0xs2w", "s0xs2x", "s0es2x",
    "s0ys1y", "s0ys2y", "s0xb0t", "s0xb0w",
    # counts (Tokgöz and Eryiğit 2015):
    "s0P", "s0C", "s0wP", "s0wC",
    "b0P", "b0C", "b0wP", "b0wC",
    # existing edges (Tokgöz and Eryiğit 2015):
    "s0s1", "s1s0", "s0b0", "b0s0",
    "s0b0e", "b0s0e",
    # past actions (Tokgöz and Eryiğit 2015):
    "a0we", "a1we",
    # UCCA-specific
    "s0I", "s0R", "s0wI", "s0wR",
    "b0I", "b0R", "b0wI", "b0wR",
)


class SparseFeatureExtractor(FeatureExtractor):
    """
    Object to extract features from the parser state to be used in action classification
    To be used with SparsePerceptron classifier.
    """
    def __init__(self):
        super(SparseFeatureExtractor, self).__init__(FEATURE_TEMPLATES)

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> value
        """
        features = {
            "b": 1,  # Bias
            "n/t": state.node_ratio(),  # number of nodes divided by number of terminals
        }
        for feature_template in self.feature_templates:
            values = self.calc_feature(feature_template, state)
            if values is not None:
                features["%s=%s" % (feature_template.name, " ".join(map(str, values)))] = 1
        return features
