#!/usr/bin/python3
"""
The evaluation software for UCCA layer 1.
"""

import sys, collections, pdb
from ucca import layer0, layer1, convert
import ucca_db
from ucca.core import Passage, Node
from xml.etree.ElementTree import ElementTree, tostring, fromstring
from optparse import OptionParser
from collections import Counter

UNLABELED = "unlabeled"
WEAK_LABELED = "weak_labeled"
LABELED = "labeled"

# Pairs that are considered as equivalent for the purposes of evaluation
EQUIV = [["P", "S"], ["H", "C"], ["N", "L"], ["F", "R"]]

#RELATORS = ["that", "than", "who", "what", "to", "how", "of"]

##############################################################################
# UTILITY METHODS
##############################################################################


def flatten_Cs(P):
    "If there are Cs inside Cs in layer1 of passage C, cancel the external C."
    to_ungroup = []
    for u in P.layer("1").all:
        if u.tag == "FN" and u.ftag == "C":
            ch_C_indices = [x for x in u.children
                            if x.tag == "FN" and x.ftag == "C"]
            parent = u.fparent
            if parent is not None:
                sib_C_indices = [x for x in parent.children
                                 if x.tag == "FN" and x.ftag == "C"]
            if len(ch_C_indices) == 1 and \
               (parent is None or len(sib_C_indices) == 1):
                to_ungroup.append(u)

    # debug
    pr = [(u, u.fparent) for u in to_ungroup]
    for x, p in pr:
        print('\n'.join([str(x), str(p)]))

    for u in to_ungroup:
        ungroup(u)

    print('\n\n')
    print('\n'.join([str(x[1]) for x in pr]))


def ungroup(x):
    """
    If the unit has an fparent,
    removes the unit and adds its children to that parent.
    """
    if x.tag != "FN":
        return None
    fparent = x.fparent
    if fparent is not None:
        if len(x.parents) > 1:
            # if there is only one child, assign that child as the ??????
            if len(x.centers) == 1:
                for e in x.incoming:
                    if e.attrib.get("remote"):
                        e.parent.add(e.tag, x.centers[0], edge_attrib=e.attrib)
            else:
                # don't ungroup
                # if there is more than one parent and no single center
                return None
        for e in x.outgoing:
            fparent.add(e.tag, e.child, edge_attrib=e.attrib)
    x.destroy()
    return fparent


def to_text(P, terms):
    """
    Return:
        Text representation of the terminals whose terminals are in terms
    """
    L = sorted(list(terms))
    words = [P.layer("0").by_position(pos).text for pos in L]
    pre_context = [P.layer("0").by_position(pos).text
                   for pos in range(min(L) - 3, min(L) + 1)]
    post_context = [P.layer("0").by_position(pos).text
                    for pos in range(max(L), max(L) + 3)]
    return ' '.join(pre_context) + '\t' + ' '.join(words) +\
           '\t' + ' '.join(post_context)


def mutual_yields(passage1, passage2, eval_type, separate_remotes=True):
    """
    return:
    Set of all the yields such that both passages have a unit with that yield.
    eval type can be:
    1. UNLABELED: it doesn't matter what labels are there.
    2. LABELED: also requires tag match
    (if there are multiple units with the same yield, requires one match)
    3. WEAK_LABELED: also requires weak tag match
    (if there are multiple units with the same yield, requires one match)
    the constants are defind with the module

    returns a 4-tuple:
    -- the set of mutual yields
    -- the set of mutual remote yields
    (unless separate_remotes is False, then None)
    -- the set of yields of passage1
    -- the set of yields of passage2
    """
    def _find_mutuals(map1, map2):
        mutual_ys = set()
        error_counter = Counter()

        for y in map1.keys():
            if y in map2.keys():
                if eval_type == UNLABELED:
                    mutual_ys.add(y)
                else:
                    tags1 = set(e.tag for e in map1[y])
                    tags2 = set(e.tag for e in map2[y])
                    if eval_type == WEAK_LABELED:
                        tags1 = expand_equivalents(tags1)
                    if (tags1 & tags2):  # non-empty intersection
                        mutual_ys.add(y)
                    else:
                        error_counter[(str(tags1), str(tags2))] += 1
                        if ('E' in tags1 and 'C' in tags2) or \
                           ('C' in tags1 and 'E' in tags2):
                            print('C-E', to_text(passage1, y))
                        elif ('P' in tags1 and 'C' in tags2) or \
                             ('C' in tags1 and {'P', 'S'} & tags2):
                            print('P|S-C', to_text(passage1, y))
                        elif ('A' in tags1 and 'E' in tags2) or \
                             ('E' in tags1 and 'A' in tags2):
                            print('A-E', to_text(passage1, y))

        return mutual_ys, error_counter

    map1, map1_remotes = create_passage_yields(passage1, not separate_remotes)
    map2, map2_remotes = create_passage_yields(passage2, not separate_remotes)

    output, error_counter = _find_mutuals(map1, map2)
    output_remotes = None
    if separate_remotes:
        # temp will be discarded
        output_remotes, temp = _find_mutuals(map1_remotes, map2_remotes)

    return (output, set(map1.keys()), set(map2.keys()), output_remotes,
            set(map1_remotes.keys()), set(map2_remotes.keys()), error_counter)


def create_passage_yields(p, remoteTerminals=False):
    """
    returns two dicts:
    1. maps a set of terminal indices (excluding punctuation)
       to a list of layer1 edges whose yield
       (excluding remotes and punctuation) is that set.
    2. maps a set of terminal indices (excluding punctuation)
       to a set of remote edges whose yield
       (excluding remotes and punctuation) is that set.
    remoteTerminals - if true, regular table includes remotes.
    """
    l1 = p.layer("1")
    edges = []
    for node in l1.all:
        edges.extend([e for e in node if e.tag not in ['U', 'LA', 'LR', 'T']])

    table_reg, table_remote = dict(), dict()
    for e in edges:
        pos = frozenset(t.position for t
                        in e.child.get_terminals(
                            punct=False, remotes=remoteTerminals))
        if e.attrib.get("remote"):
            table_remote[pos] = table_remote.get(pos, []) + [e]
        else:
            table_reg[pos] = table_reg.get(pos, []) + [e]

    return table_reg, table_remote


def expand_equivalents(tag_set):
    "Returns a set of all the tags in the tag set or those equivalent to them"
    output = tag_set.copy()
    for t in tag_set:
        for pair in EQUIV:
            if t in pair:
                output.update(pair)
    return output


def tag_distribution(unit_list):
    """
    Input:
        list of units
    Return:
    Dict that maps the tags of the units to their frequency in the text"""
    output = collections.Counter()
    for u in unit_list:
        output[u.tag] += 1
    return output


##############################################################################
# Returns the command line parser.
##############################################################################
def cmd_line_parser():
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("--db", "-d", dest="db_filename",
                          action="store", type="string",
                          help="the db file name")
    opt_parser.add_option("--host", "--hst", dest="host",
                          action="store", type="string",
                          help="the host name")
    opt_parser.add_option("--pid", "-p", dest="pid", action="store",
                          type="int", help="the passage ID")
    opt_parser.add_option("--from_xids", "-x", dest="from_xids",
                          action="store_true", help="interpret the ref \
                          and the guessed parameters as Xids in the db")
    opt_parser.add_option("--guessed", "-g", dest="guessed", action="store",
                          type="string", help="if a db is defined - \
                          the username for the guessed annotation; \
                          else - the xml file name for the guessed annotation")
    opt_parser.add_option("--ref", "-r", dest="ref", action="store",
                          type="string", help="if a db is defined - \
                          the username for the reference annotation; else - \
                          the xml file name for the reference annotation")
    opt_parser.add_option("--units", "-u", dest="units", action="store_true",
                          help="the units the annotations have in common, \
                          and those each has separately")
    opt_parser.add_option("--fscore", "-f", dest="fscore", action="store_true",
                          help="outputs the traditional P,R,F \
                          instead of the scene structure evaluation")
    opt_parser.add_option("--debug", dest="debug", action="store_true",
                          help="run in debug mode")
    opt_parser.add_option("--reference_from_file", dest="ref_from_file",
                          action="store_true",
                          help="loads the reference \
                          from a file and not from the db")
    opt_parser.add_option("--errors", "-e", dest="errors", action="store_true",
                          help="prints the error distribution\
                          according to its frequency")
    return opt_parser


def get_scores(p1, p2, eval_type):
    """
    LEGACY: works just like print_scores, the newer name.
    NOTE: this function returns no value.
    prints the relevant statistics and f-scores.
    eval_type can be UNLABELED, LABELED or WEAK_LABELED.
    """
    def _print_scores(num_matches, num_only_guessed, num_only_ref):
        "Prints the F scores according to the given numbers."
        num_guessed = num_matches + num_only_guessed
        num_ref = num_matches + num_only_ref
        if num_guessed == 0:
            P = "NaN"
        else:
            P = 1.0 * num_matches / num_guessed
        if num_ref == 0:
            R = "NaN"
        else:
            R = 1.0 * num_matches / num_ref
        if P == "NaN" or R == "NaN":
            F = "NaN"
        elif P == 0 and R == 0:
            F = 0.0
        else:
            F = 2 * P * R / (P + R)

        print("Precision: {:.3} ({}/{})".format(P, num_matches, num_guessed))
        print("Recall: {:.3} ({}/{})".format(R, num_matches, num_ref))
        print("F1: {:.3}".format(F))

    mutual, all1, all2, mutual_rem, all1_rem, all2_rem, err_counter \
        = mutual_yields(p1, p2, eval_type)
    print("Evaluation type: (" + eval_type + ")")

    if options.units:
        print("==> Mutual Units:")
        for y in mutual:
            print([p1.layer("0").by_position(pos).text for pos in y])

        print("==> Only in guessed:")
        for y in (all1 - mutual):
            print([p1.layer("0").by_position(pos).text for pos in y])

        print("==> Only in reference:")
        for y in (all2 - mutual):
            print([p1.layer("0").by_position(pos).text for pos in y])

    if options.fscore:
        print("\nRegular Edges:")
        _print_scores(len(mutual), len(all1 - mutual), len(all2 - mutual))

        print("\nRemote Edges:")
        _print_scores(len(mutual_rem), len(all1_rem - mutual_rem), len(all2_rem - mutual_rem))
        print()

    if options.errors:
        print("\nConfusion Matrix:\n")
        for error, freq in err_counter.most_common():
            print(error[0], '\t', error[1], '\t', freq)


def print_scores(p1, p2, eval_type):
    """
    prints the relevant statistics and f-scores.
    eval_type can be UNLABELED, LABELED or WEAK_LABELED.
    NOTE: equivalent to get_score, more accurate naming.
    """
    return get_scores(p1, p2, eval_type)


def main():
    opt_parser = cmd_line_parser()
    (options, args) = opt_parser.parse_args()
    print (args)
    print(options)
    if len(args) > 0:
        opt_parser.error("all arguments must be flagged")

    if (options.guessed is None) or (options.ref is None):
        opt_parser.error("missing arguments. type --help for help.")
    if (options.pid is not None and options.from_xids is not None):
        opt_parser.error("inconsistent parameters. \
        you can't have both a pid and from_xids paramters.")

    if options.db_filename is None:
        # Read the xmls from files
        xmls = []
        files = [options.guessed, options.ref]
        for filename in files:
            f = open(filename)
            xmls.append(ElementTree().parse(f))
            f.close()
    else:
        if options.ref_from_file:
            xmls = ucca_db.get_xml_trees(options.db_filename, options.host,
                                         options.pid, [options.guessed])
            f = open(options.ref)
            xmls.append(ElementTree().parse(f))
            f.close()
        else:
            keys = [options.guessed, options.ref]
            if options.from_xids:
                xmls = ucca_db.get_by_xids(options.db_filename, options.host, keys)
            else:
                xmls = ucca_db.get_xml_trees(options.db_filename, options.host,
                                             options.pid, keys)

    passages = [convert.from_site(x) for x in xmls]

    for x in passages:
        flatten_Cs(x)  # flatten Cs inside Cs
        #normalize_preps(x)  # normalizes prepositions to be Rs

    if (options.units or options.fscore or options.errors):
        for eval_type in [LABELED, UNLABELED, WEAK_LABELED]:
            get_scores(passages[0], passages[1], eval_type)
    #else:
    #    scene_structures = [layer1s.SceneStructure(x) for x in passages]
    #    comp = comparison.PassageComparison(scene_structures[0],
    #                                        scene_structures[1])
    #    comp.text_report(sys.stdout)

if __name__ == '__main__':
    main()
