"""Utility functions for UCCA scripts."""
import pickle
from xml.etree.ElementTree import ElementTree, tostring

from ucca.convert import from_standard, to_standard
from ucca.textutil import indent_xml


def file2passage(filename):
    """Opens a file and returns its parsed Passage object
    Tries to read both as a standard XML file and as a binary pickle
    :param filename: file name to write to
    """
    try:
        with open(filename) as f:
            etree = ElementTree().parse(f)
        return from_standard(etree)
    except Exception as e:
        try:
            with open(filename, 'rb') as h:
                return pickle.load(h)
        except Exception:
            raise e


def passage2file(passage, filename, indent=True, binary=False):
    """Writes a UCCA passage as a standard XML file or a binary pickle
    :param passage: passage object to write
    :param filename: file name to write to
    :param indent: whether to indent each line
    :param binary: whether to write pickle format (or XML)
    """
    if binary:
        with open(filename, 'wb') as h:
            pickle.dump(passage, h)
    else:  # xml
        root = to_standard(passage)
        xml = tostring(root).decode()
        output = indent_xml(xml) if indent else xml
        with open(filename, 'w') as h:
            h.write(output)
