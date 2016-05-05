"""Utility functions for UCCA package."""
from ucca import layer0, layer1

SENTENCE_END_MARKS = ('.', '?', '!')


def break2sentences(passage):
    """
    Breaks paragraphs into sentences according to the annotation.

    A sentence is a list of terminals which ends with a mark from
    SENTENCE_END_MARKS, and is also the end of a paragraph or parallel scene.
    :param passage: the Passage object to operate on
    :return: a list of positions in the Passage, each denotes a closing Terminal
        of a sentence.
    """
    l1 = passage.layer(layer1.LAYER_ID)
    terminals = passage.layer(layer0.LAYER_ID).all
    ps_ends = [ps.end_position for ps in l1.top_scenes]
    ps_starts = [ps.start_position for ps in l1.top_scenes]
    marks = [t.position for t in terminals if t.text[-1] in SENTENCE_END_MARKS]
    # Annotations doesn't always include the ending period (or other mark)
    # with the parallel scene it closes. Hence, if the terminal before the
    # mark closed the parallel scene, and this mark doesn't open a scene
    # in any way (hence it probably just "hangs" there), it's a sentence end
    marks = [x for x in marks
             if x in ps_ends or ((x - 1) in ps_ends and x not in ps_starts)]
    return sorted(set(marks + break2paragraphs(passage)))


def break2paragraphs(passage):
    """
    Breaks into paragraphs according to the annotation.

    Uses the `paragraph' attribute of layer 0 to find paragraphs.
    :param passage: the Passage object to operate on
    :return: a list of positions in the Passage, each denotes a closing Terminal
        of a paragraph.
    """
    terminals = passage.layer(layer0.LAYER_ID).all
    paragraph_ends = [(t.position - 1) for t in terminals
                      if t.position != 1 and t.para_pos == 1]
    paragraph_ends.append(terminals[-1].position)
    return paragraph_ends


def indent_xml(xml_as_string):
    """
    Indents a string of XML-like objects.

    This works only for units with no text or tail members, and only for
    strings whose leaves are written as <tag /> and not <tag></tag>.
    :param xml_as_string: XML string to indent
    :return: indented XML string
    """
    tabs = 0
    lines = str(xml_as_string).replace('><', '>\n<').splitlines()
    s = ''
    for line in lines:
        if line.startswith('</'):
            tabs -= 1
        s += ("  " * tabs) + line + '\n'
        if not (line.endswith('/>') or line.startswith('</')):
            tabs += 1
    return s
