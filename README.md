Universal Conceptual Cognitive Annotation [![Build Status](https://travis-ci.org/danielhers/ucca.svg?branch=master)](https://travis-ci.org/danielhers/ucca)
============================
UCCA is a novel linguistic framework for semantic annotation, whose details
are available at [the following paper][1]:

    Universal Conceptual Cognitive Annotation (UCCA)
    Omri Abend and Ari Rappoport, ACL 2013

This Python3-only package provides an API to the UCCA annotation and tools to
manipulate and process it. It's main features are conversion between different
representations of UCCA annotations, and rich objects for all of the linguistic
relations which appear in the theoretical framework (see `core`, `layer0`, `layer1`
and `convert` modules under the `ucca` package).

Installation (on Linux):
------------------------

    make dev-install  # creates soft links to the current files
    make full-install  # copies the package to the user's python search path
    
run `make help` for details


See [`ucca/README.md`](ucca/README.md) for a list of modules under the `ucca` package.

The `scripts` package contains various utilities for processing passage files.

The `parsing` package contains code for a full UCCA parser, currently under construction.

Authors
------
* Amit Beka: amit.beka@gmail.com
* Daniel Hershcovich: danielh@cs.huji.ac.il


License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](master/LICENSE.txt)).

[1]: http://homepages.inf.ed.ac.uk/oabend/papers/ucca_acl.pdf