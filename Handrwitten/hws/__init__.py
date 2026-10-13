"""Handwriting synthesis — stroke-level generative model and document renderer.

Package layout
--------------
    hws.alphabet    text  -> integer indices (and back)
    hws.iam         raw IAM-OnDB files -> Python objects
    hws.preprocess  raw pen coordinates -> model-ready arrays
    hws.render      pen coordinates -> SVG

The dependency arrow only ever points one way: `render` and `preprocess`
know nothing about IAM, and `iam` knows nothing about the model. That means
when you later swap the LSTM for a Transformer, nothing in this package has
to change.
"""

__version__ = "0.1.0"
