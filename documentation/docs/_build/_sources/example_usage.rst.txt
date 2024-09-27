
Example Usage
=============

Here is a quick example of how to use ETIA:

.. code-block:: python

    from ETIA.AFS import AFS
    from ETIA.CausalLearning import CausalLearner

    # Feature selection
    afs = AFS()
    selected_features = afs.select_features(dataset)

    # Causal discovery
    cl = CausalLearner()
    causal_graph = cl.learn(dataset)
