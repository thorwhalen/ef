"""
Plugin to integrate imbed's production implementations into ef.

This is a stub/bridge module that will connect ef to the imbed package
when it's available.

Usage:
    from ef import Project
    from ef.plugins import imbed

    project = Project.create('production')
    imbed.register(project)

    # Now has all imbed components
    print(project.embedders.keys())  # openai-small, openai-large, ...
"""


def register(project, *, include_datasets=True, include_utils=True):
    """
    Register all imbed components with an ef project.

    Args:
        project: An ef.Project instance
        include_datasets: Whether to include dataset-related components
        include_utils: Whether to add utility methods

    Raises:
        ImportError: If imbed package is not installed

    Example:
        >>> from ef import Project
        >>> from ef.plugins import imbed
        >>> project = Project.create('production')
        >>> imbed.register(project)  # doctest: +SKIP
    """
    try:
        import imbed
    except ImportError:
        raise ImportError(
            "The imbed package is required for this plugin.\n"
            "Install it with: pip install imbed\n"
            "Or install ef with imbed support: pip install ef[imbed]"
        )

    _register_embedders(project)
    _register_planarizers(project)
    _register_clusterers(project)
    _register_segmenters(project)

    if include_datasets:
        _register_dataset_classes(project)

    if include_utils:
        _add_utility_methods(project)


def _register_embedders(project):
    """Add imbed's real embedders to project."""
    try:
        from imbed.components.vectorization import embedders as imbed_embedders

        # Wrap each imbed embedder
        for name, func in imbed_embedders.items():
            project.embedders.register(name)(func)

        print(f"✓ Registered {len(imbed_embedders)} imbed embedders")
    except ImportError as e:
        print(f"○ Could not register imbed embedders: {e}")


def _register_planarizers(project):
    """Add imbed's real planarizers."""
    try:
        from imbed.components.planarization import planarizers as imbed_planarizers

        for name, func in imbed_planarizers.items():
            project.planarizers.register(name)(func)

        print(f"✓ Registered {len(imbed_planarizers)} imbed planarizers")
    except ImportError as e:
        print(f"○ Could not register imbed planarizers: {e}")


def _register_clusterers(project):
    """Add imbed's real clusterers."""
    try:
        from imbed.components.clusterization import clusterers as imbed_clusterers

        for name, func in imbed_clusterers.items():
            project.clusterers.register(name)(func)

        print(f"✓ Registered {len(imbed_clusterers)} imbed clusterers")
    except ImportError as e:
        print(f"○ Could not register imbed clusterers: {e}")


def _register_segmenters(project):
    """Add imbed's real segmenters."""
    try:
        from imbed.components.segmentation import segmenters as imbed_segmenters

        for name, func in imbed_segmenters.items():
            project.segmenters.register(name)(func)

        print(f"✓ Registered {len(imbed_segmenters)} imbed segmenters")
    except ImportError as e:
        print(f"○ Could not register imbed segmenters: {e}")


def _register_dataset_classes(project):
    """Add imbed's dataset classes."""
    # TODO: Implement when imbed dataset structure is stable
    pass


def _add_utility_methods(project):
    """Add imbed utility methods to project."""
    # TODO: Implement when imbed utility structure is stable
    pass


# Convenience functions


def register_all(project):
    """
    Convenience: register everything from imbed.

    >>> from ef import Project
    >>> from ef.plugins import imbed
    >>> project = Project.create('full')
    >>> imbed.register_all(project)  # doctest: +SKIP
    """
    register(project, include_datasets=True, include_utils=True)


def register_embedders_only(project):
    """
    Register only imbed embedders (no other components).

    >>> from ef import Project
    >>> from ef.plugins import imbed
    >>> project = Project.create('embed_only')
    >>> imbed.register_embedders_only(project)  # doctest: +SKIP
    """
    _register_embedders(project)


def register_ml_only(project):
    """
    Just embedders, planarizers, clusterers - no datasets.

    >>> from ef import Project
    >>> from ef.plugins import imbed
    >>> project = Project.create('ml_only')
    >>> imbed.register_ml_only(project)  # doctest: +SKIP
    """
    register(project, include_datasets=False, include_utils=False)
