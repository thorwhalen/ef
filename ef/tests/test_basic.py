"""Basic tests for ef package."""

import pytest
from ef import Project, Projects, ComponentRegistry


def test_project_creation():
    """Test basic project creation."""
    project = Project.create('test', backend='memory')
    assert project.project_id == 'test'
    assert len(project.segments) == 0


def test_add_source():
    """Test adding source data."""
    project = Project.create('test', backend='memory')
    project.add_source('doc1', 'Sample text')

    assert 'doc1' in project.segments
    assert project.segments['doc1'] == 'Sample text'


def test_component_discovery():
    """Test listing components."""
    project = Project.create('test', backend='memory')
    components = project.list_components()

    assert 'embedders' in components
    assert 'clusterers' in components
    assert 'simple' in components['embedders']


def test_pipeline_creation():
    """Test creating a pipeline."""
    project = Project.create('test', backend='memory')
    project.create_pipeline('test_pipe', embedder='simple')

    assert 'test_pipe' in project.pipelines
    assert 'test_pipe' in project.list_pipelines()


def test_pipeline_execution():
    """Test running a pipeline."""
    project = Project.create('test', backend='memory')
    project.add_source('doc1', 'Test document')

    project.create_pipeline('test', embedder='simple')
    results = project.run_pipeline('test')

    assert 'embeddings' in results
    assert 'doc1' in results['embeddings']
    assert len(results['embeddings']['doc1']) == 3  # simple embedder returns 3D


def test_quick_embed():
    """Test quick embed functionality."""
    project = Project.create('test', backend='memory')
    embeddings = project.quick_embed('Test text')

    assert 'main' in embeddings
    assert len(embeddings['main']) == 3


def test_custom_component():
    """Test registering custom component."""
    project = Project.create('test', backend='memory')

    @project.embedders.register('custom', dimension=2)
    def custom_embedder(segments):
        return {k: [1.0, 2.0] for k in segments}

    assert 'custom' in project.embedders
    meta = project.embedders.get_metadata('custom')
    assert meta['dimension'] == 2


def test_component_registry():
    """Test ComponentRegistry."""
    registry = ComponentRegistry('test')

    # Add component
    registry['func1'] = lambda x: x * 2

    # Test access
    assert 'func1' in registry
    assert len(registry) == 1
    assert list(registry.keys()) == ['func1']

    # Test decorator
    @registry.register('func2', param=42)
    def func2(x):
        return x + 1

    assert 'func2' in registry
    assert registry.get_metadata('func2')['param'] == 42


def test_projects_manager():
    """Test Projects manager."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        projects = Projects(root_dir=tmpdir)

        # Create project
        proj1 = projects.create_project('proj1', backend='memory')
        assert proj1.project_id == 'proj1'

        # Access project
        assert 'proj1' in projects
        retrieved = projects['proj1']
        assert retrieved.project_id == 'proj1'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
