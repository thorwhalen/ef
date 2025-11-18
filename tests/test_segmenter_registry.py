"""
Tests for the segmenter registration system.

These tests verify that:
1. Registration functions work correctly
2. Segmenters are properly wrapped to match ef's interface
3. Conditional imports handle missing packages gracefully
4. Metadata is properly attached to registered segmenters
"""

import pytest
from ef import Project
from ef.plugins import segmenter_registry


class TestSegmenterRegistry:
    """Test the segmenter registration system."""

    def test_register_all_segmenters(self):
        """Test that register_all_segmenters works without errors."""
        project = Project.create('test', backend='memory')

        # Should not raise any errors even if packages are missing
        results = segmenter_registry.register_all_segmenters(project, verbose=False)

        # Results should be a dict with package names
        assert isinstance(results, dict)
        assert 'langchain' in results
        assert 'ast' in results
        assert 'tree_sitter' in results
        assert 'spacy' in results
        assert 'nltk' in results
        assert 'tiktoken' in results
        assert 'segtok' in results

        # All values should be non-negative integers
        for package, count in results.items():
            assert isinstance(count, int)
            assert count >= 0

    def test_ast_segmenters_always_available(self):
        """Test that AST segmenters are always registered (built-in)."""
        project = Project.create('test', backend='memory')

        count = segmenter_registry.register_ast_segmenters(project)

        # AST is built-in, should always be available
        assert count > 0

        # Check specific segmenters are registered
        segmenters = project.list_components()['segmenters']
        assert 'ast_python' in segmenters
        assert 'ast_python_statements' in segmenters

    def test_ast_python_segmenter_interface(self):
        """Test that AST Python segmenter follows ef's interface."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_ast_segmenters(project)

        # Get the segmenter function
        segmenter = project.segmenters['ast_python']

        # Test with Python code
        code = '''
def hello():
    """Say hello."""
    print("Hello")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}"
'''

        # Should return dict[str, str]
        result = segmenter(code)

        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, str) for v in result.values())

        # Should have segmented the code
        assert len(result) >= 2  # At least function and class

        # Check that function and class are in the results
        keys = list(result.keys())
        assert any('FunctionDef' in k or 'hello' in k for k in keys)
        assert any('ClassDef' in k or 'Greeter' in k for k in keys)

    def test_ast_python_statements_segmenter(self):
        """Test AST statement-level segmenter."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_ast_segmenters(project)

        segmenter = project.segmenters['ast_python_statements']

        code = '''import os
import sys

DEBUG = True

def main():
    print("Hello")
'''

        result = segmenter(code)

        assert isinstance(result, dict)
        assert len(result) >= 4  # At least 2 imports + DEBUG + function

        # Check that we have import statements
        values = list(result.values())
        assert any('import os' in v for v in values)
        assert any('import sys' in v for v in values)

    def test_segmenter_with_dict_input(self):
        """Test that segmenters handle dict input (already segmented text)."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_ast_segmenters(project)

        segmenter = project.segmenters['ast_python']

        # Input as dict (already segmented)
        input_dict = {
            'part1': 'def foo(): pass',
            'part2': 'def bar(): pass'
        }

        result = segmenter(input_dict)

        assert isinstance(result, dict)
        # Should have concatenated and re-segmented
        assert len(result) >= 1

    def test_segmenter_metadata(self):
        """Test that segmenters are registered with metadata."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_ast_segmenters(project)

        # Check metadata for AST segmenter
        metadata = project.segmenters.get_metadata('ast_python')

        assert metadata is not None
        assert 'package' in metadata
        assert 'description' in metadata
        assert metadata['package'] == 'ast (built-in)'

    def test_langchain_registration(self):
        """Test LangChain registration (conditional on installation)."""
        project = Project.create('test', backend='memory')

        # Should not raise even if not installed
        count = segmenter_registry.register_langchain_segmenters(project)

        assert isinstance(count, int)
        assert count >= 0

        # If LangChain is installed, check some segmenters exist
        if count > 0:
            segmenters = project.list_components()['segmenters']
            # Should have character and recursive splitters
            assert any('langchain_char' in s for s in segmenters)
            assert any('langchain_recursive' in s for s in segmenters)

    def test_tree_sitter_registration(self):
        """Test tree-sitter registration (conditional on installation)."""
        project = Project.create('test', backend='memory')

        count = segmenter_registry.register_tree_sitter_segmenters(project)

        assert isinstance(count, int)
        assert count >= 0

        # If tree-sitter is installed, check Python segmenter exists
        if count > 0:
            segmenters = project.list_components()['segmenters']
            assert 'tree_sitter_python' in segmenters

    def test_spacy_registration(self):
        """Test spaCy registration (conditional on installation)."""
        project = Project.create('test', backend='memory')

        count = segmenter_registry.register_spacy_segmenters(project)

        assert isinstance(count, int)
        assert count >= 0

        if count > 0:
            segmenters = project.list_components()['segmenters']
            assert 'spacy_sentences' in segmenters

    def test_nltk_registration(self):
        """Test NLTK registration (conditional on installation)."""
        project = Project.create('test', backend='memory')

        count = segmenter_registry.register_nltk_segmenters(project)

        assert isinstance(count, int)
        assert count >= 0

        if count > 0:
            segmenters = project.list_components()['segmenters']
            assert 'nltk_sentences' in segmenters

    def test_tiktoken_registration(self):
        """Test tiktoken registration (conditional on installation)."""
        project = Project.create('test', backend='memory')

        count = segmenter_registry.register_tiktoken_segmenters(project)

        assert isinstance(count, int)
        assert count >= 0

        if count > 0:
            segmenters = project.list_components()['segmenters']
            assert any('tiktoken' in s for s in segmenters)

    def test_segtok_registration(self):
        """Test segtok registration (conditional on installation)."""
        project = Project.create('test', backend='memory')

        count = segmenter_registry.register_segtok_segmenters(project)

        assert isinstance(count, int)
        assert count >= 0

        if count > 0:
            segmenters = project.list_components()['segmenters']
            assert 'segtok_sentences' in segmenters

    def test_segmenter_in_pipeline(self):
        """Test using a registered segmenter in a pipeline."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_ast_segmenters(project)

        # Add Python code as source
        code = '''
def hello():
    print("Hello")

def goodbye():
    print("Goodbye")
'''
        project.add_source('code', code)

        # Create pipeline with AST segmenter
        project.create_pipeline(
            'analyze',
            segmenter='ast_python',
            embedder='simple'
        )

        # Run pipeline
        results = project.run_pipeline('analyze')

        # Should have segments and embeddings
        assert 'segments' in results
        assert 'embeddings' in results

        # Segments should be split by functions
        assert len(results['segments']) >= 2

        # Each segment should have an embedding
        assert len(results['embeddings']) == len(results['segments'])

    def test_empty_source_handling(self):
        """Test that segmenters handle empty sources gracefully."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_ast_segmenters(project)

        segmenter = project.segmenters['ast_python']

        # Empty string
        result = segmenter('')
        assert isinstance(result, dict)

        # Empty dict
        result = segmenter({})
        assert isinstance(result, dict)

    def test_invalid_python_code_handling(self):
        """Test that Python segmenters handle syntax errors gracefully."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_ast_segmenters(project)

        segmenter = project.segmenters['ast_python']

        # Invalid Python code
        invalid_code = 'def foo(: pass'

        # Should not crash, should return the source as-is or handle gracefully
        result = segmenter(invalid_code)
        assert isinstance(result, dict)
        # Should have a 'main' key with the original source
        assert 'main' in result


class TestSegmenterRegistryIntegration:
    """Integration tests for the segmenter registry."""

    def test_multiple_registrations_no_conflicts(self):
        """Test that multiple registration calls don't conflict."""
        project = Project.create('test', backend='memory')

        # Register AST multiple times
        count1 = segmenter_registry.register_ast_segmenters(project)
        count2 = segmenter_registry.register_ast_segmenters(project)

        # Counts should be the same (not additive)
        assert count1 == count2

        # Segmenters should still work
        segmenters = project.list_components()['segmenters']
        assert 'ast_python' in segmenters

    def test_register_all_verbose_output(self, capsys):
        """Test that verbose mode prints status information."""
        project = Project.create('test', backend='memory')

        segmenter_registry.register_all_segmenters(project, verbose=True)

        # Capture printed output
        captured = capsys.readouterr()

        # Should print status for AST (always available)
        assert 'AST' in captured.out or 'ast' in captured.out.lower()

        # Should show total count
        assert 'Total' in captured.out or 'total' in captured.out.lower()

    def test_segmenter_list_components(self):
        """Test that list_components returns all registered segmenters."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_all_segmenters(project, verbose=False)

        components = project.list_components()

        assert 'segmenters' in components
        assert isinstance(components['segmenters'], list)

        # Should have at least AST segmenters
        assert len(components['segmenters']) >= 2
        assert 'ast_python' in components['segmenters']

    def test_segmenter_function_callable(self):
        """Test that all registered segmenters are callable."""
        project = Project.create('test', backend='memory')
        segmenter_registry.register_all_segmenters(project, verbose=False)

        segmenters = project.list_components()['segmenters']

        for name in segmenters:
            segmenter = project.segmenters[name]
            assert callable(segmenter), f"Segmenter {name} is not callable"

    def test_real_world_example(self):
        """Test a real-world usage example."""
        project = Project.create('documentation', backend='memory')

        # Register all segmenters
        results = segmenter_registry.register_all_segmenters(project, verbose=False)

        # Add a document
        text = """
        # Introduction

        This is a sample document.
        It has multiple sentences.

        ## Code Example

        Here's some Python code:

        def hello():
            print("Hello, World!")

        ## Conclusion

        That's all!
        """

        project.add_source('doc', text)

        # Try using an AST segmenter (always available)
        project.create_pipeline('analyze', segmenter='ast_python_statements', embedder='simple')

        # This should work without errors
        result = project.run_pipeline('analyze')

        assert 'segments' in result
        assert 'embeddings' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
