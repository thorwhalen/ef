"""
Configuration management for segmentation strategies.

This module provides tools to save, load, and share segmentation
configurations for reproducibility.
"""

from typing import Any, Callable, Optional, Union
import json
import yaml
import warnings
from pathlib import Path


class SegmentationConfig:
    """
    Segmentation configuration.

    Attributes:
        segmenter: Name of segmenter
        params: Segmenter parameters
        preprocessors: List of preprocessing steps
        postprocessors: List of postprocessing steps
        metadata: Additional metadata
    """

    def __init__(
        self,
        segmenter: str,
        params: Optional[dict] = None,
        preprocessors: Optional[list] = None,
        postprocessors: Optional[list] = None,
        metadata: Optional[dict] = None
    ):
        """Initialize configuration."""
        self.segmenter = segmenter
        self.params = params or {}
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'segmenter': self.segmenter,
            'params': self.params,
            'preprocessors': self.preprocessors,
            'postprocessors': self.postprocessors,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SegmentationConfig':
        """Create from dictionary."""
        return cls(
            segmenter=data['segmenter'],
            params=data.get('params', {}),
            preprocessors=data.get('preprocessors', []),
            postprocessors=data.get('postprocessors', []),
            metadata=data.get('metadata', {})
        )

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'SegmentationConfig':
        """Create from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'SegmentationConfig':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def create_config(
    segmenter: str,
    params: Optional[dict] = None,
    preprocessors: Optional[list] = None,
    postprocessors: Optional[list] = None,
    **metadata
) -> SegmentationConfig:
    """
    Create a segmentation configuration.

    Args:
        segmenter: Name of segmenter
        params: Segmenter parameters
        preprocessors: Preprocessing steps
        postprocessors: Postprocessing steps
        **metadata: Additional metadata

    Returns:
        SegmentationConfig instance

    Example:
        >>> config = create_config(
        ...     segmenter='sliding_window',
        ...     params={'window_size': 500, 'stride': 250},
        ...     preprocessors=['lowercase', 'remove_urls'],
        ...     author='user@example.com',
        ...     description='Sliding window for articles'
        ... )
    """
    return SegmentationConfig(
        segmenter=segmenter,
        params=params,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        metadata=metadata
    )


def save(config: SegmentationConfig, file_path: str, format: str = 'auto') -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        file_path: Path to save to
        format: Format ('yaml', 'json', or 'auto' to detect from extension)

    Example:
        >>> save(config, 'my_strategy.yaml')
    """
    if format == 'auto':
        ext = Path(file_path).suffix.lower()
        if ext in ['.yaml', '.yml']:
            format = 'yaml'
        elif ext == '.json':
            format = 'json'
        else:
            format = 'yaml'  # default

    with open(file_path, 'w') as f:
        if format == 'yaml':
            f.write(config.to_yaml())
        elif format == 'json':
            f.write(config.to_json())
        else:
            raise ValueError(f"Unknown format: {format}")


def load(file_path: str, format: str = 'auto') -> SegmentationConfig:
    """
    Load configuration from file.

    Args:
        file_path: Path to load from
        format: Format ('yaml', 'json', or 'auto')

    Returns:
        SegmentationConfig instance

    Example:
        >>> config = load('my_strategy.yaml')
    """
    if format == 'auto':
        ext = Path(file_path).suffix.lower()
        if ext in ['.yaml', '.yml']:
            format = 'yaml'
        elif ext == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot auto-detect format from: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    if format == 'yaml':
        return SegmentationConfig.from_yaml(content)
    elif format == 'json':
        return SegmentationConfig.from_json(content)
    else:
        raise ValueError(f"Unknown format: {format}")


def build_from_config(project, config: SegmentationConfig) -> Callable:
    """
    Build a segmenter from configuration.

    Args:
        project: Project instance
        config: SegmentationConfig

    Returns:
        Configured segmenter function

    Example:
        >>> config = load('strategy.yaml')
        >>> segmenter = build_from_config(project, config)
        >>> segments = segmenter(text)
    """
    if config.segmenter not in project.segmenters:
        raise ValueError(f"Segmenter '{config.segmenter}' not found")

    base_segmenter = project.segmenters[config.segmenter]

    # Apply parameters
    if config.params:
        from functools import partial
        segmenter = partial(base_segmenter, **config.params)
    else:
        segmenter = base_segmenter

    # Apply preprocessors
    if config.preprocessors:
        original_segmenter = segmenter

        def preprocessed_segmenter(source):
            processed_source = source

            for preprocessor in config.preprocessors:
                processed_source = apply_preprocessor(processed_source, preprocessor)

            return original_segmenter(processed_source)

        segmenter = preprocessed_segmenter

    # Apply postprocessors
    if config.postprocessors:
        original_segmenter = segmenter

        def postprocessed_segmenter(source):
            segments = original_segmenter(source)

            for postprocessor in config.postprocessors:
                segments = apply_postprocessor(segments, postprocessor)

            return segments

        segmenter = postprocessed_segmenter

    return segmenter


def apply_preprocessor(source: Any, preprocessor: Union[str, dict]) -> Any:
    """Apply a preprocessing step."""
    if isinstance(preprocessor, str):
        # Built-in preprocessors
        if preprocessor == 'lowercase':
            if isinstance(source, str):
                return source.lower()
            elif isinstance(source, dict):
                return {k: v.lower() for k, v in source.items()}

        elif preprocessor == 'remove_urls':
            import re
            url_pattern = r'https?://\S+'

            if isinstance(source, str):
                return re.sub(url_pattern, '', source)
            elif isinstance(source, dict):
                return {k: re.sub(url_pattern, '', v) for k, v in source.items()}

        elif preprocessor == 'strip_whitespace':
            if isinstance(source, str):
                return source.strip()
            elif isinstance(source, dict):
                return {k: v.strip() for k, v in source.items()}

    # Unknown preprocessor, return as-is
    warnings.warn(f"Unknown preprocessor: {preprocessor}")
    return source


def apply_postprocessor(segments: dict[str, str], postprocessor: Union[str, dict]) -> dict[str, str]:
    """Apply a postprocessing step."""
    if isinstance(postprocessor, str):
        # Built-in postprocessors
        if postprocessor == 'deduplicate':
            seen = set()
            deduped = {}

            for key, text in segments.items():
                if text not in seen:
                    seen.add(text)
                    deduped[key] = text

            return deduped

        elif postprocessor == 'min_length_filter':
            # Default min length 10
            return {k: v for k, v in segments.items() if len(v) >= 10}

        elif postprocessor == 'sort_by_length':
            sorted_items = sorted(segments.items(), key=lambda x: len(x[1]))
            return dict(sorted_items)

    elif isinstance(postprocessor, dict):
        # Parameterized postprocessor
        if postprocessor.get('type') == 'min_length_filter':
            min_len = postprocessor.get('min_length', 10)
            return {k: v for k, v in segments.items() if len(v) >= min_len}

        elif postprocessor.get('type') == 'max_length_filter':
            max_len = postprocessor.get('max_length', 10000)
            return {k: v for k, v in segments.items() if len(v) <= max_len}

    # Unknown postprocessor, return as-is
    warnings.warn(f"Unknown postprocessor: {postprocessor}")
    return segments


# Configuration presets
PRESETS = {
    'article': {
        'segmenter': 'langchain_recursive_1000',
        'params': {},
        'preprocessors': ['strip_whitespace'],
        'postprocessors': ['deduplicate'],
        'metadata': {
            'description': 'Good for articles and blog posts',
            'use_cases': ['semantic_search', 'summarization']
        }
    },
    'code_python': {
        'segmenter': 'ast_python',
        'params': {},
        'preprocessors': [],
        'postprocessors': [],
        'metadata': {
            'description': 'Python code segmentation',
            'use_cases': ['code_analysis', 'code_search']
        }
    },
    'documentation': {
        'segmenter': 'markdown_hierarchical',
        'params': {},
        'preprocessors': [],
        'postprocessors': [],
        'metadata': {
            'description': 'Markdown documentation',
            'use_cases': ['documentation_search', 'help_systems']
        }
    },
    'chat_context': {
        'segmenter': 'sliding_window',
        'params': {'window_size': 1000, 'stride': 500},
        'preprocessors': ['strip_whitespace'],
        'postprocessors': [],
        'metadata': {
            'description': 'Chat context with overlap',
            'use_cases': ['chatbots', 'conversation_analysis']
        }
    },
    'llm_preparation': {
        'segmenter': 'tiktoken_gpt-4',
        'params': {},
        'preprocessors': [],
        'postprocessors': [],
        'metadata': {
            'description': 'Prepare text for GPT-4',
            'use_cases': ['llm_input', 'token_optimization']
        }
    }
}


def get_preset(preset_name: str) -> SegmentationConfig:
    """
    Get a preset configuration.

    Args:
        preset_name: Name of preset

    Returns:
        SegmentationConfig

    Available presets:
        - 'article': For articles and blog posts
        - 'code_python': For Python code
        - 'documentation': For markdown docs
        - 'chat_context': For chat/conversation
        - 'llm_preparation': For LLM input

    Example:
        >>> config = get_preset('article')
        >>> save(config, 'my_article_strategy.yaml')
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    preset_data = PRESETS[preset_name]
    return SegmentationConfig.from_dict(preset_data)


def list_presets() -> dict[str, str]:
    """
    List available configuration presets.

    Returns:
        Dict mapping preset names to descriptions
    """
    return {
        name: data['metadata']['description']
        for name, data in PRESETS.items()
    }


def export_current_setup(
    project,
    segmenter_name: str,
    output_path: str,
    **metadata
) -> None:
    """
    Export current segmenter setup to configuration file.

    Args:
        project: Project instance
        segmenter_name: Name of segmenter
        output_path: Where to save configuration
        **metadata: Additional metadata

    Example:
        >>> export_current_setup(
        ...     project,
        ...     'sliding_window',
        ...     'my_setup.yaml',
        ...     author='me',
        ...     description='My custom setup'
        ... )
    """
    if segmenter_name not in project.segmenters:
        raise ValueError(f"Segmenter '{segmenter_name}' not found")

    # Get metadata if available
    seg_metadata = project.segmenters.get_metadata(segmenter_name)

    config = create_config(
        segmenter=segmenter_name,
        params=seg_metadata or {},
        **metadata
    )

    save(config, output_path)


def compare_configs(config1: SegmentationConfig, config2: SegmentationConfig) -> dict:
    """
    Compare two configurations.

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        Dict with comparison results
    """
    return {
        'same_segmenter': config1.segmenter == config2.segmenter,
        'segmenter1': config1.segmenter,
        'segmenter2': config2.segmenter,
        'params_diff': {
            k: (config1.params.get(k), config2.params.get(k))
            for k in set(config1.params.keys()) | set(config2.params.keys())
            if config1.params.get(k) != config2.params.get(k)
        },
        'preprocessors1': config1.preprocessors,
        'preprocessors2': config2.preprocessors,
        'postprocessors1': config1.postprocessors,
        'postprocessors2': config2.postprocessors,
    }
