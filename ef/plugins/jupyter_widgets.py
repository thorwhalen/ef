"""
Jupyter notebook integration and interactive widgets.

Provides interactive widgets for exploring segmentation in Jupyter notebooks.
"""

from typing import Optional, Any
import json


class SegmentationExplorer:
    """Interactive segmentation explorer for Jupyter."""

    def __init__(self, project, text: str):
        """
        Initialize explorer.

        Args:
            project: EF project
            text: Text to explore
        """
        self.project = project
        self.text = text
        self.current_segmenter = None
        self.current_segments = None

    def show(self):
        """Display the interactive widget."""
        try:
            from IPython.display import display, HTML
            import ipywidgets as widgets

            # Create segmenter dropdown
            segmenter_names = list(self.project.segmenters.keys())
            segmenter_dropdown = widgets.Dropdown(
                options=segmenter_names,
                description='Segmenter:',
                value=segmenter_names[0] if segmenter_names else None
            )

            # Create output area
            output = widgets.Output()

            # Define update function
            def on_segmenter_change(change):
                with output:
                    output.clear_output()
                    segmenter_name = change['new']
                    self._display_segmentation(segmenter_name)

            segmenter_dropdown.observe(on_segmenter_change, names='value')

            # Initial display
            with output:
                self._display_segmentation(segmenter_dropdown.value)

            # Layout
            display(widgets.VBox([segmenter_dropdown, output]))

        except ImportError:
            print("⚠ ipywidgets not installed. Install with: pip install ipywidgets")
            print("\nFalling back to simple display...")
            self._simple_display()

    def _display_segmentation(self, segmenter_name: str):
        """Display segmentation results."""
        try:
            from IPython.display import display, HTML
        except ImportError:
            self._simple_display()
            return

        if segmenter_name not in self.project.segmenters:
            print(f"Segmenter '{segmenter_name}' not found")
            return

        seg_func = self.project.segmenters[segmenter_name]
        segments = seg_func(self.text)

        self.current_segmenter = segmenter_name
        self.current_segments = segments

        # Create HTML display
        html = f"<h3>Segmentation: {segmenter_name}</h3>"
        html += f"<p><strong>Total segments:</strong> {len(segments)}</p>"

        # Display segments with colors
        colors = ['#e3f2fd', '#fff3e0', '#f3e5f5', '#e8f5e9', '#fce4ec']

        html += "<div style='margin-top: 20px;'>"
        for i, (key, text) in enumerate(segments.items()):
            color = colors[i % len(colors)]
            html += f"""
            <div style='background: {color}; padding: 10px; margin: 5px 0; border-left: 3px solid #1976d2;'>
                <strong>{key}</strong><br/>
                {self._escape_html(text[:200])}{'...' if len(text) > 200 else ''}
            </div>
            """

        html += "</div>"

        display(HTML(html))

    def _simple_display(self):
        """Simple text-based display."""
        print(f"Text length: {len(self.text)} characters")
        print(f"Available segmenters: {', '.join(self.project.segmenters.keys())}")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;')
                .replace('\n', '<br/>'))


class SegmentationComparison:
    """Compare multiple segmentation strategies in Jupyter."""

    def __init__(self, project, text: str, segmenters: list[str]):
        """
        Initialize comparison widget.

        Args:
            project: EF project
            text: Text to segment
            segmenters: List of segmenter names to compare
        """
        self.project = project
        self.text = text
        self.segmenters = segmenters
        self.results = {}

    def show(self):
        """Display comparison."""
        try:
            from IPython.display import display, HTML
            import ipywidgets as widgets
        except ImportError:
            self._simple_comparison()
            return

        # Run all segmenters
        for seg_name in self.segmenters:
            if seg_name in self.project.segmenters:
                seg_func = self.project.segmenters[seg_name]
                self.results[seg_name] = seg_func(self.text)

        # Create comparison table
        html = "<h3>Segmentation Comparison</h3>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        html += "<tr style='background: #1976d2; color: white;'>"
        html += "<th style='padding: 10px; border: 1px solid #ddd;'>Segmenter</th>"
        html += "<th style='padding: 10px; border: 1px solid #ddd;'>Segments</th>"
        html += "<th style='padding: 10px; border: 1px solid #ddd;'>Avg Length</th>"
        html += "<th style='padding: 10px; border: 1px solid #ddd;'>Preview</th>"
        html += "</tr>"

        for seg_name, segments in self.results.items():
            avg_len = sum(len(s) for s in segments.values()) // len(segments) if segments else 0
            preview = list(segments.values())[0][:50] + '...' if segments else 'N/A'

            html += "<tr>"
            html += f"<td style='padding: 10px; border: 1px solid #ddd;'><strong>{seg_name}</strong></td>"
            html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{len(segments)}</td>"
            html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{avg_len}</td>"
            html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{self._escape_html(preview)}</td>"
            html += "</tr>"

        html += "</table>"

        display(HTML(html))

    def _simple_comparison(self):
        """Simple text-based comparison."""
        print("Segmentation Comparison")
        print("=" * 60)

        for seg_name in self.segmenters:
            if seg_name in self.project.segmenters:
                seg_func = self.project.segmenters[seg_name]
                segments = seg_func(self.text)
                self.results[seg_name] = segments

                avg_len = sum(len(s) for s in segments.values()) // len(segments) if segments else 0
                print(f"{seg_name}: {len(segments)} segments, avg {avg_len} chars")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;')
                .replace('\n', ' '))


class SegmentationVisualizer:
    """Visualize segmentation boundaries in Jupyter."""

    def __init__(self, text: str, segments: dict[str, str]):
        """
        Initialize visualizer.

        Args:
            text: Original text
            segments: Segmented text
        """
        self.text = text
        self.segments = segments

    def show(self, style: str = 'highlight'):
        """
        Display visualization.

        Args:
            style: Visualization style ('highlight', 'boundaries', 'interactive')
        """
        try:
            from IPython.display import display, HTML
        except ImportError:
            self._simple_visualization()
            return

        if style == 'highlight':
            self._show_highlight()
        elif style == 'boundaries':
            self._show_boundaries()
        elif style == 'interactive':
            self._show_interactive()
        else:
            print(f"Unknown style: {style}")

    def _show_highlight(self):
        """Show with highlighted segments."""
        from IPython.display import display, HTML

        html = "<div style='font-family: monospace; line-height: 1.8;'>"

        colors = [
            '#ffebee', '#e3f2fd', '#f3e5f5', '#e8f5e9',
            '#fff3e0', '#fce4ec', '#e0f2f1', '#f9fbe7'
        ]

        for i, (key, text) in enumerate(self.segments.items()):
            color = colors[i % len(colors)]
            html += f"<span style='background: {color}; padding: 2px 4px; margin: 1px;' title='{key}'>"
            html += self._escape_html(text)
            html += "</span>"

        html += "</div>"

        display(HTML(html))

    def _show_boundaries(self):
        """Show with boundary markers."""
        from IPython.display import display, HTML

        html = "<div style='font-family: monospace; white-space: pre-wrap;'>"

        for i, (key, text) in enumerate(self.segments.items()):
            html += f"<div style='border-top: 2px solid #1976d2; padding: 10px 0;'>"
            html += f"<div style='color: #1976d2; font-size: 0.8em;'>[{key}]</div>"
            html += self._escape_html(text)
            html += "</div>"

        html += "</div>"

        display(HTML(html))

    def _show_interactive(self):
        """Show with interactive controls."""
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML
        except ImportError:
            self._simple_visualization()
            return

        segment_keys = list(self.segments.keys())

        # Create slider
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(segment_keys) - 1,
            description='Segment:',
            continuous_update=False
        )

        # Create output
        output = widgets.Output()

        def on_slider_change(change):
            with output:
                output.clear_output()
                idx = change['new']
                key = segment_keys[idx]
                text = self.segments[key]

                html = f"<h4>Segment {idx + 1} of {len(segment_keys)}: {key}</h4>"
                html += f"<div style='padding: 10px; background: #f5f5f5; border-left: 3px solid #1976d2;'>"
                html += f"<p><strong>Length:</strong> {len(text)} characters</p>"
                html += f"<p>{self._escape_html(text)}</p>"
                html += "</div>"

                display(HTML(html))

        slider.observe(on_slider_change, names='value')

        # Initial display
        with output:
            key = segment_keys[0]
            text = self.segments[key]
            html = f"<h4>Segment 1 of {len(segment_keys)}: {key}</h4>"
            html += f"<div style='padding: 10px; background: #f5f5f5; border-left: 3px solid #1976d2;'>"
            html += f"<p><strong>Length:</strong> {len(text)} characters</p>"
            html += f"<p>{self._escape_html(text)}</p>"
            html += "</div>"
            display(HTML(html))

        display(widgets.VBox([slider, output]))

    def _simple_visualization(self):
        """Simple text-based visualization."""
        print("Segmentation Visualization")
        print("=" * 60)

        for i, (key, text) in enumerate(self.segments.items(), 1):
            print(f"\n[Segment {i}: {key}]")
            print("-" * 60)
            print(text[:200] + ('...' if len(text) > 200 else ''))

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;')
                .replace('\n', '<br/>'))


def notebook_segment(
    project,
    text: str,
    segmenter: str = 'sentences',
    visualize: bool = True
) -> dict[str, str]:
    """
    Segment text in a notebook with optional visualization.

    Args:
        project: EF project
        text: Text to segment
        segmenter: Segmenter name
        visualize: Whether to show visualization

    Returns:
        Segments dictionary
    """
    if segmenter not in project.segmenters:
        print(f"Error: Segmenter '{segmenter}' not found")
        return {}

    seg_func = project.segmenters[segmenter]
    segments = seg_func(text)

    if visualize:
        viz = SegmentationVisualizer(text, segments)
        viz.show()

    return segments


def notebook_compare(
    project,
    text: str,
    segmenters: list[str]
) -> dict[str, dict[str, str]]:
    """
    Compare segmenters in a notebook.

    Args:
        project: EF project
        text: Text to segment
        segmenters: List of segmenter names

    Returns:
        Dictionary mapping segmenter names to segments
    """
    comparison = SegmentationComparison(project, text, segmenters)
    comparison.show()
    return comparison.results


def notebook_explore(project, text: str):
    """
    Launch interactive explorer in notebook.

    Args:
        project: EF project
        text: Text to explore
    """
    explorer = SegmentationExplorer(project, text)
    explorer.show()


def export_notebook_example(output_path: str = 'segmentation_demo.ipynb'):
    """
    Export example Jupyter notebook.

    Args:
        output_path: Path for output notebook
    """
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# EF Segmentation Demo\n", "\n", "Interactive demonstration of EF segmentation features."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "from ef import Project\n",
                    "from ef.plugins.jupyter_widgets import notebook_explore, notebook_compare, notebook_segment\n",
                    "\n",
                    "# Create project with auto-registration\n",
                    "project = Project.create('demo', auto_register_segmenters=True)\n",
                    "\n",
                    "# Sample text\n",
                    "text = '''Your sample text here...'''"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Interactive Explorer\n", "\n", "Explore different segmentation strategies:"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["notebook_explore(project, text)"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Compare Segmenters\n", "\n", "Compare multiple segmentation strategies:"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["results = notebook_compare(project, text, ['sentences', 'paragraphs', 'lines'])"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Visualize Segmentation\n", "\n", "Visualize segment boundaries:"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": ["segments = notebook_segment(project, text, 'sentences', visualize=True)"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Exported example notebook to: {output_path}")


def register_jupyter_helpers(project) -> int:
    """Register Jupyter helper functions."""
    count = 0

    # Note: These are helper functions, not segmenters
    # They're for use in notebooks

    print(f"✓ Jupyter integration ready (use helper functions)")
    return count
