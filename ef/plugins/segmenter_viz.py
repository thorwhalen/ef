"""
Visualization tools for segmentation analysis.

This module provides visualization capabilities for understanding
and comparing segmentation strategies.
"""

from typing import Any, Optional, Union
import warnings


def plot_segmentation(
    text: str,
    segments: dict[str, str],
    show_overlap: bool = False,
    highlight_boundaries: bool = True,
    output_file: Optional[str] = None
) -> None:
    """
    Plot segmentation boundaries on text.

    Args:
        text: Original text
        segments: Segmented text
        show_overlap: Whether to highlight overlapping regions
        highlight_boundaries: Whether to mark boundaries
        output_file: Optional file to save plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate segment positions in original text
        positions = []
        for key, segment in segments.items():
            # Find segment in text
            idx = text.find(segment)
            if idx >= 0:
                positions.append((key, idx, idx + len(segment)))
        
        # Plot
        y_pos = 0
        colors = plt.cm.Set3(range(len(positions)))
        
        for i, (key, start, end) in enumerate(positions):
            ax.barh(y_pos, end - start, left=start, height=0.8, 
                   color=colors[i], alpha=0.6, label=key)
            y_pos += 1
        
        ax.set_xlabel('Character Position')
        ax.set_ylabel('Segments')
        ax.set_title('Segmentation Visualization')
        ax.set_xlim(0, len(text))
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")


def compare_visual(
    text: str,
    project,
    strategies: list[str],
    output_file: Optional[str] = None
) -> None:
    """
    Visually compare multiple segmentation strategies.
    
    Args:
        text: Text to segment
        project: Project instance
        strategies: List of segmenter names
        output_file: Optional output file
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(strategies), 1, figsize=(14, 4*len(strategies)))
        
        if len(strategies) == 1:
            axes = [axes]
        
        for idx, strategy in enumerate(strategies):
            if strategy not in project.segmenters:
                continue
                
            segmenter = project.segmenters[strategy]
            segments = segmenter(text)
            
            ax = axes[idx]
            
            # Plot segments as horizontal bars
            positions = []
            for key, segment in segments.items():
                start = text.find(segment)
                if start >= 0:
                    positions.append((start, start + len(segment)))
            
            for i, (start, end) in enumerate(positions):
                ax.barh(0, end - start, left=start, height=0.5, alpha=0.7)
            
            ax.set_title(f'{strategy} ({len(segments)} segments)')
            ax.set_xlim(0, len(text))
            ax.set_yticks([])
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
        else:
            plt.show()
            
        plt.close()
        
    except ImportError:
        print("matplotlib not available")


def plot_metrics_comparison(
    comparison_results: dict,
    output_file: Optional[str] = None
) -> None:
    """
    Plot comparison metrics as bar charts.
    
    Args:
        comparison_results: Results from compare_segmenters
        output_file: Optional output file
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        segmenters = list(comparison_results.keys())
        metrics = ['count', 'avg_length']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, metric in enumerate(metrics):
            values = [
                comparison_results[seg]['metrics'].get(metric, 0)
                for seg in segmenters
            ]
            
            axes[i].bar(range(len(segmenters)), values)
            axes[i].set_xticks(range(len(segmenters)))
            axes[i].set_xticklabels(segmenters, rotation=45, ha='right')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
        else:
            plt.show()
            
        plt.close()
        
    except ImportError:
        print("matplotlib not available")


def interactive_explorer(project, text: str, port: int = 8050) -> None:
    """
    Launch interactive web UI for exploring segmentation.
    
    Requires: dash, plotly
    
    Args:
        project: Project instance
        text: Text to explore
        port: Port for web server
    """
    try:
        import dash
        from dash import dcc, html, Input, Output
        import plotly.graph_objects as go
        
        app = dash.Dash(__name__)
        
        segmenters = project.list_components()['segmenters']
        
        app.layout = html.Div([
            html.H1("Segmentation Explorer"),
            
            html.Label("Select Segmenter:"),
            dcc.Dropdown(
                id='segmenter-dropdown',
                options=[{'label': s, 'value': s} for s in segmenters],
                value=segmenters[0] if segmenters else None
            ),
            
            html.Div(id='metrics-display'),
            dcc.Graph(id='segmentation-plot'),
            html.Div(id='segments-list', style={'maxHeight': '400px', 'overflow': 'scroll'})
        ])
        
        @app.callback(
            [Output('metrics-display', 'children'),
             Output('segmentation-plot', 'figure'),
             Output('segments-list', 'children')],
            [Input('segmenter-dropdown', 'value')]
        )
        def update_display(segmenter_name):
            if not segmenter_name or segmenter_name not in project.segmenters:
                return "No segmenter selected", {}, ""
            
            segmenter = project.segmenters[segmenter_name]
            segments = segmenter(text)
            
            # Metrics
            from ef.plugins.segmenter_utils import analyze_segmentation
            metrics = analyze_segmentation(segments)
            
            metrics_html = html.Div([
                html.H3("Metrics"),
                html.P(f"Count: {metrics['count']}"),
                html.P(f"Avg Length: {metrics['avg_length']:.1f} chars"),
                html.P(f"Total: {metrics['total_chars']} chars"),
            ])
            
            # Plot
            fig = go.Figure()
            lengths = [len(s) for s in segments.values()]
            fig.add_trace(go.Bar(
                x=list(range(len(segments))),
                y=lengths,
                name='Segment Lengths'
            ))
            fig.update_layout(title='Segment Lengths', xaxis_title='Segment', yaxis_title='Length')
            
            # Segments list
            segments_html = html.Div([
                html.H3("Segments"),
                *[html.Div([
                    html.H4(key),
                    html.P(text[:200] + '...' if len(text) > 200 else text)
                ]) for key, text in segments.items()]
            ])
            
            return metrics_html, fig, segments_html
        
        print(f"Starting interactive explorer at http://localhost:{port}")
        app.run_server(debug=False, port=port)
        
    except ImportError as e:
        print(f"Required packages not available: {e}")
        print("Install with: pip install dash plotly")


def export_html_report(
    project,
    text: str,
    strategies: list[str],
    output_file: str = 'segmentation_report.html'
) -> None:
    """
    Generate HTML report comparing segmentation strategies.
    
    Args:
        project: Project instance
        text: Text to analyze
        strategies: List of strategies to compare
        output_file: Output HTML file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Segmentation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .strategy { border: 1px solid #ccc; padding: 15px; margin: 10px 0; }
            .segment { background: #f0f0f0; padding: 10px; margin: 5px 0; border-left: 3px solid #007bff; }
            .metrics { background: #e9ecef; padding: 10px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #007bff; color: white; }
        </style>
    </head>
    <body>
        <h1>Segmentation Analysis Report</h1>
        <p><strong>Text Length:</strong> {} characters</p>
    """.format(len(text))
    
    from ef.plugins.segmenter_utils import analyze_segmentation
    
    for strategy in strategies:
        if strategy not in project.segmenters:
            continue
        
        segmenter = project.segmenters[strategy]
        segments = segmenter(text)
        metrics = analyze_segmentation(segments)
        
        html_content += f"""
        <div class="strategy">
            <h2>{strategy}</h2>
            <div class="metrics">
                <h3>Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Segment Count</td><td>{metrics['count']}</td></tr>
                    <tr><td>Avg Length</td><td>{metrics['avg_length']:.1f} chars</td></tr>
                    <tr><td>Min Length</td><td>{metrics['min_length']} chars</td></tr>
                    <tr><td>Max Length</td><td>{metrics['max_length']} chars</td></tr>
                    <tr><td>Std Dev</td><td>{metrics['std_dev']:.1f}</td></tr>
                </table>
            </div>
            
            <h3>Sample Segments</h3>
        """
        
        for i, (key, segment) in enumerate(list(segments.items())[:5]):
            preview = segment[:200] + '...' if len(segment) > 200 else segment
            html_content += f"""
            <div class="segment">
                <strong>{key}</strong> ({len(segment)} chars)<br>
                {preview}
            </div>
            """
        
        if len(segments) > 5:
            html_content += f"<p><em>... and {len(segments) - 5} more segments</em></p>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to {output_file}")
