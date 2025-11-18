"""
Command-line interface tools for segmentation operations.

Provides CLI commands for common segmentation tasks.
"""

import sys
import json
from pathlib import Path
from typing import Optional


def segment_file_cli(
    input_file: str,
    output_file: str = None,
    segmenter: str = 'sentences',
    format: str = 'json',
    project = None
) -> None:
    """
    CLI command to segment a file.

    Args:
        input_file: Path to input file
        output_file: Path to output file (defaults to input + .segmented.json)
        segmenter: Name of segmenter to use
        format: Output format ('json', 'jsonl', 'text')
        project: EF project instance
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get segmenter
    if project is None:
        from ef import Project
        project = Project.create('_cli_temp', auto_register_segmenters=True)

    if segmenter not in project.segmenters:
        print(f"Error: Segmenter '{segmenter}' not found", file=sys.stderr)
        print(f"Available: {', '.join(project.segmenters.keys())}", file=sys.stderr)
        sys.exit(1)

    # Segment
    seg_func = project.segmenters[segmenter]
    segments = seg_func(text)

    # Determine output file
    if output_file is None:
        output_file = str(input_path.with_suffix(f'.segmented.{format}'))

    # Write output
    output_path = Path(output_file)

    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for key, value in segments.items():
                f.write(json.dumps({key: value}, ensure_ascii=False) + '\n')

    elif format == 'text':
        with open(output_path, 'w', encoding='utf-8') as f:
            for key, value in segments.items():
                f.write(f"=== {key} ===\n{value}\n\n")

    else:
        print(f"Error: Unknown format '{format}'", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Segmented {input_file} -> {output_file}")
    print(f"  Segments: {len(segments)}")


def list_segmenters_cli(project = None, verbose: bool = False) -> None:
    """
    CLI command to list available segmenters.

    Args:
        project: EF project instance
        verbose: Show detailed information
    """
    if project is None:
        from ef import Project
        project = Project.create('_cli_temp', auto_register_segmenters=True)

    segmenters = project.segmenters

    print(f"\nAvailable Segmenters ({len(segmenters)}):")
    print("=" * 60)

    for name in sorted(segmenters.keys()):
        if verbose:
            meta = segmenters.get_metadata(name)
            package = meta.get('package', 'unknown')
            desc = meta.get('description', 'No description')
            print(f"  {name}")
            print(f"    Package: {package}")
            print(f"    Description: {desc}")
            print()
        else:
            print(f"  - {name}")

    print()


def benchmark_segmenters_cli(
    input_file: str,
    segmenters: list[str] = None,
    project = None
) -> None:
    """
    CLI command to benchmark segmenters.

    Args:
        input_file: Path to test file
        segmenters: List of segmenter names to benchmark
        project: EF project instance
    """
    from ef.plugins.profiler import profile_segmenter

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get project
    if project is None:
        from ef import Project
        project = Project.create('_cli_temp', auto_register_segmenters=True)

    # Default to all segmenters
    if segmenters is None:
        segmenters = list(project.segmenters.keys())

    print(f"\nBenchmarking {len(segmenters)} segmenters on {input_file}")
    print("=" * 60)

    results = []
    for seg_name in segmenters:
        if seg_name not in project.segmenters:
            print(f"  Skipping unknown segmenter: {seg_name}")
            continue

        seg_func = project.segmenters[seg_name]
        profile = profile_segmenter(seg_func, [text])

        if 'time' in profile and 'throughput' in profile:
            results.append({
                'name': seg_name,
                'time': profile['time']['mean'],
                'throughput': profile['throughput']['mean']
            })

    # Sort by speed
    results.sort(key=lambda x: x['time'])

    print("\nResults (sorted by speed):")
    print(f"{'Segmenter':<30} {'Time (s)':<12} {'Throughput (chars/s)':<20}")
    print("-" * 62)

    for r in results:
        print(f"{r['name']:<30} {r['time']:<12.4f} {r['throughput']:<20.0f}")

    print()


def compare_segmenters_cli(
    input_file: str,
    segmenters: list[str],
    output_html: Optional[str] = None,
    project = None
) -> None:
    """
    CLI command to compare segmenters visually.

    Args:
        input_file: Path to input file
        segmenters: List of segmenter names to compare
        output_html: Optional path to save HTML report
        project: EF project instance
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get project
    if project is None:
        from ef import Project
        project = Project.create('_cli_temp', auto_register_segmenters=True)

    print(f"\nComparing {len(segmenters)} segmenters:")
    print("=" * 60)

    all_segments = {}
    for seg_name in segmenters:
        if seg_name not in project.segmenters:
            print(f"  Skipping unknown segmenter: {seg_name}")
            continue

        seg_func = project.segmenters[seg_name]
        segments = seg_func(text)
        all_segments[seg_name] = segments

        print(f"  {seg_name}: {len(segments)} segments")

    # Generate HTML report if requested
    if output_html:
        try:
            from ef.plugins.segmenter_viz import export_html_report
            export_html_report(text, all_segments, output_html)
            print(f"\n✓ HTML report saved to: {output_html}")
        except Exception as e:
            print(f"\nWarning: Could not generate HTML report: {e}")

    print()


def auto_segment_cli(
    input_file: str,
    output_file: str = None,
    task: str = 'general',
    project = None
) -> None:
    """
    CLI command for automatic segmenter selection.

    Args:
        input_file: Path to input file
        output_file: Path to output file
        task: Task type for auto-detection
        project: EF project instance
    """
    from ef.plugins.segmenter_utils import auto_detect_segmenter

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get project
    if project is None:
        from ef import Project
        project = Project.create('_cli_temp', auto_register_segmenters=True)

    # Auto-detect
    print(f"Auto-detecting best segmenter for task: {task}")
    segmenter_name = auto_detect_segmenter(text, task=task)
    print(f"✓ Selected: {segmenter_name}")

    # Segment
    if segmenter_name in project.segmenters:
        seg_func = project.segmenters[segmenter_name]
        segments = seg_func(text)

        # Save output
        if output_file is None:
            output_file = str(input_path.with_suffix('.segmented.json'))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        print(f"✓ Segmented {input_file} -> {output_file}")
        print(f"  Segments: {len(segments)}")
    else:
        print(f"Error: Detected segmenter '{segmenter_name}' not available", file=sys.stderr)
        sys.exit(1)


def batch_segment_cli(
    input_dir: str,
    output_dir: str = None,
    segmenter: str = 'sentences',
    pattern: str = '*.txt',
    project = None
) -> None:
    """
    CLI command to batch segment files in a directory.

    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        segmenter: Name of segmenter to use
        pattern: Glob pattern for input files
        project: EF project instance
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: Directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Get project
    if project is None:
        from ef import Project
        project = Project.create('_cli_temp', auto_register_segmenters=True)

    if segmenter not in project.segmenters:
        print(f"Error: Segmenter '{segmenter}' not found", file=sys.stderr)
        sys.exit(1)

    # Find files
    files = list(input_path.glob(pattern))
    if not files:
        print(f"No files found matching pattern: {pattern}", file=sys.stderr)
        sys.exit(1)

    # Setup output directory
    if output_dir is None:
        output_dir = input_dir + '_segmented'
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\nBatch segmenting {len(files)} files:")
    print("=" * 60)

    seg_func = project.segmenters[segmenter]

    for i, file in enumerate(files, 1):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()

            segments = seg_func(text)

            output_file = output_path / f"{file.stem}.segmented.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)

            print(f"  [{i}/{len(files)}] {file.name} -> {len(segments)} segments")

        except Exception as e:
            print(f"  [{i}/{len(files)}] {file.name} - Error: {e}", file=sys.stderr)

    print(f"\n✓ Batch complete. Output in: {output_dir}")


def main_cli():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='EF Segmentation CLI Tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # segment command
    segment_parser = subparsers.add_parser('segment', help='Segment a file')
    segment_parser.add_argument('input', help='Input file')
    segment_parser.add_argument('-o', '--output', help='Output file')
    segment_parser.add_argument('-s', '--segmenter', default='sentences', help='Segmenter to use')
    segment_parser.add_argument('-f', '--format', default='json', choices=['json', 'jsonl', 'text'], help='Output format')

    # list command
    list_parser = subparsers.add_parser('list', help='List available segmenters')
    list_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed info')

    # benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark segmenters')
    bench_parser.add_argument('input', help='Input file')
    bench_parser.add_argument('-s', '--segmenters', nargs='+', help='Segmenters to benchmark')

    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare segmenters')
    compare_parser.add_argument('input', help='Input file')
    compare_parser.add_argument('-s', '--segmenters', nargs='+', required=True, help='Segmenters to compare')
    compare_parser.add_argument('-o', '--output', help='Output HTML file')

    # auto command
    auto_parser = subparsers.add_parser('auto', help='Auto-select segmenter')
    auto_parser.add_argument('input', help='Input file')
    auto_parser.add_argument('-o', '--output', help='Output file')
    auto_parser.add_argument('-t', '--task', default='general', help='Task type')

    # batch command
    batch_parser = subparsers.add_parser('batch', help='Batch segment files')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('-o', '--output-dir', help='Output directory')
    batch_parser.add_argument('-s', '--segmenter', default='sentences', help='Segmenter to use')
    batch_parser.add_argument('-p', '--pattern', default='*.txt', help='File pattern')

    args = parser.parse_args()

    if args.command == 'segment':
        segment_file_cli(args.input, args.output, args.segmenter, args.format)
    elif args.command == 'list':
        list_segmenters_cli(verbose=args.verbose)
    elif args.command == 'benchmark':
        benchmark_segmenters_cli(args.input, args.segmenters)
    elif args.command == 'compare':
        compare_segmenters_cli(args.input, args.segmenters, args.output)
    elif args.command == 'auto':
        auto_segment_cli(args.input, args.output, args.task)
    elif args.command == 'batch':
        batch_segment_cli(args.input_dir, args.output_dir, args.segmenter, args.pattern)
    else:
        parser.print_help()


if __name__ == '__main__':
    main_cli()
