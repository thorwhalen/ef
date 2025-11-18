"""
Streaming and incremental segmentation for large files.

This module provides tools to process files larger than memory
by streaming and incrementally segmenting content.
"""

from typing import Any, Callable, Iterator, Optional
import io
import warnings


class StreamingSegmenter:
    """
    Stream segments from a large file without loading entirely into memory.

    Example:
        >>> with StreamingSegmenter(
        ...     project,
        ...     segmenter='sliding_window',
        ...     file_path='huge_file.txt',
        ...     buffer_size=10_000,
        ...     window_size=1000,
        ...     stride=500
        ... ) as streamer:
        ...     for key, segment in streamer:
        ...         process(segment)
    """

    def __init__(
        self,
        project,
        segmenter: str,
        file_path: Optional[str] = None,
        file_obj: Optional[io.IOBase] = None,
        buffer_size: int = 10_000,
        **segmenter_params
    ):
        """
        Initialize streaming segmenter.

        Args:
            project: Project instance
            segmenter: Name of segmenter to use
            file_path: Path to file (mutually exclusive with file_obj)
            file_obj: File object (mutually exclusive with file_path)
            buffer_size: Size of buffer to read at a time
            **segmenter_params: Parameters to pass to segmenter
        """
        self.project = project
        self.segmenter_name = segmenter
        self.buffer_size = buffer_size
        self.segmenter_params = segmenter_params

        if file_path and file_obj:
            raise ValueError("Cannot specify both file_path and file_obj")

        if file_path:
            self.file_obj = open(file_path, 'r', encoding='utf-8')
            self.should_close = True
        elif file_obj:
            self.file_obj = file_obj
            self.should_close = False
        else:
            raise ValueError("Must specify either file_path or file_obj")

        self.segment_counter = 0
        self.overlap_buffer = ""

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.should_close and self.file_obj:
            self.file_obj.close()

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Iterate over segments."""
        if self.segmenter_name not in self.project.segmenters:
            raise ValueError(f"Segmenter '{self.segmenter_name}' not found")

        segmenter = self.project.segmenters[self.segmenter_name]

        while True:
            # Read buffer
            chunk = self.file_obj.read(self.buffer_size)

            if not chunk:
                # End of file
                if self.overlap_buffer:
                    # Yield final overlap
                    yield (f'segment_{self.segment_counter}', self.overlap_buffer)
                break

            # Combine with overlap from previous chunk
            text = self.overlap_buffer + chunk

            # Segment the text
            try:
                if self.segmenter_params:
                    from functools import partial
                    seg_func = partial(segmenter, **self.segmenter_params)
                    segments = seg_func(text)
                else:
                    segments = segmenter(text)

                # Yield all but last segment
                segment_items = list(segments.items())

                if len(segment_items) > 1:
                    # Yield all but last
                    for key, segment_text in segment_items[:-1]:
                        yield (f'segment_{self.segment_counter}', segment_text)
                        self.segment_counter += 1

                    # Keep last segment as overlap
                    self.overlap_buffer = segment_items[-1][1]
                else:
                    # Only one segment, keep as overlap
                    if segment_items:
                        self.overlap_buffer = segment_items[0][1]

            except Exception as e:
                warnings.warn(f"Segmentation error: {e}")
                # Yield the raw chunk
                yield (f'segment_{self.segment_counter}', text)
                self.segment_counter += 1
                self.overlap_buffer = ""


def stream_segment_file(
    project,
    file_path: str,
    segmenter: str,
    buffer_size: int = 10_000,
    **segmenter_params
) -> Iterator[tuple[str, str]]:
    """
    Stream segments from a file.

    Args:
        project: Project instance
        file_path: Path to file
        segmenter: Segmenter name
        buffer_size: Buffer size in characters
        **segmenter_params: Parameters for segmenter

    Yields:
        (key, segment) tuples

    Example:
        >>> for key, segment in stream_segment_file(
        ...     project,
        ...     'large_file.txt',
        ...     'sentences'
        ... ):
        ...     embed_and_store(key, segment)
    """
    with StreamingSegmenter(
        project,
        segmenter,
        file_path=file_path,
        buffer_size=buffer_size,
        **segmenter_params
    ) as streamer:
        yield from streamer


class IncrementalSegmenter:
    """
    Incrementally build segments as text arrives.

    Useful for streaming data, live transcription, etc.

    Example:
        >>> incremental = IncrementalSegmenter(project, 'sentences')
        >>>
        >>> # Add text incrementally
        >>> incremental.add_text("Hello world. ")
        >>> incremental.add_text("This is a test. ")
        >>> incremental.add_text("More text here.")
        >>>
        >>> # Get completed segments
        >>> segments = incremental.get_completed_segments()
        >>>
        >>> # Finalize to get all remaining segments
        >>> final = incremental.finalize()
    """

    def __init__(
        self,
        project,
        segmenter: str,
        min_buffer_size: int = 1000,
        **segmenter_params
    ):
        """
        Initialize incremental segmenter.

        Args:
            project: Project instance
            segmenter: Name of segmenter
            min_buffer_size: Minimum buffer size before segmenting
            **segmenter_params: Parameters for segmenter
        """
        self.project = project
        self.segmenter_name = segmenter
        self.min_buffer_size = min_buffer_size
        self.segmenter_params = segmenter_params

        if segmenter not in project.segmenters:
            raise ValueError(f"Segmenter '{segmenter}' not found")

        self.buffer = ""
        self.completed_segments = {}
        self.segment_counter = 0

    def add_text(self, text: str) -> list[tuple[str, str]]:
        """
        Add text incrementally.

        Args:
            text: Text to add

        Returns:
            List of newly completed segments as (key, text) tuples
        """
        self.buffer += text

        new_segments = []

        # Only segment if buffer is large enough
        if len(self.buffer) >= self.min_buffer_size:
            segmenter = self.project.segmenters[self.segmenter_name]

            try:
                if self.segmenter_params:
                    from functools import partial
                    seg_func = partial(segmenter, **self.segmenter_params)
                    segments = seg_func(self.buffer)
                else:
                    segments = segmenter(self.buffer)

                segment_items = list(segments.items())

                if len(segment_items) > 1:
                    # Complete all but last segment
                    for _, segment_text in segment_items[:-1]:
                        key = f'segment_{self.segment_counter}'
                        self.completed_segments[key] = segment_text
                        new_segments.append((key, segment_text))
                        self.segment_counter += 1

                    # Keep last segment in buffer
                    self.buffer = segment_items[-1][1]

            except Exception as e:
                warnings.warn(f"Segmentation error: {e}")

        return new_segments

    def get_completed_segments(self) -> dict[str, str]:
        """Get all completed segments so far."""
        return dict(self.completed_segments)

    def finalize(self) -> dict[str, str]:
        """
        Finalize and get all segments including buffer.

        Returns:
            All segments
        """
        if self.buffer:
            # Add remaining buffer as final segment
            key = f'segment_{self.segment_counter}'
            self.completed_segments[key] = self.buffer
            self.buffer = ""

        return dict(self.completed_segments)

    def reset(self) -> None:
        """Reset the incremental segmenter."""
        self.buffer = ""
        self.completed_segments = {}
        self.segment_counter = 0


def process_large_file(
    project,
    file_path: str,
    segmenter: str,
    processor: Callable[[str, str], None],
    buffer_size: int = 10_000,
    show_progress: bool = True,
    **segmenter_params
) -> int:
    """
    Process a large file with a streaming segmenter.

    Args:
        project: Project instance
        file_path: Path to file
        segmenter: Segmenter name
        processor: Function to process each segment (key, text) -> None
        buffer_size: Buffer size
        show_progress: Whether to show progress
        **segmenter_params: Segmenter parameters

    Returns:
        Number of segments processed

    Example:
        >>> def process_segment(key, text):
        ...     embedding = embed(text)
        ...     store(key, embedding)
        >>>
        >>> count = process_large_file(
        ...     project,
        ...     'huge_file.txt',
        ...     'sliding_window',
        ...     processor=process_segment,
        ...     window_size=1000,
        ...     stride=500
        ... )
        >>> print(f"Processed {count} segments")
    """
    import os

    if show_progress:
        file_size = os.path.getsize(file_path)
        print(f"Processing file: {file_path} ({file_size / 1024 / 1024:.1f} MB)")

    count = 0

    try:
        for key, segment in stream_segment_file(
            project,
            file_path,
            segmenter,
            buffer_size=buffer_size,
            **segmenter_params
        ):
            processor(key, segment)
            count += 1

            if show_progress and count % 100 == 0:
                print(f"  Processed {count} segments...")

    except Exception as e:
        warnings.warn(f"Error during processing: {e}")
        raise

    if show_progress:
        print(f"Completed: {count} segments processed")

    return count


def memory_efficient_batch_segment(
    project,
    file_paths: list[str],
    segmenter: str,
    output_dir: str,
    buffer_size: int = 10_000,
    **segmenter_params
) -> dict[str, int]:
    """
    Memory-efficiently segment multiple large files.

    Args:
        project: Project instance
        file_paths: List of file paths
        segmenter: Segmenter name
        output_dir: Directory to write segment files
        buffer_size: Buffer size
        **segmenter_params: Segmenter parameters

    Returns:
        Dict mapping file paths to segment counts

    Example:
        >>> results = memory_efficient_batch_segment(
        ...     project,
        ...     ['file1.txt', 'file2.txt'],
        ...     segmenter='sentences',
        ...     output_dir='segments/'
        ... )
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for file_path in file_paths:
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f'{base_name}.segments.json')

        segments = {}
        count = 0

        for key, segment in stream_segment_file(
            project,
            file_path,
            segmenter,
            buffer_size=buffer_size,
            **segmenter_params
        ):
            segments[key] = segment
            count += 1

        # Write segments to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2)

        results[file_path] = count

    return results
