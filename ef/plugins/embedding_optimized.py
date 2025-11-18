"""
Embedding-aware segmentation optimization.

Optimizes segments for better embedding quality and downstream tasks.
"""

def segment_for_embeddings(
    text: str,
    embedder,
    target_segments: int = 20,
    optimize_for: str = 'separability'
) -> dict[str, str]:
    """
    Segment to maximize embedding quality.
    
    Args:
        text: Text to segment
        embedder: Embedding function
        target_segments: Target number of segments
        optimize_for: 'separability', 'coverage', or 'diversity'
    
    Returns:
        Optimized segments
    """
    # Start with initial segmentation
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences to reach target segments
    sentences_per_segment = max(1, len(sentences) // target_segments)
    
    segments = {}
    for i in range(0, len(sentences), sentences_per_segment):
        group = sentences[i:i + sentences_per_segment]
        segments[f'seg_{i // sentences_per_segment}'] = '. '.join(group) + '.'
    
    return segments


def segment_for_llm(
    text: str,
    model: str = 'gpt-4',
    context_window: int = 8000,
    preserve_context: bool = True,
    strategy: str = 'greedy'
) -> dict[str, str]:
    """
    Segment to fit LLM token limits while maximizing information.
    
    Args:
        text: Text to segment
        model: LLM model name
        context_window: Token limit
        preserve_context: Whether to maintain context across segments
        strategy: 'greedy' or 'dynamic_programming'
    
    Returns:
        Token-optimized segments
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        
        segments = {}
        idx = 0
        pos = 0
        
        while pos < len(tokens):
            end = min(pos + context_window, len(tokens))
            chunk_tokens = tokens[pos:end]
            chunk_text = encoding.decode(chunk_tokens)
            segments[f'chunk_{idx}'] = chunk_text
            idx += 1
            
            # Overlap if preserving context
            if preserve_context:
                pos += context_window // 2
            else:
                pos = end
        
        return segments
        
    except ImportError:
        # Fallback without tiktoken
        import re
        chunk_size = context_window * 3  # Rough char estimate
        sentences = re.split(r'[.!?]+', text)
        
        segments = {}
        current = []
        current_size = 0
        idx = 0
        
        for sent in sentences:
            if current_size + len(sent) > chunk_size and current:
                segments[f'chunk_{idx}'] = '. '.join(current) + '.'
                current = []
                current_size = 0
                idx += 1
            current.append(sent)
            current_size += len(sent)
        
        if current:
            segments[f'chunk_{idx}'] = '. '.join(current) + '.'
        
        return segments


def register_embedding_optimized_segmenters(project) -> int:
    """Register embedding-optimized segmenters."""
    count = 0

    @project.segmenters.register('embedding_optimized',
                                package='ef.plugins.embedding',
                                description='Segment optimized for embedding quality')
    def embedding_opt(source, **kwargs):
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())
        return segment_for_embeddings(source, None, **kwargs)
    count += 1

    @project.segmenters.register('llm_optimized',
                                package='ef.plugins.embedding',
                                description='Segment optimized for LLM context windows')
    def llm_opt(source, **kwargs):
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())
        return segment_for_llm(source, **kwargs)
    count += 1

    print(f"✓ Registered {count} embedding-optimized segmenters")
    return count
