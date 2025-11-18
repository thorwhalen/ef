"""
Multilingual segmentation support.

Provides language-aware segmentation for multiple languages.
"""

def detect_language(text: str) -> str:
    """Detect text language."""
    try:
        from langdetect import detect
        return detect(text)
    except:
        return 'en'  # Default to English


def register_multilingual_segmenters(project) -> int:
    """Register multilingual segmenters."""
    count = 0

    @project.segmenters.register('multilingual_auto', 
                                package='ef.plugins.multilingual',
                                description='Auto-detect language and segment appropriately')
    def multilingual_auto(source):
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())
        
        lang = detect_language(source)
        
        # Use language-specific logic
        if lang == 'zh':  # Chinese
            return chinese_segment(source)
        elif lang == 'ja':  # Japanese
            return japanese_segment(source)
        elif lang == 'ar':  # Arabic
            return arabic_segment(source)
        else:
            # Default to sentence-based
            import re
            sents = re.split(r'[.!?]+', source)
            return {f'sent_{i}': s.strip() for i, s in enumerate(sents) if s.strip()}
    count += 1

    return count


def chinese_segment(text: str) -> dict[str, str]:
    """Segment Chinese text."""
    try:
        import jieba
        words = jieba.cut(text)
        # Group into phrases
        phrases = []
        current = []
        for word in words:
            current.append(word)
            if len(''.join(current)) > 100:
                phrases.append(''.join(current))
                current = []
        if current:
            phrases.append(''.join(current))
        return {f'phrase_{i}': p for i, p in enumerate(phrases)}
    except:
        # Fallback: split by punctuation
        import re
        parts = re.split(r'[。！？]', text)
        return {f'sent_{i}': p.strip() for i, p in enumerate(parts) if p.strip()}


def japanese_segment(text: str) -> dict[str, str]:
    """Segment Japanese text."""
    try:
        import MeCab
        mecab = MeCab.Tagger()
        # Segment by sentences
        sents = text.split('。')
        return {f'sent_{i}': s.strip() for i, s in enumerate(sents) if s.strip()}
    except:
        import re
        sents = re.split(r'[。！？]', text)
        return {f'sent_{i}': s.strip() for i, s in enumerate(sents) if s.strip()}


def arabic_segment(text: str) -> dict[str, str]:
    """Segment Arabic text."""
    import re
    # Arabic sentence endings
    sents = re.split(r'[.؟!]', text)
    return {f'sent_{i}': s.strip() for i, s in enumerate(sents) if s.strip()}
