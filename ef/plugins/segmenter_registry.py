"""
Advanced segmenter registration for the ef framework.

This module provides conditional registration of segmenters from various
third-party packages, including:

- LangChain text splitters (langchain-text-splitters)
- Code segmenters (tree-sitter, ast)
- NLP-based segmenters (spaCy, NLTK, segtok)
- Token-based segmenters (tiktoken)

All segmenters are registered conditionally based on package availability.
"""

from typing import Any, Callable, Optional
import warnings


def _wrap_langchain_splitter(splitter_class, name: str, **kwargs) -> Callable:
    """
    Wrap a LangChain text splitter to match ef's segmenter interface.

    Args:
        splitter_class: LangChain TextSplitter class
        name: Name for the segmenter
        **kwargs: Arguments to pass to the splitter constructor

    Returns:
        Function with signature (source: Any) -> dict[str, str]
    """
    def segmenter(source: Any) -> dict[str, str]:
        """Segment text using LangChain splitter."""
        if isinstance(source, dict):
            # If source is already segmented, concatenate it
            source = '\n\n'.join(source.values())

        splitter = splitter_class(**kwargs)
        chunks = splitter.split_text(source)

        return {f'{name}_{i}': chunk for i, chunk in enumerate(chunks)}

    return segmenter


def register_langchain_segmenters(project) -> int:
    """
    Register LangChain text splitters as segmenters.

    LangChain provides many sophisticated text splitting strategies including:
    - Character-based splitting with overlap
    - Recursive character splitting (tries multiple separators)
    - Language-specific code splitters (Python, JS, etc.)
    - Markdown and HTML structure-aware splitters
    - Semantic splitting based on embeddings

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    try:
        from langchain_text_splitters import (
            CharacterTextSplitter,
            RecursiveCharacterTextSplitter,
            MarkdownTextSplitter,
            HTMLHeaderTextSplitter,
            PythonCodeTextSplitter,
            Language,
            RecursiveCharacterTextSplitter as LanguageTextSplitter,
        )

        count = 0

        # Character-based splitters with different chunk sizes
        for chunk_size in [500, 1000, 2000]:
            name = f'langchain_char_{chunk_size}'
            project.segmenters.register(
                name,
                package='langchain-text-splitters',
                description=f'Character-based text splitter with {chunk_size} char chunks and 200 char overlap',
                chunk_size=chunk_size,
                chunk_overlap=200
            )(_wrap_langchain_splitter(
                CharacterTextSplitter,
                name,
                chunk_size=chunk_size,
                chunk_overlap=200,
                separator='\n\n'
            ))
            count += 1

        # Recursive character splitter (tries multiple separators)
        for chunk_size in [500, 1000, 2000]:
            name = f'langchain_recursive_{chunk_size}'
            project.segmenters.register(
                name,
                package='langchain-text-splitters',
                description=f'Recursive character splitter with {chunk_size} char chunks, tries multiple separators',
                chunk_size=chunk_size,
                chunk_overlap=200
            )(_wrap_langchain_splitter(
                RecursiveCharacterTextSplitter,
                name,
                chunk_size=chunk_size,
                chunk_overlap=200
            ))
            count += 1

        # Markdown-aware splitter
        project.segmenters.register(
            'langchain_markdown',
            package='langchain-text-splitters',
            description='Markdown structure-aware text splitter (respects headers, code blocks, etc.)',
            chunk_size=1000,
            chunk_overlap=100
        )(_wrap_langchain_splitter(
            MarkdownTextSplitter,
            'langchain_markdown',
            chunk_size=1000,
            chunk_overlap=100
        ))
        count += 1

        # Python code splitter
        project.segmenters.register(
            'langchain_python',
            package='langchain-text-splitters',
            description='Python code splitter (respects functions, classes, etc.)',
            chunk_size=1000,
            chunk_overlap=100
        )(_wrap_langchain_splitter(
            PythonCodeTextSplitter,
            'langchain_python',
            chunk_size=1000,
            chunk_overlap=100
        ))
        count += 1

        # Language-specific code splitters
        code_languages = [
            ('js', Language.JS, 'JavaScript'),
            ('ts', Language.TS, 'TypeScript'),
            ('java', Language.JAVA, 'Java'),
            ('cpp', Language.CPP, 'C++'),
            ('go', Language.GO, 'Go'),
            ('rust', Language.RUST, 'Rust'),
            ('html', Language.HTML, 'HTML'),
        ]

        for lang_code, lang_enum, lang_name in code_languages:
            try:
                name = f'langchain_{lang_code}'

                def make_lang_segmenter(language_enum, segmenter_name):
                    def segmenter(source: Any) -> dict[str, str]:
                        if isinstance(source, dict):
                            source = '\n\n'.join(source.values())

                        splitter = LanguageTextSplitter.from_language(
                            language=language_enum,
                            chunk_size=1000,
                            chunk_overlap=100
                        )
                        chunks = splitter.split_text(source)
                        return {f'{segmenter_name}_{i}': chunk for i, chunk in enumerate(chunks)}
                    return segmenter

                project.segmenters.register(
                    name,
                    package='langchain-text-splitters',
                    description=f'{lang_name} code splitter (respects syntax structure)',
                    chunk_size=1000,
                    chunk_overlap=100
                )(make_lang_segmenter(lang_enum, name))
                count += 1
            except Exception:
                # Some languages might not be available in all versions
                pass

        print(f"✓ Registered {count} LangChain segmenters")
        return count

    except ImportError as e:
        print(f"○ LangChain text splitters not available: {e}")
        return 0


def register_tree_sitter_segmenters(project) -> int:
    """
    Register tree-sitter based code segmenters.

    Tree-sitter provides industrial-grade parsing for many programming languages,
    allowing segmentation by:
    - Function definitions
    - Class definitions
    - Top-level statements
    - Import blocks

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    try:
        from tree_sitter_languages import get_parser, get_language

        count = 0

        # Python function-level segmenter
        def tree_sitter_python_functions(source: Any) -> dict[str, str]:
            """Segment Python code by function and class definitions."""
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())

            parser = get_parser('python')
            tree = parser.parse(bytes(source, 'utf8'))

            segments = {}

            def extract_definitions(node, depth=0):
                """Recursively extract function and class definitions."""
                if node.type in ('function_definition', 'class_definition'):
                    start_byte = node.start_byte
                    end_byte = node.end_byte
                    code = source[start_byte:end_byte]

                    # Get the name
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        name = source[name_node.start_byte:name_node.end_byte]
                        key = f"{node.type}_{name}_{len(segments)}"
                    else:
                        key = f"{node.type}_{len(segments)}"

                    segments[key] = code

                # Recurse into children
                for child in node.children:
                    extract_definitions(child, depth + 1)

            extract_definitions(tree.root_node)

            # If no definitions found, return whole source
            if not segments:
                return {'main': source}

            return segments

        project.segmenters.register(
            'tree_sitter_python',
            package='tree-sitter-languages',
            description='Segment Python code by function and class definitions using tree-sitter',
            granularity='function'
        )(tree_sitter_python_functions)
        count += 1

        # Generic code segmenter for other languages
        def make_tree_sitter_segmenter(language: str):
            def segmenter(source: Any) -> dict[str, str]:
                """Segment code by top-level definitions."""
                if isinstance(source, dict):
                    source = '\n\n'.join(source.values())

                try:
                    parser = get_parser(language)
                    tree = parser.parse(bytes(source, 'utf8'))

                    segments = {}

                    # Extract top-level children
                    for i, child in enumerate(tree.root_node.children):
                        if child.type != 'comment':  # Skip standalone comments
                            start_byte = child.start_byte
                            end_byte = child.end_byte
                            code = source[start_byte:end_byte]
                            segments[f'{child.type}_{i}'] = code

                    return segments if segments else {'main': source}

                except Exception as e:
                    warnings.warn(f"Tree-sitter parsing failed for {language}: {e}")
                    return {'main': source}

            return segmenter

        # Register for common languages
        for lang in ['javascript', 'typescript', 'java', 'cpp', 'rust', 'go']:
            try:
                # Test if language is available
                get_language(lang)

                project.segmenters.register(
                    f'tree_sitter_{lang}',
                    package='tree-sitter-languages',
                    description=f'Segment {lang} code by top-level definitions using tree-sitter',
                    language=lang
                )(make_tree_sitter_segmenter(lang))
                count += 1
            except Exception:
                # Language not available, skip
                pass

        print(f"✓ Registered {count} tree-sitter segmenters")
        return count

    except ImportError as e:
        print(f"○ tree-sitter not available: {e}")
        return 0


def register_ast_segmenters(project) -> int:
    """
    Register Python AST-based code segmenters.

    Uses Python's built-in ast module to segment Python code by:
    - Functions
    - Classes
    - Top-level statements

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    import ast

    count = 0

    def ast_python_functions(source: Any) -> dict[str, str]:
        """Segment Python code by function and class definitions using AST."""
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        try:
            tree = ast.parse(source)
            segments = {}

            lines = source.split('\n')

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    # Extract the source code for this definition
                    start_line = node.lineno - 1  # AST uses 1-based indexing
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1

                    code_lines = lines[start_line:end_line]
                    code = '\n'.join(code_lines)

                    # Create a key with the type and name
                    node_type = type(node).__name__
                    key = f"{node_type}_{node.name}_{len(segments)}"

                    segments[key] = code

            return segments if segments else {'main': source}

        except SyntaxError as e:
            warnings.warn(f"AST parsing failed: {e}")
            return {'main': source}

    project.segmenters.register(
        'ast_python',
        package='ast (built-in)',
        description='Segment Python code by function and class definitions using AST',
        granularity='function'
    )(ast_python_functions)
    count += 1

    def ast_python_top_level(source: Any) -> dict[str, str]:
        """Segment Python code by top-level statements using AST."""
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        try:
            tree = ast.parse(source)
            segments = {}

            lines = source.split('\n')

            for i, node in enumerate(tree.body):
                # Extract source code for this top-level statement
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1

                code_lines = lines[start_line:end_line]
                code = '\n'.join(code_lines)

                node_type = type(node).__name__
                key = f"{node_type}_{i}"

                segments[key] = code

            return segments if segments else {'main': source}

        except SyntaxError as e:
            warnings.warn(f"AST parsing failed: {e}")
            return {'main': source}

    project.segmenters.register(
        'ast_python_statements',
        package='ast (built-in)',
        description='Segment Python code by top-level statements using AST',
        granularity='statement'
    )(ast_python_top_level)
    count += 1

    print(f"✓ Registered {count} AST-based segmenters")
    return count


def register_spacy_segmenters(project) -> int:
    """
    Register spaCy-based text segmenters.

    spaCy provides industrial-strength NLP including:
    - Sentence segmentation
    - Token segmentation
    - Named entity recognition

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    try:
        import spacy

        count = 0

        # Try to load a small English model
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            # Model not installed, try to use blank model
            try:
                nlp = spacy.blank('en')
                nlp.add_pipe('sentencizer')
            except Exception:
                print("○ spaCy models not available")
                return 0

        def spacy_sentences(source: Any) -> dict[str, str]:
            """Segment text into sentences using spaCy."""
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())

            doc = nlp(source)
            return {f'sent_{i}': sent.text.strip()
                    for i, sent in enumerate(doc.sents)
                    if sent.text.strip()}

        project.segmenters.register(
            'spacy_sentences',
            package='spacy',
            description='Segment text into sentences using spaCy NLP',
            granularity='sentence'
        )(spacy_sentences)
        count += 1

        def spacy_tokens(source: Any) -> dict[str, str]:
            """Segment text into tokens using spaCy."""
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())

            doc = nlp(source)
            return {f'token_{i}': token.text
                    for i, token in enumerate(doc)
                    if not token.is_space}

        project.segmenters.register(
            'spacy_tokens',
            package='spacy',
            description='Segment text into tokens using spaCy NLP',
            granularity='token'
        )(spacy_tokens)
        count += 1

        print(f"✓ Registered {count} spaCy segmenters")
        return count

    except ImportError as e:
        print(f"○ spaCy not available: {e}")
        return 0


def register_nltk_segmenters(project) -> int:
    """
    Register NLTK-based text segmenters.

    NLTK (Natural Language Toolkit) provides classic NLP tools including:
    - Sentence tokenization
    - Word tokenization
    - Punkt sentence segmentation

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    try:
        import nltk

        count = 0

        # Try to ensure punkt tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                print("○ NLTK punkt tokenizer not available and couldn't download")
                return 0

        def nltk_sentences(source: Any) -> dict[str, str]:
            """Segment text into sentences using NLTK."""
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())

            sentences = nltk.sent_tokenize(source)
            return {f'sent_{i}': sent.strip()
                    for i, sent in enumerate(sentences)
                    if sent.strip()}

        project.segmenters.register(
            'nltk_sentences',
            package='nltk',
            description='Segment text into sentences using NLTK punkt tokenizer',
            granularity='sentence'
        )(nltk_sentences)
        count += 1

        def nltk_words(source: Any) -> dict[str, str]:
            """Segment text into words using NLTK."""
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())

            words = nltk.word_tokenize(source)
            return {f'word_{i}': word for i, word in enumerate(words)}

        project.segmenters.register(
            'nltk_words',
            package='nltk',
            description='Segment text into words using NLTK word tokenizer',
            granularity='word'
        )(nltk_words)
        count += 1

        print(f"✓ Registered {count} NLTK segmenters")
        return count

    except ImportError as e:
        print(f"○ NLTK not available: {e}")
        return 0


def register_tiktoken_segmenters(project) -> int:
    """
    Register tiktoken-based token segmenters.

    tiktoken is OpenAI's tokenizer, useful for:
    - Token-count-based segmentation (for LLM context windows)
    - GPT model-specific tokenization
    - Efficient token counting

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    try:
        import tiktoken

        count = 0

        # Create segmenters for common models
        models = [
            ('gpt-4', 'cl100k_base', 8000),
            ('gpt-3.5', 'cl100k_base', 4000),
        ]

        for model_name, encoding_name, max_tokens in models:
            def make_tiktoken_segmenter(enc_name, max_toks, seg_name):
                def segmenter(source: Any) -> dict[str, str]:
                    """Segment text by token count."""
                    if isinstance(source, dict):
                        source = '\n\n'.join(source.values())

                    encoding = tiktoken.get_encoding(enc_name)
                    tokens = encoding.encode(source)

                    segments = {}
                    for i in range(0, len(tokens), max_toks):
                        chunk_tokens = tokens[i:i + max_toks]
                        chunk_text = encoding.decode(chunk_tokens)
                        segments[f'{seg_name}_{i // max_toks}'] = chunk_text

                    return segments if segments else {'main': source}

                return segmenter

            seg_name = f'tiktoken_{model_name.replace(".", "_")}'

            project.segmenters.register(
                seg_name,
                package='tiktoken',
                description=f'Segment text by {max_tokens} tokens for {model_name}',
                encoding=encoding_name,
                max_tokens=max_tokens
            )(make_tiktoken_segmenter(encoding_name, max_tokens, seg_name))
            count += 1

        print(f"✓ Registered {count} tiktoken segmenters")
        return count

    except ImportError as e:
        print(f"○ tiktoken not available: {e}")
        return 0


def register_segtok_segmenters(project) -> int:
    """
    Register segtok-based segmenters.

    segtok provides simple, fast sentence and word segmentation.

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    try:
        from segtok import segmenter, tokenizer

        count = 0

        def segtok_sentences(source: Any) -> dict[str, str]:
            """Segment text into sentences using segtok."""
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())

            sentences = segmenter.split_single(source)
            return {f'sent_{i}': sent.strip()
                    for i, sent in enumerate(sentences)
                    if sent.strip()}

        project.segmenters.register(
            'segtok_sentences',
            package='segtok',
            description='Segment text into sentences using segtok',
            granularity='sentence'
        )(segtok_sentences)
        count += 1

        def segtok_words(source: Any) -> dict[str, str]:
            """Segment text into words using segtok."""
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())

            words = list(tokenizer.word_tokenizer(source))
            return {f'word_{i}': word for i, word in enumerate(words)}

        project.segmenters.register(
            'segtok_words',
            package='segtok',
            description='Segment text into words using segtok',
            granularity='word'
        )(segtok_words)
        count += 1

        print(f"✓ Registered {count} segtok segmenters")
        return count

    except ImportError as e:
        print(f"○ segtok not available: {e}")
        return 0


def register_all_segmenters(project, verbose: bool = True) -> dict[str, int]:
    """
    Register all available segmenters from all supported packages.

    This function attempts to register segmenters from:
    - LangChain text splitters
    - tree-sitter (code parsing)
    - AST (Python code)
    - spaCy (NLP)
    - NLTK (NLP)
    - tiktoken (OpenAI tokenizer)
    - segtok (simple segmentation)

    All registrations are conditional - packages that aren't installed
    will be silently skipped.

    Args:
        project: Project instance with segmenters registry
        verbose: Whether to print registration status (default: True)

    Returns:
        Dictionary mapping package names to number of segmenters registered
    """
    if verbose:
        print("\n" + "="*60)
        print("Registering external segmenters...")
        print("="*60 + "\n")

    results = {}

    # Register from each package
    results['langchain'] = register_langchain_segmenters(project)
    results['tree_sitter'] = register_tree_sitter_segmenters(project)
    results['ast'] = register_ast_segmenters(project)
    results['spacy'] = register_spacy_segmenters(project)
    results['nltk'] = register_nltk_segmenters(project)
    results['tiktoken'] = register_tiktoken_segmenters(project)
    results['segtok'] = register_segtok_segmenters(project)

    if verbose:
        total = sum(results.values())
        print(f"\n{'='*60}")
        print(f"Total: {total} segmenters registered from {sum(1 for v in results.values() if v > 0)} packages")
        print("="*60 + "\n")

    return results
