"""
Domain-specific segmentation strategies.

Specialized segmenters for legal, scientific, code, medical, and other domains.
"""

import re


def register_legal_segmenters(project) -> int:
    """Register legal document segmenters."""
    count = 0

    @project.segmenters.register('legal_clauses',
                                package='ef.plugins.domain',
                                description='Segment legal documents by clauses')
    def legal_clauses(source):
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())
        
        # Detect numbered clauses
        clauses = re.split(r'\n\s*\d+\.?\s+', source)
        return {f'clause_{i}': c.strip() for i, c in enumerate(clauses) if c.strip()}
    count += 1

    return count


def register_scientific_segmenters(project) -> int:
    """Register scientific paper segmenters."""
    count = 0

    @project.segmenters.register('paper_sections',
                                package='ef.plugins.domain',
                                description='Segment by paper sections (Abstract, Methods, etc.)')
    def paper_sections(source):
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())
        
        sections = {}
        current_section = None
        current_text = []
        
        for line in source.split('\n'):
            # Detect section headers
            if re.match(r'^(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References)', 
                       line, re.IGNORECASE):
                if current_section:
                    sections[current_section] = '\n'.join(current_text)
                current_section = line.strip().lower()
                current_text = []
            else:
                current_text.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_text)
        
        return sections if sections else {'main': source}
    count += 1

    return count


def register_code_segmenters(project) -> int:
    """Register code-specific segmenters."""
    count = 0

    @project.segmenters.register('by_complexity',
                                package='ef.plugins.domain',
                                description='Segment code by complexity')
    def by_complexity(source):
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())
        
        # Simple complexity: count control structures
        lines = source.split('\n')
        segments = {}
        current = []
        complexity = 0
        idx = 0
        
        for line in lines:
            current.append(line)
            # Count complexity indicators
            if any(kw in line for kw in ['if', 'for', 'while', 'try', 'except']):
                complexity += 1
            
            # Break at high complexity
            if complexity > 5:
                segments[f'block_{idx}'] = '\n'.join(current)
                current = []
                complexity = 0
                idx += 1
        
        if current:
            segments[f'block_{idx}'] = '\n'.join(current)
        
        return segments
    count += 1

    return count


def register_medical_segmenters(project) -> int:
    """Register medical text segmenters."""
    count = 0

    @project.segmenters.register('clinical_notes',
                                package='ef.plugins.domain',
                                description='Segment clinical notes by sections')
    def clinical_notes(source):
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())
        
        sections = {}
        for match in re.finditer(r'([A-Z\s]+):\s*([^\n]+(?:\n(?![A-Z\s]+:)[^\n]+)*)', source):
            section_name = match.group(1).strip().lower().replace(' ', '_')
            section_text = match.group(2).strip()
            sections[section_name] = section_text
        
        return sections if sections else {'main': source}
    count += 1

    return count
