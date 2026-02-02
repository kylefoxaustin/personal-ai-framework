#!/usr/bin/env python3
"""
Smart Chunking Module
Intelligent text chunking that respects document structure.
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    metadata: Dict
    

class SmartChunker:
    """
    Intelligent text chunking that:
    - Respects sentence boundaries
    - Respects paragraph boundaries  
    - Handles email structure
    - Preserves context
    """
    
    def __init__(
        self,
        target_size: int = 500,      # Target words per chunk
        max_size: int = 750,          # Maximum words per chunk
        min_size: int = 100,          # Minimum words per chunk
        overlap_sentences: int = 2    # Sentences to overlap
    ):
        self.target_size = target_size
        self.max_size = max_size
        self.min_size = min_size
        self.overlap_sentences = overlap_sentences
        
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n'
        )
        
        # Paragraph pattern
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # Email header pattern
        self.email_header_pattern = re.compile(
            r'^(From|To|Subject|Date|Sent|Cc|Bcc):\s*.+$',
            re.MULTILINE
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        Chunk text intelligently based on structure.
        """
        metadata = metadata or {}
        
        # Detect content type and route to appropriate chunker
        if self._is_email(text):
            return self._chunk_email(text, metadata)
        elif self._has_clear_sections(text):
            return self._chunk_by_sections(text, metadata)
        else:
            return self._chunk_by_paragraphs(text, metadata)
    
    def _is_email(self, text: str) -> bool:
        """Detect if text is an email."""
        header_matches = self.email_header_pattern.findall(text[:500])
        return len(header_matches) >= 2
    
    def _has_clear_sections(self, text: str) -> bool:
        """Detect if text has clear section headers."""
        # Look for markdown headers or numbered sections
        section_patterns = [
            r'^#{1,3}\s+.+$',           # Markdown headers
            r'^\d+\.\s+[A-Z].+$',        # Numbered sections
            r'^[A-Z][A-Z\s]{5,}$',       # ALL CAPS headers
        ]
        for pattern in section_patterns:
            if len(re.findall(pattern, text, re.MULTILINE)) >= 3:
                return True
        return False
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s', r'\1<DOT> ', text)
        text = re.sub(r'\b(Inc|Ltd|Corp|etc|vs|i\.e|e\.g)\.\s', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = self.sentence_pattern.split(text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _word_count(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _chunk_email(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk email preserving headers with body.
        """
        chunks = []
        
        # Extract headers
        header_match = re.match(
            r'^((?:(?:From|To|Subject|Date|Sent|Cc|Bcc):.*\n)+)',
            text,
            re.MULTILINE
        )
        
        if header_match:
            headers = header_match.group(1).strip()
            body = text[header_match.end():].strip()
            
            # Parse headers for metadata
            for line in headers.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    if key in ['from', 'to', 'subject', 'date']:
                        metadata[f'email_{key}'] = value.strip()[:100]
        else:
            headers = ""
            body = text
        
        # If body is small enough, keep as single chunk
        if self._word_count(body) <= self.max_size:
            chunk_text = f"{headers}\n\n{body}".strip() if headers else body
            chunks.append(Chunk(text=chunk_text, metadata={**metadata, 'chunk_type': 'email'}))
            return chunks
        
        # Otherwise, chunk the body but prepend headers to first chunk
        body_chunks = self._chunk_by_paragraphs(body, {**metadata, 'chunk_type': 'email_body'})
        
        # Add headers to first chunk
        if body_chunks and headers:
            first = body_chunks[0]
            first.text = f"{headers}\n\n{first.text}"
            first.metadata['has_headers'] = True
        
        return body_chunks
    
    def _chunk_by_sections(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk by document sections (headers).
        """
        chunks = []
        
        # Find section headers
        section_pattern = re.compile(
            r'^(#{1,3}\s+.+|\d+\.\s+[A-Z].+|[A-Z][A-Z\s]{5,})$',
            re.MULTILINE
        )
        
        sections = section_pattern.split(text)
        current_header = None
        current_content = []
        
        for part in sections:
            part = part.strip()
            if not part:
                continue
                
            if section_pattern.match(part):
                # Save previous section
                if current_content:
                    section_text = '\n\n'.join(current_content)
                    if current_header:
                        section_text = f"{current_header}\n\n{section_text}"
                    
                    # Sub-chunk if too large
                    if self._word_count(section_text) > self.max_size:
                        sub_chunks = self._chunk_by_paragraphs(
                            section_text, 
                            {**metadata, 'section': current_header, 'chunk_type': 'section'}
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(Chunk(
                            text=section_text,
                            metadata={**metadata, 'section': current_header, 'chunk_type': 'section'}
                        ))
                
                current_header = part
                current_content = []
            else:
                current_content.append(part)
        
        # Don't forget last section
        if current_content:
            section_text = '\n\n'.join(current_content)
            if current_header:
                section_text = f"{current_header}\n\n{section_text}"
            
            if self._word_count(section_text) > self.max_size:
                sub_chunks = self._chunk_by_paragraphs(
                    section_text,
                    {**metadata, 'section': current_header, 'chunk_type': 'section'}
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=section_text,
                    metadata={**metadata, 'section': current_header, 'chunk_type': 'section'}
                ))
        
        return chunks if chunks else self._chunk_by_paragraphs(text, metadata)
    
    def _chunk_by_paragraphs(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk by paragraphs, respecting sentence boundaries.
        """
        chunks = []
        
        # Split into paragraphs
        paragraphs = self.paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk_parts = []
        current_word_count = 0
        
        for para in paragraphs:
            para_words = self._word_count(para)
            
            # If single paragraph exceeds max, split by sentences
            if para_words > self.max_size:
                # Flush current chunk first
                if current_chunk_parts:
                    chunks.append(Chunk(
                        text='\n\n'.join(current_chunk_parts),
                        metadata={**metadata, 'chunk_type': 'paragraph'}
                    ))
                    current_chunk_parts = []
                    current_word_count = 0
                
                # Split large paragraph by sentences
                sentence_chunks = self._chunk_by_sentences(para, metadata)
                chunks.extend(sentence_chunks)
                continue
            
            # Would adding this paragraph exceed target?
            if current_word_count + para_words > self.target_size and current_chunk_parts:
                # Save current chunk
                chunks.append(Chunk(
                    text='\n\n'.join(current_chunk_parts),
                    metadata={**metadata, 'chunk_type': 'paragraph'}
                ))
                current_chunk_parts = []
                current_word_count = 0
            
            current_chunk_parts.append(para)
            current_word_count += para_words
        
        # Don't forget the last chunk
        if current_chunk_parts:
            chunks.append(Chunk(
                text='\n\n'.join(current_chunk_parts),
                metadata={**metadata, 'chunk_type': 'paragraph'}
            ))
        
        # Add overlap between chunks
        chunks = self._add_overlap(chunks)
        
        return chunks if chunks else [Chunk(text=text, metadata=metadata)]
    
    def _chunk_by_sentences(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk by sentences when paragraphs are too large.
        """
        chunks = []
        sentences = self._split_sentences(text)
        
        current_sentences = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = self._word_count(sentence)
            
            if current_word_count + sentence_words > self.target_size and current_sentences:
                chunks.append(Chunk(
                    text=' '.join(current_sentences),
                    metadata={**metadata, 'chunk_type': 'sentence'}
                ))
                
                # Keep overlap sentences
                overlap_start = max(0, len(current_sentences) - self.overlap_sentences)
                current_sentences = current_sentences[overlap_start:]
                current_word_count = sum(self._word_count(s) for s in current_sentences)
            
            current_sentences.append(sentence)
            current_word_count += sentence_words
        
        if current_sentences:
            chunks.append(Chunk(
                text=' '.join(current_sentences),
                metadata={**metadata, 'chunk_type': 'sentence'}
            ))
        
        return chunks
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add sentence overlap between paragraph chunks for context continuity.
        """
        if len(chunks) <= 1:
            return chunks
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get last N sentences from previous chunk
            prev_sentences = self._split_sentences(prev_chunk.text)
            if len(prev_sentences) > self.overlap_sentences:
                overlap = ' '.join(prev_sentences[-self.overlap_sentences:])
                curr_chunk.text = f"[...] {overlap}\n\n{curr_chunk.text}"
                curr_chunk.metadata['has_overlap'] = True
        
        return chunks


def smart_chunk(text: str, metadata: Dict = None, **kwargs) -> List[Tuple[str, Dict]]:
    """
    Convenience function for smart chunking.
    Returns list of (text, metadata) tuples for compatibility.
    """
    chunker = SmartChunker(**kwargs)
    chunks = chunker.chunk_text(text, metadata or {})
    return [(c.text, c.metadata) for c in chunks]


# Test
if __name__ == '__main__':
    # Test email
    email_text = """From: kyle@example.com
To: bob@example.com
Subject: Project Update
Date: 2024-01-15

Hey Bob,

Just wanted to give you a quick update on the project. We've made significant progress this week.

The new feature is almost complete. I've been testing it extensively and it's looking good.

Let me know if you have any questions.

Best,
Kyle"""
    
    chunker = SmartChunker(target_size=50, max_size=100)
    chunks = chunker.chunk_text(email_text, {'source': 'test'})
    
    print(f"Email chunked into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk.text.split())} words) ---")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
        print(f"Metadata: {chunk.metadata}")
