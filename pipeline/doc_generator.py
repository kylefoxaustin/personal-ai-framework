#!/usr/bin/env python3
"""
Document Generator - Create professional documents using RAG and your writing style
"""
import os
import json
import requests
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime

# Document templates
TEMPLATES = {
    "technical_spec": {
        "name": "Technical Specification",
        "sections": ["Overview", "Requirements", "Architecture", "Implementation", "Testing", "Timeline"],
        "prompt_prefix": "Write a detailed technical specification document about",
    },
    "project_proposal": {
        "name": "Project Proposal", 
        "sections": ["Executive Summary", "Problem Statement", "Proposed Solution", "Benefits", "Timeline", "Budget", "Risks"],
        "prompt_prefix": "Write a professional project proposal for",
    },
    "status_report": {
        "name": "Status Report",
        "sections": ["Summary", "Accomplishments", "In Progress", "Blockers", "Next Steps"],
        "prompt_prefix": "Write a project status report about",
    },
    "one_pager": {
        "name": "One-Pager",
        "sections": ["Overview", "Key Points", "Benefits", "Call to Action"],
        "prompt_prefix": "Write a concise one-page summary about",
    },
    "meeting_notes": {
        "name": "Meeting Notes",
        "sections": ["Attendees", "Agenda", "Discussion", "Decisions", "Action Items"],
        "prompt_prefix": "Write meeting notes for a meeting about",
    },
    "email": {
        "name": "Email",
        "sections": [],
        "prompt_prefix": "Write a professional email about",
    },
    "blog_post": {
        "name": "Blog Post",
        "sections": ["Introduction", "Main Content", "Conclusion"],
        "prompt_prefix": "Write an engaging blog post about",
    },
}


@dataclass
class GeneratedDocument:
    """A generated document."""
    title: str
    doc_type: str
    content: str
    context_used: List[str]
    created_at: str
    
    def to_markdown(self) -> str:
        """Export as markdown."""
        return self.content
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "doc_type": self.doc_type,
            "content": self.content,
            "context_used": self.context_used,
            "created_at": self.created_at
        }


class DocumentGenerator:
    """Generate professional documents using RAG and LLM."""
    
    def __init__(self, llm_url: str = "http://localhost:8080"):
        self.llm_url = llm_url
    
    def list_templates(self) -> Dict[str, str]:
        """List available document templates."""
        return {k: v["name"] for k, v in TEMPLATES.items()}
    
    def _get_rag_context(self, topic: str, k: int = 5) -> List[str]:
        """Retrieve relevant context from knowledge base."""
        try:
            response = requests.post(
                f"{self.llm_url}/search",
                json={"query": topic, "k": k},
                timeout=30
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                return [r["content"] for r in results]
        except Exception as e:
            print(f"RAG retrieval failed: {e}")
        return []
    
    def _generate_with_llm(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        use_rag: bool = True,
        rag_k: int = 5
    ) -> tuple[str, List[str]]:
        """Generate text using the LLM."""
        try:
            response = requests.post(
                f"{self.llm_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "use_rag": use_rag,
                    "rag_k": rag_k
                },
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("text", ""), data.get("context_used", [])
        except Exception as e:
            print(f"LLM generation failed: {e}")
        return "", []
    
    def generate(
        self,
        doc_type: str,
        topic: str,
        title: Optional[str] = None,
        additional_context: Optional[str] = None,
        use_rag: bool = True,
        use_style: bool = True,
        max_tokens: int = 2000
    ) -> GeneratedDocument:
        """
        Generate a document.
        
        Args:
            doc_type: Type of document (from TEMPLATES)
            topic: Main topic/subject
            title: Optional document title
            additional_context: Extra context to include
            use_rag: Whether to use RAG for context
            use_style: Whether to use personal writing style
            max_tokens: Maximum tokens for generation
        
        Returns:
            GeneratedDocument object
        """
        if doc_type not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            raise ValueError(f"Unknown document type: {doc_type}. Available: {available}")
        
        template = TEMPLATES[doc_type]
        
        if title is None:
            title = f"{template['name']}: {topic}"
        
        # Build the prompt
        prompt_parts = []
        
        # Add style instruction if enabled
        if use_style:
            prompt_parts.append("### Instruction:")
        
        prompt_parts.append(f"{template['prompt_prefix']} {topic}.")
        
        # Add section guidance if template has sections
        if template["sections"]:
            sections_str = ", ".join(template["sections"])
            prompt_parts.append(f"\nInclude these sections: {sections_str}")
        
        # Add additional context
        if additional_context:
            prompt_parts.append(f"\nAdditional context: {additional_context}")
        
        # Add formatting instructions
        prompt_parts.append("\nFormat the document in clean markdown with proper headers.")
        prompt_parts.append(f"\nDocument title: {title}")
        
        if use_style:
            prompt_parts.append("\n\n### Response:")
        
        full_prompt = "\n".join(prompt_parts)
        
        print(f"📝 Generating {template['name']}...")
        print(f"   Topic: {topic}")
        
        # Generate content
        content, context_used = self._generate_with_llm(
            full_prompt,
            max_tokens=max_tokens,
            use_rag=use_rag,
            rag_k=5 if use_rag else 0
        )
        
        # Clean up response
        content = content.strip()
        if content.startswith("### Response:"):
            content = content[13:].strip()
        
        # Add title if not in content
        if not content.startswith("#"):
            content = f"# {title}\n\n{content}"
        
        return GeneratedDocument(
            title=title,
            doc_type=doc_type,
            content=content,
            context_used=context_used or [],
            created_at=datetime.now().isoformat()
        )
    
    def generate_from_template(
        self,
        doc_type: str,
        variables: Dict[str, str],
        use_rag: bool = True
    ) -> GeneratedDocument:
        """
        Generate a document filling in template variables.
        
        Args:
            doc_type: Document type
            variables: Dict of variable name -> value
            use_rag: Whether to use RAG
        
        Returns:
            GeneratedDocument
        """
        topic = variables.get("topic", "")
        title = variables.get("title")
        additional = "\n".join(f"{k}: {v}" for k, v in variables.items() 
                               if k not in ["topic", "title"])
        
        return self.generate(
            doc_type=doc_type,
            topic=topic,
            title=title,
            additional_context=additional if additional else None,
            use_rag=use_rag
        )


def main():
    """CLI interface for document generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate professional documents using your knowledge base"
    )
    parser.add_argument(
        "type",
        choices=list(TEMPLATES.keys()),
        help="Document type to generate"
    )
    parser.add_argument(
        "topic",
        help="Main topic or subject"
    )
    parser.add_argument(
        "-t", "--title",
        help="Document title (auto-generated if not provided)"
    )
    parser.add_argument(
        "-c", "--context",
        help="Additional context to include"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (markdown)"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Don't use knowledge base for context"
    )
    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Don't use personal writing style"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum tokens for generation (default: 2000)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available document types"
    )
    
    args = parser.parse_args()
    
    generator = DocumentGenerator()
    
    if args.list:
        print("Available document types:")
        for key, name in generator.list_templates().items():
            print(f"  {key:20} - {name}")
        return
    
    print(f"\n{'='*60}")
    print(f"📄 Document Generator")
    print(f"{'='*60}\n")
    
    doc = generator.generate(
        doc_type=args.type,
        topic=args.topic,
        title=args.title,
        additional_context=args.context,
        use_rag=not args.no_rag,
        use_style=not args.no_style,
        max_tokens=args.max_tokens
    )
    
    output = doc.to_markdown()
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\n✅ Saved to: {args.output}")
    else:
        print("\n" + "="*60)
        print(output)
    
    if doc.context_used:
        print(f"\n📚 Used {len(doc.context_used)} sources from knowledge base")


if __name__ == "__main__":
    main()
