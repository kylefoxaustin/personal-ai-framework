#!/usr/bin/env python3
"""
Meeting Summarizer - Transcribe and summarize audio/video recordings
Uses Whisper for transcription and LLM for summarization
"""
import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime

# Supported formats
AUDIO_FORMATS = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.wma', '.amr', '.aac'}
VIDEO_FORMATS = {'.mp4', '.webm', '.mkv', '.mov', '.avi'}
SUPPORTED_FORMATS = AUDIO_FORMATS | VIDEO_FORMATS


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with timing."""
    start: float
    end: float
    text: str
    
    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }


@dataclass
class MeetingSummary:
    """Complete meeting summary with transcript and analysis."""
    title: str
    date: str
    duration_seconds: float
    transcript_text: str
    segments: List[TranscriptSegment]
    summary: str
    key_points: List[str]
    action_items: List[str]
    decisions: List[str]
    participants: List[str]
    source_file: str
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "date": self.date,
            "duration_seconds": self.duration_seconds,
            "transcript_text": self.transcript_text,
            "segments": [s.to_dict() for s in self.segments],
            "summary": self.summary,
            "key_points": self.key_points,
            "action_items": self.action_items,
            "decisions": self.decisions,
            "participants": self.participants,
            "source_file": self.source_file
        }
    
    def to_markdown(self) -> str:
        """Export as markdown document."""
        md = f"# {self.title}\n\n"
        md += f"**Date:** {self.date}\n"
        md += f"**Duration:** {self.duration_seconds / 60:.1f} minutes\n"
        md += f"**Source:** {self.source_file}\n\n"
        
        if self.participants:
            md += "## Participants\n"
            for p in self.participants:
                md += f"- {p}\n"
            md += "\n"
        
        md += "## Summary\n"
        md += f"{self.summary}\n\n"
        
        if self.key_points:
            md += "## Key Points\n"
            for point in self.key_points:
                md += f"- {point}\n"
            md += "\n"
        
        if self.action_items:
            md += "## Action Items\n"
            for item in self.action_items:
                md += f"- [ ] {item}\n"
            md += "\n"
        
        if self.decisions:
            md += "## Decisions Made\n"
            for decision in self.decisions:
                md += f"- {decision}\n"
            md += "\n"
        
        md += "## Full Transcript\n\n"
        md += self.transcript_text
        
        return md


class MeetingSummarizer:
    """Transcribe and summarize meetings."""
    
    def __init__(self, whisper_model: str = "base", llm_url: str = "http://localhost:8080"):
        """
        Initialize the meeting summarizer.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            llm_url: URL of the LLM server for summarization
        """
        self.whisper_model = whisper_model
        self.llm_url = llm_url
        self._whisper = None
    
    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper is None:
            try:
                import whisper
                print(f"📥 Loading Whisper model: {self.whisper_model}")
                self._whisper = whisper.load_model(self.whisper_model)
                print("✅ Whisper loaded")
            except ImportError:
                raise ImportError(
                    "Whisper not installed. Run: pip install openai-whisper"
                )
        return self._whisper
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _extract_audio(self, input_path: str, output_path: str) -> bool:
        """
        Extract/convert audio to WAV format for Whisper.
        
        Args:
            input_path: Path to input audio/video file
            output_path: Path for output WAV file
        
        Returns:
            True if successful
        """
        try:
            cmd = [
                "ffmpeg", "-y",  # Overwrite output
                "-i", input_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
                "-ac", "1",  # Mono
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"FFmpeg error: {e}")
            return False
    
    def _get_duration(self, file_path: str) -> float:
        """Get duration of audio/video file in seconds."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
            
        except Exception:
            return 0.0
    
    def transcribe(self, file_path: str) -> tuple[str, List[TranscriptSegment]]:
        """
        Transcribe an audio/video file.
        
        Args:
            file_path: Path to the audio/video file
        
        Returns:
            Tuple of (full_text, segments)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        
        # Check FFmpeg
        if not self._check_ffmpeg():
            raise RuntimeError(
                "FFmpeg not installed. Run: sudo apt install ffmpeg"
            )
        
        # Load Whisper
        whisper_model = self._load_whisper()
        
        # Convert to WAV if needed
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            print(f"🔄 Converting to WAV: {file_path.name}")
            if not self._extract_audio(str(file_path), tmp_path):
                raise RuntimeError("Failed to extract audio with FFmpeg")
            
            print(f"🎤 Transcribing with Whisper ({self.whisper_model})...")
            result = whisper_model.transcribe(
                tmp_path,
                language="en",
                verbose=False
            )
            
            # Extract segments
            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip()
                ))
            
            full_text = result.get("text", "").strip()
            print(f"✅ Transcribed {len(segments)} segments")
            
            return full_text, segments
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def summarize_transcript(self, transcript: str, title: str = "Meeting") -> Dict:
        """
        Summarize a transcript using the LLM.
        
        Args:
            transcript: Full transcript text
            title: Meeting title/topic
        
        Returns:
            Dict with summary, key_points, action_items, decisions, participants
        """
        import requests
        
        prompt = f"""Analyze this meeting transcript and provide:
1. A brief 2-3 sentence summary
2. Key points discussed (bullet list)
3. Action items with owners if mentioned (bullet list)
4. Decisions made (bullet list)
5. Participants mentioned by name (list)

Meeting: {title}

Transcript:
{transcript[:8000]}  # Limit to avoid context overflow

Respond in this exact JSON format:
{{
    "summary": "Brief summary here",
    "key_points": ["point 1", "point 2"],
    "action_items": ["action 1", "action 2"],
    "decisions": ["decision 1", "decision 2"],
    "participants": ["Name 1", "Name 2"]
}}

JSON response:"""

        try:
            response = requests.post(
                f"{self.llm_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "temperature": 0.3,
                    "use_rag": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                text = response.json().get("text", "")
                
                # Try to parse JSON from response
                try:
                    # Find JSON in response
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
                
                # Fallback if JSON parsing fails
                return {
                    "summary": text[:500],
                    "key_points": [],
                    "action_items": [],
                    "decisions": [],
                    "participants": []
                }
            else:
                print(f"LLM request failed: {response.status_code}")
                return {
                    "summary": "Summarization failed",
                    "key_points": [],
                    "action_items": [],
                    "decisions": [],
                    "participants": []
                }
                
        except Exception as e:
            print(f"Summarization error: {e}")
            return {
                "summary": f"Error: {str(e)}",
                "key_points": [],
                "action_items": [],
                "decisions": [],
                "participants": []
            }
    
    def process_meeting(
        self,
        file_path: str,
        title: Optional[str] = None,
        add_to_knowledge_base: bool = False
    ) -> MeetingSummary:
        """
        Process a meeting recording: transcribe and summarize.
        
        Args:
            file_path: Path to audio/video file
            title: Optional meeting title (defaults to filename)
            add_to_knowledge_base: Whether to add transcript to RAG
        
        Returns:
            MeetingSummary object
        """
        file_path = Path(file_path)
        
        if title is None:
            title = file_path.stem.replace("_", " ").replace("-", " ").title()
        
        print(f"\n{'='*60}")
        print(f"📼 Processing: {file_path.name}")
        print(f"📋 Title: {title}")
        print(f"{'='*60}\n")
        
        # Get duration
        duration = self._get_duration(str(file_path))
        print(f"⏱️  Duration: {duration/60:.1f} minutes")
        
        # Transcribe
        transcript_text, segments = self.transcribe(str(file_path))
        
        # Summarize
        print("🧠 Generating summary...")
        analysis = self.summarize_transcript(transcript_text, title)
        
        # Create summary object
        summary = MeetingSummary(
            title=title,
            date=datetime.now().strftime("%Y-%m-%d"),
            duration_seconds=duration,
            transcript_text=transcript_text,
            segments=segments,
            summary=analysis.get("summary", ""),
            key_points=analysis.get("key_points", []),
            action_items=analysis.get("action_items", []),
            decisions=analysis.get("decisions", []),
            participants=analysis.get("participants", []),
            source_file=str(file_path)
        )
        
        # Add to knowledge base if requested
        if add_to_knowledge_base:
            self._add_to_knowledge_base(summary)
        
        print(f"\n✅ Processing complete!")
        return summary
    
    def _add_to_knowledge_base(self, summary: MeetingSummary):
        """Add meeting transcript to the knowledge base."""
        import requests
        
        try:
            # Create document content
            content = f"""Meeting: {summary.title}
Date: {summary.date}
Duration: {summary.duration_seconds/60:.1f} minutes

Summary: {summary.summary}

Key Points:
{chr(10).join('- ' + p for p in summary.key_points)}

Action Items:
{chr(10).join('- ' + a for a in summary.action_items)}

Transcript:
{summary.transcript_text}
"""
            
            response = requests.post(
                f"{self.llm_url}/ingest",
                json={
                    "content": content,
                    "metadata": {
                        "source_file": summary.source_file,
                        "category": "meetings",
                        "title": summary.title,
                        "date": summary.date
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                print(f"📚 Added to knowledge base")
            else:
                print(f"⚠️ Failed to add to knowledge base: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Knowledge base error: {e}")


def main():
    """CLI interface for meeting summarizer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transcribe and summarize meeting recordings"
    )
    parser.add_argument("file", help="Path to audio/video file")
    parser.add_argument("-t", "--title", help="Meeting title")
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (markdown)"
    )
    parser.add_argument(
        "--add-to-kb",
        action="store_true",
        help="Add transcript to knowledge base"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of markdown"
    )
    
    args = parser.parse_args()
    
    # Process meeting
    summarizer = MeetingSummarizer(whisper_model=args.model)
    summary = summarizer.process_meeting(
        args.file,
        title=args.title,
        add_to_knowledge_base=args.add_to_kb
    )
    
    # Output
    if args.json:
        output = json.dumps(summary.to_dict(), indent=2)
    else:
        output = summary.to_markdown()
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"📄 Saved to: {args.output}")
    else:
        print("\n" + "="*60)
        print(output)


if __name__ == "__main__":
    main()
