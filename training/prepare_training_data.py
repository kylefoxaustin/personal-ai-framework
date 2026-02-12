#!/usr/bin/env python3
"""
Prepare training data from sent emails for LoRA fine-tuning.
Extracts your writing style from emails you've written.
"""
import os
import json
import random
import email
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import re

class TrainingDataPreparator:
    """Extract training examples from sent emails."""
    
    def __init__(self, emails_dir: str, output_dir: str):
        self.emails_dir = Path(emails_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_email_content(self, filepath: Path) -> Dict:
        """Parse an email file and extract relevant fields."""
        try:
            with open(filepath, 'rb') as f:
                msg = email.message_from_binary_file(f)
            
            # Extract headers
            subject = msg.get('Subject', '')
            to_addr = msg.get('To', '')
            from_addr = msg.get('From', '')
            date = msg.get('Date', '')
            
            # Extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode('utf-8', errors='ignore')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
            
            # Clean up body
            body = self._clean_body(body)
            
            return {
                'subject': subject,
                'to': to_addr,
                'from': from_addr,
                'date': date,
                'body': body,
                'filepath': str(filepath)
            }
        except Exception as e:
            return None
    
    def _clean_body(self, body: str) -> str:
        """Clean email body text."""
        if not body:
            return ""
        
        # Remove quoted replies (lines starting with >)
        lines = body.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith('>'):
                continue
            # Stop at common reply indicators
            if any(indicator in line.lower() for indicator in [
                'original message', 'wrote:', 'from:', '-----', '___'
            ]):
                break
            cleaned_lines.append(line)
        
        body = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        body = re.sub(r'\n{3,}', '\n\n', body)
        body = body.strip()
        
        return body
    
    def create_instruction_pairs(self, emails: List[Dict]) -> List[Dict]:
        """
        Create instruction/response pairs from emails.
        Uses subject as prompt context, body as response.
        """
        pairs = []
        
        for email_data in emails:
            if not email_data or not email_data.get('body'):
                continue
            
            body = email_data['body']
            subject = email_data.get('subject', '')
            
            # Skip very short or very long emails
            word_count = len(body.split())
            if word_count < 20 or word_count > 1000:
                continue
            
            # Create instruction based on subject
            if subject:
                instruction = f"Write a professional email about: {subject}"
            else:
                instruction = "Write a professional email response."
            
            pairs.append({
                'instruction': instruction,
                'input': '',  # Could add context here
                'output': body,
                'subject': subject,
                'word_count': word_count
            })
        
        return pairs
    
    def create_conversation_pairs(self, emails: List[Dict]) -> List[Dict]:
        """
        Create conversation-style training data.
        Format: User asks question/topic, Assistant responds in your style.
        """
        pairs = []
        
        for email_data in emails:
            if not email_data or not email_data.get('body'):
                continue
            
            body = email_data['body']
            subject = email_data.get('subject', '').strip()
            
            word_count = len(body.split())
            if word_count < 30 or word_count > 800:
                continue
            
            # Skip emails that are mostly signatures or auto-replies
            if any(skip in body.lower() for skip in [
                'out of office', 'auto-reply', 'unsubscribe', 
                'sent from my iphone', 'sent from my android'
            ]):
                continue
            
            # Create varied prompts
            prompts = [
                f"Help me write about {subject}" if subject else "Help me write a message",
                f"Draft a response regarding {subject}" if subject else "Draft a professional response",
                f"Write something about {subject}" if subject else "Write a professional message",
            ]
            
            pairs.append({
                'conversations': [
                    {'role': 'user', 'content': random.choice(prompts)},
                    {'role': 'assistant', 'content': body}
                ],
                'subject': subject,
                'word_count': word_count
            })
        
        return pairs
    
    def scan_sent_emails(self) -> List[Dict]:
        """Scan the extracted sent items folders."""
        emails = []
        
        # Look for Sent Items folders
        sent_patterns = ['Sent Items', 'Sent', 'sent_items', 'sent']
        
        for pattern in sent_patterns:
            for sent_dir in self.emails_dir.rglob(f'*{pattern}*'):
                if sent_dir.is_dir():
                    print(f"Scanning: {sent_dir}")
                    for email_file in sent_dir.iterdir():
                        if email_file.is_file() and not email_file.name.startswith('.'):
                            email_data = self.extract_email_content(email_file)
                            if email_data and email_data.get('body'):
                                emails.append(email_data)
        
        print(f"Found {len(emails)} emails with content")
        return emails
    
    def prepare_alpaca_format(self, pairs: List[Dict], output_file: str):
        """Save in Alpaca format (instruction/input/output)."""
        alpaca_data = []
        for pair in pairs:
            alpaca_data.append({
                'instruction': pair['instruction'],
                'input': pair.get('input', ''),
                'output': pair['output']
            })
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(alpaca_data, f, indent=2)
        
        print(f"Saved {len(alpaca_data)} examples to {output_path}")
        return output_path
    
    def prepare_conversation_format(self, pairs: List[Dict], output_file: str):
        """Save in conversation format (for chat fine-tuning)."""
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(pairs, f, indent=2)
        
        print(f"Saved {len(pairs)} conversations to {output_path}")
        return output_path
    
    def prepare_all(self):
        """Run full preparation pipeline."""
        print("=" * 60)
        print("Training Data Preparation")
        print("=" * 60)
        
        # Scan emails
        emails = self.scan_sent_emails()
        
        if not emails:
            print("No emails found! Check the emails directory.")
            return
        
        # Create instruction pairs
        print("\nCreating instruction pairs...")
        instruction_pairs = self.create_instruction_pairs(emails)
        print(f"Created {len(instruction_pairs)} instruction pairs")
        
        # Create conversation pairs
        print("\nCreating conversation pairs...")
        conversation_pairs = self.create_conversation_pairs(emails)
        print(f"Created {len(conversation_pairs)} conversation pairs")
        
        # Shuffle and split
        random.shuffle(instruction_pairs)
        random.shuffle(conversation_pairs)
        
        # 90/10 train/val split
        split_idx_inst = int(len(instruction_pairs) * 0.9)
        split_idx_conv = int(len(conversation_pairs) * 0.9)
        
        # Save datasets
        print("\nSaving datasets...")
        self.prepare_alpaca_format(
            instruction_pairs[:split_idx_inst], 
            'train_alpaca.json'
        )
        self.prepare_alpaca_format(
            instruction_pairs[split_idx_inst:], 
            'val_alpaca.json'
        )
        self.prepare_conversation_format(
            conversation_pairs[:split_idx_conv],
            'train_conversations.json'
        )
        self.prepare_conversation_format(
            conversation_pairs[split_idx_conv:],
            'val_conversations.json'
        )
        
        # Stats
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total emails processed: {len(emails)}")
        print(f"Instruction pairs: {len(instruction_pairs)}")
        print(f"  - Training: {split_idx_inst}")
        print(f"  - Validation: {len(instruction_pairs) - split_idx_inst}")
        print(f"Conversation pairs: {len(conversation_pairs)}")
        print(f"  - Training: {split_idx_conv}")
        print(f"  - Validation: {len(conversation_pairs) - split_idx_conv}")
        
        # Sample
        if instruction_pairs:
            print("\nSample training example:")
            sample = random.choice(instruction_pairs)
            print(f"  Instruction: {sample['instruction']}")
            print(f"  Output preview: {sample['output'][:200]}...")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--emails-dir', default='knowledge/emails/extracted',
                        help='Directory with extracted emails')
    parser.add_argument('--output-dir', default='training/data',
                        help='Output directory for training data')
    
    args = parser.parse_args()
    
    preparator = TrainingDataPreparator(args.emails_dir, args.output_dir)
    preparator.prepare_all()
