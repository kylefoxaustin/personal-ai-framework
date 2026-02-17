#!/usr/bin/env python3
"""
Email Service - Draft and send emails via Gmail or Outlook
"""
import os
import json
import base64
import pickle
import urllib.parse
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration directory
CONFIG_DIR = Path.home() / ".personal-ai"
CONFIG_DIR.mkdir(exist_ok=True)


@dataclass
class EmailDraft:
    """An email draft."""
    to: List[str]
    subject: str
    body: str
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    html: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "to": self.to,
            "subject": self.subject,
            "body": self.body,
            "cc": self.cc,
            "bcc": self.bcc,
            "html": self.html
        }
    
    def to_mailto_link(self) -> str:
        """Create mailto: link to open in default email client."""
        params = {
            'subject': self.subject,
            'body': self.body
        }
        if self.cc:
            params['cc'] = ','.join(self.cc)
        if self.bcc:
            params['bcc'] = ','.join(self.bcc)
        
        to = ','.join(self.to) if self.to else ''
        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        
        return f"mailto:{to}?{query}"
    
    def to_text(self) -> str:
        """Format as plain text for copying."""
        text = f"To: {', '.join(self.to) if self.to else '[recipients]'}\n"
        text += f"Subject: {self.subject}\n"
        if self.cc:
            text += f"CC: {', '.join(self.cc)}\n"
        text += "\n" + self.body
        return text


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard."""
    try:
        # Try xclip (Linux)
        process = subprocess.Popen(['xclip', '-selection', 'clipboard'], 
                                   stdin=subprocess.PIPE)
        process.communicate(text.encode())
        return process.returncode == 0
    except FileNotFoundError:
        try:
            # Try xsel (Linux alternative)
            process = subprocess.Popen(['xsel', '--clipboard', '--input'],
                                       stdin=subprocess.PIPE)
            process.communicate(text.encode())
            return process.returncode == 0
        except FileNotFoundError:
            return False


def open_url(url: str) -> bool:
    """Open URL in default browser/app."""
    try:
        subprocess.run(['xdg-open', url], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


class GmailService:
    """Gmail integration using OAuth."""
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.send',
              'https://www.googleapis.com/auth/gmail.compose']
    TOKEN_FILE = CONFIG_DIR / "gmail_token.pickle"
    CREDS_FILE = CONFIG_DIR / "gmail_credentials.json"
    
    def __init__(self):
        self.service = None
        self._authenticated = False
    
    def is_configured(self) -> bool:
        """Check if Gmail credentials are configured."""
        return self.CREDS_FILE.exists()
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication."""
        return self._authenticated and self.service is not None
    
    def setup_instructions(self) -> str:
        """Return setup instructions for Gmail."""
        return """
Gmail Setup Instructions:
=========================

1. Go to Google Cloud Console: https://console.cloud.google.com/

2. Create a new project or select existing one

3. Enable Gmail API:
   - Go to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click "Enable"

4. Create OAuth credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop app"
   - Download the JSON file

5. Save the credentials:
   mv ~/Downloads/client_secret_*.json ~/.personal-ai/gmail_credentials.json

6. Run authentication:
   ./run.sh email auth gmail
"""
    
    def authenticate(self) -> bool:
        """Authenticate with Gmail."""
        try:
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            print("❌ Google API libraries not installed.")
            print("   Run: pip3 install google-auth-oauthlib google-api-python-client")
            return False
        
        if not self.CREDS_FILE.exists():
            print("❌ Gmail credentials not found.")
            print(self.setup_instructions())
            return False
        
        creds = None
        
        # Load existing token
        if self.TOKEN_FILE.exists():
            with open(self.TOKEN_FILE, 'rb') as f:
                creds = pickle.load(f)
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.CREDS_FILE), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save token
            with open(self.TOKEN_FILE, 'wb') as f:
                pickle.dump(creds, f)
        
        self.service = build('gmail', 'v1', credentials=creds)
        self._authenticated = True
        print("✅ Gmail authenticated successfully")
        return True
    
    def send(self, draft: EmailDraft) -> Dict:
        """Send an email via Gmail."""
        if not self.is_authenticated():
            if not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
        
        try:
            # Create message
            if draft.html:
                message = MIMEMultipart('alternative')
                message.attach(MIMEText(draft.body, 'html'))
            else:
                message = MIMEText(draft.body)
            
            message['to'] = ', '.join(draft.to)
            message['subject'] = draft.subject
            
            if draft.cc:
                message['cc'] = ', '.join(draft.cc)
            if draft.bcc:
                message['bcc'] = ', '.join(draft.bcc)
            
            # Encode
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send
            result = self.service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            
            return {
                "success": True,
                "message_id": result.get('id'),
                "thread_id": result.get('threadId')
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def save_draft(self, draft: EmailDraft) -> Dict:
        """Save email as draft in Gmail."""
        if not self.is_authenticated():
            if not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
        
        try:
            if draft.html:
                message = MIMEMultipart('alternative')
                message.attach(MIMEText(draft.body, 'html'))
            else:
                message = MIMEText(draft.body)
            
            message['to'] = ', '.join(draft.to)
            message['subject'] = draft.subject
            
            if draft.cc:
                message['cc'] = ', '.join(draft.cc)
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            result = self.service.users().drafts().create(
                userId='me',
                body={'message': {'raw': raw}}
            ).execute()
            
            return {
                "success": True,
                "draft_id": result.get('id'),
                "message": "Draft saved to Gmail"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class OutlookService:
    """Outlook/Microsoft 365 integration using OAuth."""
    
    TOKEN_FILE = CONFIG_DIR / "outlook_token.json"
    CREDS_FILE = CONFIG_DIR / "outlook_credentials.json"
    SCOPES = ['Mail.Send', 'Mail.ReadWrite']
    
    def __init__(self):
        self._authenticated = False
        self._token = None
    
    def is_configured(self) -> bool:
        """Check if Outlook credentials are configured."""
        return self.CREDS_FILE.exists()
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication."""
        return self._authenticated and self._token is not None
    
    def setup_instructions(self) -> str:
        """Return setup instructions for Outlook."""
        return """
Outlook/Microsoft 365 Setup Instructions:
==========================================

1. Go to Azure Portal: https://portal.azure.com/

2. Navigate to "Azure Active Directory" > "App registrations"

3. Click "New registration":
   - Name: "Personal AI Email"
   - Supported account types: "Personal Microsoft accounts only"
   - Redirect URI: "http://localhost:8400" (Web)

4. After creation, note the "Application (client) ID"

5. Go to "Certificates & secrets" > "New client secret"
   - Copy the secret value immediately

6. Go to "API permissions" > "Add a permission":
   - Microsoft Graph > Delegated > Mail.Send, Mail.ReadWrite
   - Click "Grant admin consent"

7. Create credentials file:
   cat > ~/.personal-ai/outlook_credentials.json << CREDS
   {
       "client_id": "YOUR_CLIENT_ID",
       "client_secret": "YOUR_CLIENT_SECRET",
       "redirect_uri": "http://localhost:8400"
   }
   CREDS

8. Run authentication:
   ./run.sh email auth outlook
"""
    
    def authenticate(self) -> bool:
        """Authenticate with Outlook."""
        try:
            import msal
        except ImportError:
            print("❌ MSAL library not installed.")
            print("   Run: pip3 install msal")
            return False
        
        if not self.CREDS_FILE.exists():
            print("❌ Outlook credentials not found.")
            print(self.setup_instructions())
            return False
        
        with open(self.CREDS_FILE) as f:
            config = json.load(f)
        
        # Check for existing token
        if self.TOKEN_FILE.exists():
            with open(self.TOKEN_FILE) as f:
                self._token = json.load(f)
            self._authenticated = True
            print("✅ Outlook authenticated (using cached token)")
            return True
        
        # Create MSAL app
        app = msal.PublicClientApplication(
            config['client_id'],
            authority="https://login.microsoftonline.com/consumers"
        )
        
        # Get token interactively
        result = app.acquire_token_interactive(
            scopes=['https://graph.microsoft.com/Mail.Send',
                    'https://graph.microsoft.com/Mail.ReadWrite']
        )
        
        if 'access_token' in result:
            self._token = result
            with open(self.TOKEN_FILE, 'w') as f:
                json.dump(result, f)
            self._authenticated = True
            print("✅ Outlook authenticated successfully")
            return True
        else:
            print(f"❌ Authentication failed: {result.get('error_description', 'Unknown error')}")
            return False
    
    def send(self, draft: EmailDraft) -> Dict:
        """Send an email via Outlook."""
        if not self.is_authenticated():
            if not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
        
        try:
            import requests
            
            message = {
                "message": {
                    "subject": draft.subject,
                    "body": {
                        "contentType": "HTML" if draft.html else "Text",
                        "content": draft.body
                    },
                    "toRecipients": [
                        {"emailAddress": {"address": addr}} for addr in draft.to
                    ]
                }
            }
            
            if draft.cc:
                message["message"]["ccRecipients"] = [
                    {"emailAddress": {"address": addr}} for addr in draft.cc
                ]
            
            response = requests.post(
                "https://graph.microsoft.com/v1.0/me/sendMail",
                headers={
                    "Authorization": f"Bearer {self._token['access_token']}",
                    "Content-Type": "application/json"
                },
                json=message
            )
            
            if response.status_code == 202:
                return {"success": True}
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def save_draft(self, draft: EmailDraft) -> Dict:
        """Save email as draft in Outlook."""
        if not self.is_authenticated():
            if not self.authenticate():
                return {"success": False, "error": "Not authenticated"}
        
        try:
            import requests
            
            message = {
                "subject": draft.subject,
                "body": {
                    "contentType": "HTML" if draft.html else "Text",
                    "content": draft.body
                },
                "toRecipients": [
                    {"emailAddress": {"address": addr}} for addr in draft.to
                ]
            }
            
            if draft.cc:
                message["ccRecipients"] = [
                    {"emailAddress": {"address": addr}} for addr in draft.cc
                ]
            
            response = requests.post(
                "https://graph.microsoft.com/v1.0/me/messages",
                headers={
                    "Authorization": f"Bearer {self._token['access_token']}",
                    "Content-Type": "application/json"
                },
                json=message
            )
            
            if response.status_code == 201:
                return {
                    "success": True,
                    "draft_id": response.json().get('id'),
                    "message": "Draft saved to Outlook"
                }
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


class EmailService:
    """Unified email service supporting Gmail and Outlook."""
    
    def __init__(self, llm_url: str = "http://localhost:8080"):
        self.llm_url = llm_url
        self.gmail = GmailService()
        self.outlook = OutlookService()
        self._default_provider = None
    
    def get_provider(self, provider: Optional[str] = None):
        """Get email provider service."""
        if provider == "gmail":
            return self.gmail
        elif provider == "outlook":
            return self.outlook
        elif self._default_provider:
            return self._default_provider
        elif self.gmail.is_configured():
            return self.gmail
        elif self.outlook.is_configured():
            return self.outlook
        else:
            return None
    
    def status(self) -> Dict:
        """Get status of email providers."""
        return {
            "gmail": {
                "configured": self.gmail.is_configured(),
                "authenticated": self.gmail.is_authenticated()
            },
            "outlook": {
                "configured": self.outlook.is_configured(),
                "authenticated": self.outlook.is_authenticated()
            }
        }
    
    def authenticate(self, provider: str) -> bool:
        """Authenticate with a provider."""
        svc = self.get_provider(provider)
        if svc:
            return svc.authenticate()
        else:
            print(f"❌ Unknown provider: {provider}")
            return False
    
    def draft_email(
        self,
        topic: str,
        to: Optional[List[str]] = None,
        tone: str = "professional",
        use_style: bool = True
    ) -> EmailDraft:
        """
        Draft an email using the LLM.
        """
        import requests
        
        if use_style:
            prompt = f"### Instruction:\nWrite a {tone} email about: {topic}"
            if to:
                prompt += f"\nRecipients: {', '.join(to)}"
            prompt += "\n\nInclude a subject line at the start in the format 'Subject: ...'"
            prompt += "\n\n### Response:\n"
        else:
            prompt = f"Write a {tone} email about: {topic}"
            if to:
                prompt += f"\nRecipients: {', '.join(to)}"
            prompt += "\n\nInclude a subject line at the start in the format 'Subject: ...'"
        
        try:
            response = requests.post(
                f"{self.llm_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 800,
                    "temperature": 0.7,
                    "use_rag": True,
                    "rag_k": 3
                },
                timeout=60
            )
            
            if response.status_code == 200:
                text = response.json().get("text", "")
                
                lines = text.strip().split('\n')
                subject = "Email"
                body_start = 0
                
                for i, line in enumerate(lines):
                    if line.lower().startswith('subject:'):
                        subject = line[8:].strip()
                        body_start = i + 1
                        break
                
                body = '\n'.join(lines[body_start:]).strip()
                
                return EmailDraft(
                    to=to or [],
                    subject=subject,
                    body=body
                )
        except Exception as e:
            print(f"Draft generation failed: {e}")
        
        return EmailDraft(to=to or [], subject="Email", body="")
    
    def send(self, draft: EmailDraft, provider: Optional[str] = None) -> Dict:
        """Send an email."""
        svc = self.get_provider(provider)
        if not svc:
            return {
                "success": False,
                "error": "No email provider configured. Run: ./run.sh email setup gmail"
            }
        return svc.send(draft)
    
    def save_draft(self, draft: EmailDraft, provider: Optional[str] = None) -> Dict:
        """Save as draft in email provider."""
        svc = self.get_provider(provider)
        if not svc:
            return {
                "success": False,
                "error": "No email provider configured. Run: ./run.sh email setup gmail"
            }
        return svc.save_draft(draft)


def main():
    """CLI interface for email service."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Draft and send emails using your AI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Status command
    subparsers.add_parser("status", help="Show email provider status")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Show setup instructions")
    setup_parser.add_argument("provider", choices=["gmail", "outlook"])
    
    # Auth command
    auth_parser = subparsers.add_parser("auth", help="Authenticate with provider")
    auth_parser.add_argument("provider", choices=["gmail", "outlook"])
    
    # Draft command
    draft_parser = subparsers.add_parser("draft", help="Draft an email")
    draft_parser.add_argument("topic", help="Email topic")
    draft_parser.add_argument("-t", "--to", nargs="+", help="Recipients")
    draft_parser.add_argument("--tone", default="professional",
                              choices=["professional", "casual", "formal"])
    draft_parser.add_argument("-o", "--output", help="Save draft to file")
    draft_parser.add_argument("--open", action="store_true", 
                              help="Open in default email client")
    draft_parser.add_argument("--copy", action="store_true",
                              help="Copy to clipboard")
    draft_parser.add_argument("--save-to-gmail", action="store_true",
                              help="Save as draft in Gmail")
    draft_parser.add_argument("--save-to-outlook", action="store_true",
                              help="Save as draft in Outlook")
    
    # Send command
    send_parser = subparsers.add_parser("send", help="Draft and send an email")
    send_parser.add_argument("topic", help="Email topic")
    send_parser.add_argument("-t", "--to", nargs="+", required=True, help="Recipients")
    send_parser.add_argument("--provider", choices=["gmail", "outlook"])
    send_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    service = EmailService()
    
    if args.command == "status":
        status = service.status()
        print("\n📧 Email Provider Status")
        print("="*40)
        for provider, info in status.items():
            configured = "✅" if info["configured"] else "❌"
            authenticated = "✅" if info["authenticated"] else "❌"
            print(f"{provider.capitalize():10} Configured: {configured}  Auth: {authenticated}")
        print("\nTo configure: ./run.sh email setup gmail")
    
    elif args.command == "setup":
        provider = service.get_provider(args.provider)
        if provider:
            print(provider.setup_instructions())
        else:
            # Show instructions anyway
            if args.provider == "gmail":
                print(GmailService().setup_instructions())
            else:
                print(OutlookService().setup_instructions())
    
    elif args.command == "auth":
        service.authenticate(args.provider)
    
    elif args.command == "draft":
        print("📝 Generating draft...")
        draft = service.draft_email(args.topic, args.to, args.tone)
        
        # Display the draft
        print("\n" + "="*60)
        print("📧 Email Draft")
        print("="*60)
        print(f"To: {', '.join(draft.to) if draft.to else '[add recipients]'}")
        print(f"Subject: {draft.subject}")
        print("-"*60)
        print(draft.body)
        print("="*60)
        
        # Handle actions
        if args.output:
            with open(args.output, "w") as f:
                f.write(draft.to_text())
            print(f"✅ Saved to: {args.output}")
        
        if args.copy:
            if copy_to_clipboard(draft.to_text()):
                print("✅ Copied to clipboard")
            else:
                print("❌ Failed to copy (install xclip: sudo apt install xclip)")
        
        if args.open:
            mailto = draft.to_mailto_link()
            if open_url(mailto):
                print("✅ Opened in email client")
            else:
                print(f"❌ Failed to open. Manual link:\n{mailto}")
        
        if args.save_to_gmail:
            result = service.save_draft(draft, "gmail")
            if result["success"]:
                print("✅ Saved to Gmail Drafts")
            else:
                print(f"❌ {result.get('error')}")
        
        if args.save_to_outlook:
            result = service.save_draft(draft, "outlook")
            if result["success"]:
                print("✅ Saved to Outlook Drafts")
            else:
                print(f"❌ {result.get('error')}")
        
        # If no action specified, offer options
        if not any([args.output, args.copy, args.open, args.save_to_gmail, args.save_to_outlook]):
            print("\nWhat would you like to do?")
            print("  1) Open in email client")
            print("  2) Copy to clipboard")
            print("  3) Save to file")
            print("  4) Done")
            
            choice = input("\nChoice [1-4]: ").strip()
            
            if choice == "1":
                mailto = draft.to_mailto_link()
                if open_url(mailto):
                    print("✅ Opened in email client")
                else:
                    print(f"Open this link manually:\n{mailto}")
            elif choice == "2":
                if copy_to_clipboard(draft.to_text()):
                    print("✅ Copied to clipboard")
                else:
                    print("❌ Failed to copy (install xclip)")
            elif choice == "3":
                filename = input("Filename [draft.txt]: ").strip() or "draft.txt"
                with open(filename, "w") as f:
                    f.write(draft.to_text())
                print(f"✅ Saved to: {filename}")
    
    elif args.command == "send":
        print("📝 Generating draft...")
        draft = service.draft_email(args.topic, args.to)
        
        # Show draft
        print("\n" + "="*60)
        print("📧 Email to Send")
        print("="*60)
        print(f"To: {', '.join(draft.to)}")
        print(f"Subject: {draft.subject}")
        print("-"*60)
        print(draft.body)
        print("="*60)
        
        if not args.yes:
            confirm = input("\nSend this email? [y/N]: ")
            if confirm.lower() != 'y':
                print("❌ Cancelled")
                return
        
        result = service.send(draft, args.provider)
        if result["success"]:
            print("✅ Email sent successfully!")
        else:
            print(f"❌ Failed: {result.get('error')}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
