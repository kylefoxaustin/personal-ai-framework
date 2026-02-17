#!/usr/bin/env python3
"""
Daily Digest - Automated summary email of your knowledge base activity
"""
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import subprocess

LLM_URL = "http://localhost:8080"
CONFIG_DIR = Path.home() / ".personal-ai"
DIGEST_LOG = CONFIG_DIR / "digest_history.json"


@dataclass
class DigestData:
    """Data collected for the digest."""
    new_documents: int
    new_emails: int
    new_meetings: int
    new_datasheets: int
    recent_topics: List[str]
    action_items: List[str]
    hot_topics: List[Dict]
    insights: List[str]
    date: str


def get_recent_activity(hours: int = 24) -> Dict:
    """Query knowledge base for recent activity."""
    activity = {
        "documents": 0,
        "emails": 0,
        "meetings": 0,
        "datasheets": 0,
        "topics": []
    }
    
    try:
        # Get stats
        response = requests.get(f"{LLM_URL}/knowledge/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            activity["total_docs"] = stats.get("document_count", 0)
        
        # Search for recent content by querying common terms
        queries = [
            "meeting transcript today",
            "email from yesterday", 
            "action items",
            "project update",
            "i.MX processor"
        ]
        
        topics_found = set()
        for query in queries:
            try:
                resp = requests.post(
                    f"{LLM_URL}/search",
                    json={"query": query, "k": 5},
                    timeout=30
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for r in results:
                        # Extract topic hints from content
                        content = r.get("content", "")[:500].lower()
                        if "meeting" in content or "transcript" in content:
                            activity["meetings"] += 1
                        if "email" in content or "from:" in content:
                            activity["emails"] += 1
                        if "datasheet" in content or "i.mx" in content:
                            activity["datasheets"] += 1
            except:
                pass
        
    except Exception as e:
        print(f"Error getting activity: {e}")
    
    return activity


def extract_action_items() -> List[str]:
    """Search for action items in recent content."""
    action_items = []
    
    try:
        # Search for action item patterns
        queries = [
            "action items todo",
            "need to follow up",
            "deadline upcoming",
            "should complete"
        ]
        
        seen = set()
        for query in queries:
            resp = requests.post(
                f"{LLM_URL}/search",
                json={"query": query, "k": 3},
                timeout=30
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                for r in results:
                    content = r.get("content", "")
                    # Look for bullet points or checkbox patterns
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if any(line.startswith(p) for p in ['- [ ]', '- [x]', '• ', '- ', '* ']):
                            if 'action' in line.lower() or 'todo' in line.lower() or 'follow' in line.lower():
                                clean = line.lstrip('-•*[] x').strip()
                                if clean and clean not in seen and len(clean) > 10:
                                    seen.add(clean)
                                    action_items.append(clean)
    except Exception as e:
        print(f"Error extracting action items: {e}")
    
    return action_items[:5]


def get_hot_topics() -> List[Dict]:
    """Identify frequently discussed topics."""
    topics = []
    
    topic_queries = [
        ("i.MX processors", "i.MX processor ARM NXP"),
        ("Power management", "power management PMIC voltage"),
        ("Display systems", "display LCD screen graphics"),
        ("Embedded development", "embedded firmware microcontroller"),
        ("Project updates", "project status update timeline"),
    ]
    
    for name, query in topic_queries:
        try:
            resp = requests.post(
                f"{LLM_URL}/search",
                json={"query": query, "k": 10},
                timeout=30
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if len(results) >= 3:
                    topics.append({
                        "name": name,
                        "mentions": len(results),
                        "relevance": "high" if len(results) >= 7 else "medium"
                    })
        except:
            pass
    
    # Sort by mentions
    topics.sort(key=lambda x: x["mentions"], reverse=True)
    return topics[:5]


def generate_insight() -> str:
    """Generate an AI insight based on recent content."""
    try:
        prompt = """Based on the user's recent work with i.MX processors, embedded systems, and technical documentation, 
suggest ONE specific, actionable insight or recommendation. Be concise (1-2 sentences).
Focus on something practical they might want to explore or a connection between topics they've been working on."""
        
        resp = requests.post(
            f"{LLM_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.8,
                "use_rag": True,
                "rag_k": 5
            },
            timeout=60
        )
        
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
    except:
        pass
    
    return "Keep building awesome things!"


def generate_digest_email(data: DigestData, recipient: str) -> Dict:
    """Generate the digest email content."""
    
    # Build email body
    body = f"""Good morning!

🧠 YOUR AI DAILY DIGEST - {data.date}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📥 KNOWLEDGE BASE STATUS
- Total documents: {data.new_documents:,}
- Recent meetings processed: {data.new_meetings}
- Datasheets indexed: {data.new_datasheets}

"""
    
    if data.action_items:
        body += "📌 ACTION ITEMS DETECTED\n"
        for item in data.action_items[:5]:
            body += f"  • {item}\n"
        body += "\n"
    
    if data.hot_topics:
        body += "🔥 HOT TOPICS\n"
        for topic in data.hot_topics[:3]:
            relevance = "🔴" if topic["relevance"] == "high" else "🟡"
            body += f"  {relevance} {topic['name']} ({topic['mentions']} mentions)\n"
        body += "\n"
    
    if data.insights:
        body += "💡 AI INSIGHT\n"
        body += f"  {data.insights[0]}\n\n"
    
    body += """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Have a productive day!
— Your Personal AI

Generated by: personal-ai-framework
"""
    
    return {
        "to": recipient,
        "subject": f"🧠 Your AI Daily Digest - {data.date}",
        "body": body
    }


def send_digest(email: Dict, method: str = "mailto") -> bool:
    """Send the digest email."""
    
    if method == "mailto":
        # Open in email client
        import urllib.parse
        params = urllib.parse.urlencode({
            "subject": email["subject"],
            "body": email["body"]
        }, quote_via=urllib.parse.quote)
        
        mailto = f"mailto:{email['to']}?{params}"
        
        try:
            subprocess.Popen(
                ['xdg-open', mailto],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return True
        except:
            return False
    
    elif method == "save":
        # Save to file
        filename = f"digest_{datetime.now().strftime('%Y%m%d')}.txt"
        filepath = CONFIG_DIR / filename
        with open(filepath, 'w') as f:
            f.write(f"To: {email['to']}\n")
            f.write(f"Subject: {email['subject']}\n\n")
            f.write(email['body'])
        print(f"💾 Saved to: {filepath}")
        return True
    
    elif method == "print":
        print(f"\nTo: {email['to']}")
        print(f"Subject: {email['subject']}")
        print("-" * 50)
        print(email['body'])
        return True
    
    return False


def log_digest(data: DigestData):
    """Log digest to history."""
    CONFIG_DIR.mkdir(exist_ok=True)
    
    history = []
    if DIGEST_LOG.exists():
        try:
            with open(DIGEST_LOG) as f:
                history = json.load(f)
        except:
            history = []
    
    history.append({
        "date": data.date,
        "documents": data.new_documents,
        "action_items": len(data.action_items),
        "hot_topics": [t["name"] for t in data.hot_topics],
        "generated_at": datetime.now().isoformat()
    })
    
    # Keep last 30 days
    history = history[-30:]
    
    with open(DIGEST_LOG, 'w') as f:
        json.dump(history, f, indent=2)


def run_digest(recipient: str, method: str = "print", quiet: bool = False) -> bool:
    """Run the daily digest generation."""
    
    if not quiet:
        print("🧠 Generating Daily Digest...")
        print("=" * 50)
    
    # Collect data
    if not quiet:
        print("📊 Gathering activity...")
    activity = get_recent_activity()
    
    if not quiet:
        print("📌 Extracting action items...")
    action_items = extract_action_items()
    
    if not quiet:
        print("🔥 Identifying hot topics...")
    hot_topics = get_hot_topics()
    
    if not quiet:
        print("💡 Generating insights...")
    insight = generate_insight()
    
    # Build digest data
    data = DigestData(
        new_documents=activity.get("total_docs", 0),
        new_emails=activity.get("emails", 0),
        new_meetings=activity.get("meetings", 0),
        new_datasheets=activity.get("datasheets", 0),
        recent_topics=activity.get("topics", []),
        action_items=action_items,
        hot_topics=hot_topics,
        insights=[insight] if insight else [],
        date=datetime.now().strftime("%B %d, %Y")
    )
    
    # Generate email
    if not quiet:
        print("📧 Generating email...")
    email = generate_digest_email(data, recipient)
    
    # Log it
    log_digest(data)
    
    # Send/display
    if not quiet:
        print("=" * 50)
    
    success = send_digest(email, method)
    
    if not quiet and success:
        if method == "mailto":
            print("✅ Opened in email client!")
        elif method == "save":
            print("✅ Digest saved!")
        elif method == "print":
            print("\n✅ Digest generated!")
    
    return success


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate your AI daily digest"
    )
    parser.add_argument(
        "-t", "--to",
        default="",
        help="Recipient email address"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["print", "mailto", "save"],
        default="print",
        help="Delivery method (default: print)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (no progress output)"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show digest history"
    )
    parser.add_argument(
        "--schedule",
        metavar="TIME",
        help="Schedule daily digest (e.g., --schedule 08:00)"
    )
    
    args = parser.parse_args()
    
    if args.schedule:
        if not args.to:
            print("❌ Email required: ./run.sh digest --schedule 08:00 -t you@email.com")
            return
        setup_cron(args.to, args.schedule)
        return
    
    if args.history:
        if DIGEST_LOG.exists():
            with open(DIGEST_LOG) as f:
                history = json.load(f)
            print("📜 Digest History (last 30 days)")
            print("=" * 40)
            for entry in history[-10:]:
                print(f"  {entry['date']}: {entry['documents']} docs, {entry['action_items']} actions")
        else:
            print("No digest history yet.")
        return
    
    run_digest(
        recipient=args.to,
        method=args.method,
        quiet=args.quiet
    )


if __name__ == "__main__":
    main()


def setup_cron(email: str, time: str = "08:00"):
    """Set up cron job for daily digest."""
    import subprocess
    
    hour, minute = time.split(":")
    script_path = Path(__file__).resolve()
    
    cron_line = f'{minute} {hour} * * * cd {script_path.parent.parent} && /usr/bin/python3 {script_path} -t "{email}" -m mailto -q'
    
    # Get existing crontab
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        existing = result.stdout if result.returncode == 0 else ""
    except:
        existing = ""
    
    # Remove old digest entries
    lines = [l for l in existing.split('\n') if 'daily_digest.py' not in l and l.strip()]
    lines.append(cron_line)
    
    # Install new crontab
    new_crontab = '\n'.join(lines) + '\n'
    process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE)
    process.communicate(new_crontab.encode())
    
    print(f"✅ Cron job installed: Daily digest at {time}")
    print(f"   Email: {email}")
    print(f"   To remove: crontab -e (and delete the line)")


# Add to argparse in main()
