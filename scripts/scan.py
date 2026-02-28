#!/usr/bin/env python3
"""
Iran Monitor - Automated Scan Script
Runs via GitHub Actions every hour. Calls Claude API with web search
to analyze the current situation in Iran, then saves results as JSON.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# pip install anthropic
import anthropic

# ============================================
# CONFIG
# ============================================
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096
ISRAEL_TZ = timezone(timedelta(hours=3))  # Israel Standard Time (approx)

# ============================================
# LOAD EXISTING DATA
# ============================================
def load_json(filename, default=None):
    path = DATA_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return default if default is not None else {}

def save_json(filename, data):
    path = DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ============================================
# BUILD PROMPT
# ============================================
def build_system_prompt(history, sources, user_intel, feedback):
    now = datetime.now(ISRAEL_TZ)
    time_str = now.strftime("%d/%m/%Y %H:%M:%S")

    # Previous scans context
    prev_context = ""
    for i, scan in enumerate(history[:3]):
        t = scan.get("time_str", "")
        c = scan.get("content", "")[:1200]
        prev_context += f"--- סקירה קודמת #{i+1} ({t}) ---\n{c}\n\n"

    # User intel
    intel_block = ""
    if user_intel:
        intel_block = "\n\n## מידע ממקור ישיר (עדיפות עליונה - המשתמש הזין את זה):\n"
        for item in user_intel:
            intel_block += f"- [{item.get('priority', 'normal')}] {item.get('text', '')} ({item.get('time', '')})\n"

    # Feedback
    fb_block = ""
    if feedback:
        recent_fb = feedback[-10:]
        fb_block = "\n\n## משוב מהמשתמש - למד מזה:\n"
        for fb in recent_fb:
            fb_block += f"- {fb.get('text', '')} ({fb.get('time', '')})\n"

    # Source reliability
    src_block = ""
    if sources:
        src_block = "\n\n## היסטוריית אמינות מקורות:\n"
        for name, data in sorted(sources.items(), key=lambda x: x[1].get("score", 50), reverse=True):
            src_block += f"- {name}: ציון {data.get('score', 50)}/100 (אזכורים: {data.get('mentions', 0)})\n"

    system_prompt = f"""אתה אנליסט מודיעין צבאי-אסטרטגי ישראלי ברמה הגבוהה ביותר.
הזמן הנוכחי: {time_str} (שעון ישראל)

## המשימה:
סקירה מבצעית קצרה ומדויקת של המצב באיראן. חפש חדשות עדכניות ובצע ניתוח.

## מבנה הסקירה (חובה לכתוב בדיוק בפורמט הזה):

### מצב נוכחי
- מה הותקף, איפה, על ידי מי, באיזה אמצעים
- נזקים מאומתים vs. נטענים
- תגובת איראן (רשמית + לא רשמית)
- תגובות בינלאומיות חשובות

### תרחישים - שעה קרובה
- מה הכי סביר שיקרה עכשיו
- סיכויים לכל תרחיש (אחוזים)

### תרחישים - 24-72 שעות
- לאן זה מתפתח
- מה המשמעות האסטרטגית

### הערכת מקורות
ליד כל מידע סמן:
- ✅ = מאומת ממספר מקורות
- ⚠️ = מקור בודד / לא מאושר
- ❓ = שמועה / לא מאומת

## כללי עבודה:
1. חפש חדשות בשפות: אנגלית, ערבית, פרסית, עברית
2. מקורות מועדפים: Epoch Times, ISW, Jane's, חוקרי OSINT (Aurora Intel, OSINTdefender), כתבים צבאיים ישראליים, Al Arabiya, Times of Israel, Jerusalem Post, i24 News
3. מקורות פרסיים (ציין שזו מדיה ממלכתית): IRNA, Fars News, Press TV, Tasnim
4. מקורות להיזהר מהם: BBC (נטייה אנטי-ישראלית), Al Jazeera (מוטה), CNN (שטחי) - אם משתמש בהם, ציין את ההטיה
5. עדיף "לא ידוע עדיין" מאשר ניחוש
6. אם בסקירות קודמות מישהו טעה - ציין זאת
7. כתוב בעברית, קצר, בנקודות
8. בסוף: "מקורות עיקריים" - רשימת המקורות ששימשו עם דירוג אמינות

## שפות חיפוש:
- אנגלית: Iran attack, Iran strike, Iran military, Iran war
- ערבית: ايران هجوم, ايران حرب, ايران ضربة
- פרסית: ایران حمله, ایران جنگ
- עברית: איראן תקיפה, איראן מלחמה, איראן מתקפה
{intel_block}
{fb_block}
{src_block}
"""

    if prev_context:
        system_prompt += f"\n## סקירות קודמות (להקשר והשוואה):\n{prev_context}"

    return system_prompt

# ============================================
# SOURCE TRACKING
# ============================================
SOURCE_PATTERNS = {
    "ISW": ["ISW", "Institute for the Study of War"],
    "Jane's": ["Jane's", "Janes"],
    "IRNA": ["IRNA"],
    "Fars News": ["Fars News"],
    "Tasnim": ["Tasnim"],
    "Press TV": ["Press TV"],
    "Al Arabiya": ["Al Arabiya"],
    "Al Jazeera": ["Al Jazeera"],
    "Reuters": ["Reuters"],
    "AP": ["Associated Press", " AP "],
    "Epoch Times": ["Epoch Times"],
    "Times of Israel": ["Times of Israel"],
    "Jerusalem Post": ["Jerusalem Post"],
    "i24": ["i24"],
    "Ynet": ["Ynet", "ynet", "ynetnews"],
    "Walla": ["Walla", "וואלה"],
    "Kan News": ["כאן חדשות", "Kan News", "כאן 11"],
    "Channel 12": ["חדשות 12", "Channel 12"],
    "Channel 13": ["חדשות 13", "Channel 13"],
    "OSINT Analysts": ["OSINT", "osint"],
    "Telegram": ["טלגרם", "Telegram", "telegram"],
    "X/Twitter": ["Twitter", "X.com"],
    "BBC": ["BBC"],
    "CNN": ["CNN"],
    "Aurora Intel": ["Aurora Intel"],
    "OSINTdefender": ["OSINTdefender"],
}

def update_sources(content, sources):
    for name, patterns in SOURCE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in content.lower():
                if name not in sources:
                    sources[name] = {"score": 50, "mentions": 0}
                sources[name]["mentions"] = sources[name].get("mentions", 0) + 1

                # Adjust score based on reliability markers near the source
                content_lower = content.lower()
                idx = content_lower.find(pattern.lower())
                if idx >= 0:
                    context_window = content[max(0, idx-50):idx+len(pattern)+100]
                    if any(m in context_window for m in ["✅", "אמינות גבוהה", "מאומת"]):
                        sources[name]["score"] = min(100, sources[name].get("score", 50) + 3)
                    if any(m in context_window for m in ["❓", "אמינות נמוכה", "לא מאומת", "שמועה"]):
                        sources[name]["score"] = max(0, sources[name].get("score", 50) - 3)
                    if any(m in context_window for m in ["⚠️", "מקור בודד", "לא מאושר"]):
                        sources[name]["score"] = max(0, sources[name].get("score", 50) - 1)
                break

    return sources

# ============================================
# MAIN SCAN
# ============================================
def run_scan(extra_intel=None):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print("Loading existing data...")
    history = load_json("history.json", [])
    sources = load_json("sources.json", {})
    user_intel = load_json("user_intel.json", [])
    feedback = load_json("feedback.json", [])

    # Add extra intel if provided (from manual trigger)
    if extra_intel:
        now = datetime.now(ISRAEL_TZ)
        user_intel.append({
            "text": extra_intel,
            "priority": "high",
            "time": now.strftime("%d/%m/%Y %H:%M")
        })
        save_json("user_intel.json", user_intel)

    print("Building prompt...")
    system_prompt = build_system_prompt(history, sources, user_intel, feedback)

    user_message = """בצע סקירה מבצעית עדכנית. מה קורה באיראן עכשיו?

חפש מידע עדכני ביותר על:
1. התקפות על איראן - מה הותקף, על ידי מי, באיזה נשק
2. תגובת איראן - רשמית ולא רשמית
3. תגובות בינלאומיות
4. מה צפוי לקרות בשעה הקרובה ובימים הקרובים

חפש בשפות: אנגלית, ערבית, פרסית, עברית.
תן לי סקירה קצרה, חדה, עם מקורות ורמת אמינות."""

    print("Calling Claude API with web search...")
    client = anthropic.Anthropic(api_key=api_key)

    # Use web search tool for real-time news
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 10
        }],
        messages=[{"role": "user", "content": user_message}]
    )

    # Extract text content from response
    content = ""
    for block in response.content:
        if block.type == "text":
            content += block.text

    if not content:
        print("ERROR: No text content in response", file=sys.stderr)
        sys.exit(1)

    print(f"Got response ({len(content)} chars)")

    # Create scan record
    now = datetime.now(ISRAEL_TZ)
    timestamp = int(time.time())

    scan = {
        "timestamp": timestamp,
        "time_str": now.strftime("%d/%m/%Y %H:%M:%S"),
        "content": content,
        "summary": content[:150].replace("#", "").replace("*", "").replace("\n", " ").strip(),
        "model": MODEL,
    }

    # Save as latest
    print("Saving latest.json...")
    save_json("latest.json", scan)

    # Prepend to history
    history.insert(0, scan)
    history = history[:50]  # Keep max 50
    print("Saving history.json...")
    save_json("history.json", history)

    # Update source reliability
    print("Updating source reliability...")
    sources = update_sources(content, sources)
    save_json("sources.json", sources)

    print(f"Scan complete at {now.strftime('%H:%M:%S')}")
    print(f"Summary: {scan['summary'][:80]}...")
    return scan

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    extra = None
    if len(sys.argv) > 1:
        extra = " ".join(sys.argv[1:])
        print(f"Extra intel from command line: {extra}")

    # Also check environment variable for intel (from GitHub Actions)
    env_intel = os.environ.get("USER_INTEL")
    if env_intel:
        extra = (extra or "") + " " + env_intel
        print(f"Extra intel from env: {env_intel}")

    run_scan(extra_intel=extra)
