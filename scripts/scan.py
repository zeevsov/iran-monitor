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
MODEL = "claude-opus-4-6"
MAX_TOKENS = 4000
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

    # Previous scan context (only last 1, truncated to save tokens)
    prev_context = ""
    if history:
        last = history[0]
        t = last.get("time_str", "")
        c = last.get("content", "")[:600]
        prev_context = f"--- סקירה קודמת ({t}) ---\n{c}\n"

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

    system_prompt = f"""אתה עורך מודיעין בכיר. תפקידך: לכתוב תדריך מבצעי קצר, חד, עם שורות תחתונות בלבד.

הזמן: {time_str} (ישראל)

## הכלל המרכזי:
אתה לא עיתונאי שמעתיק הכל. אתה אנליסט שנותן שורה תחתונה.
לא בולטים (bullet points). כתוב פסקאות קצרות של טקסט רציף.
ציין מקורות בתוך הטקסט (למשל: "לפי רויטרס..." או "ב-Times of Israel דווח ש...").
לא ציטוטים של פוליטיקאים (לא גרהאם, לא מדבדב, לא האו"ם). רק אם יש הצהרה שמשנה את המצב בפועל.
לא "תורים בסופר", לא "הפגנות בוטלו", לא PR. רק ברומו של עולם.
ליד מידע מרכזי סמן ✅ מאומת | ⚠️ מקור בודד | ❓ שמועה.
כל סעיף = 2-4 משפטים מקסימום. לא יותר.
אל תכתוב באנרים/כותרות מיותרים כמו "אירוע מתפתח בזמן אמת" או "מצב משתנה בדקות". תתחיל ישר עם התוכן.
אל תכתוב הקדמות כמו "מבצע סריקה" או "רגע...". תתחיל ישר מהכותרת הראשונה.

## מבנה חובה (כתוב בדיוק את הכותרות האלה):

### 🔴 מצב מבצעי נוכחי
סיכום של 4-5 משפטים: מה המצב עד עכשיו. מתי התחיל, מי תוקף מי, באילו אמצעים, סדר גודל. שורה תחתונה בלבד.

### 🔥 אירועים חמים
מה חדש ומשמעותי בשעה-שעתיים אחרונות. 3-4 פסקאות קצרות (2-4 משפטים כל אחת). דגש על מידע חם ועדכני: תקיפות חדשות, שיגורים, חיסולים, תנועות כוחות. אם יש אירועים בינלאומיים בולטים הקשורים למצב (הצהרות מעצמות, החלטות באו"ם, סנקציות, פעולות צבאיות של מדינות נוספות) - הוסף אותם כאן. אם אין חדש - תגיד "ללא שינוי משמעותי".

### 🇮🇱 המצב בישראל
פסקה אחת עד שתיים: אזעקות, יירוטים, נפגעים (מספרים), הנחיות פיקוד העורף, מצב שדות תעופה. רק פרקטיקה - מה קורה עכשיו, לא מה הצהירו.

### 📊 תרחיש - שעה קרובה
פסקה אחת: מה הכי סביר שיקרה עכשיו. אילו גלים צפויים, מאיפה, באיזה אמצעים. נקודות שבר אפשריות.

### 📈 תרחיש - התפתחות המלחמה
2-3 פסקאות: לאן המלחמה הולכת בימים-שבועות הקרובים. אירועים אפשריים בהמשך הדרך. האם יש צפי להסלמה/הפסקה/סגירה. מה מכפיל הלחץ הבא. כלול גם השפעות אזוריות ובינלאומיות אם רלוונטי.

### מקורות עיקריים
שורה אחת: 3-5 מקורות מרכזיים ששימשו.

## מקורות מועדפים: Epoch Times, ISW, Jane's, OSINT, Times of Israel, Jerusalem Post, Al Arabiya
## מקורות להיזהר: BBC (אנטי-ישראלי), Al Jazeera (מוטה), CNN (שטחי) - ציין הטיה אם משתמש
## מקורות פרסיים = מדיה ממלכתית, ציין זאת
## חפש בכל השפות: אנגלית, ערבית, פרסית, עברית
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
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
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

    user_message = """תדריך מבצעי. מה המצב עכשיו?

חפש חדשות עדכניות ביותר על: מלחמת איראן, תקיפות, שיגורים, המצב בישראל, תגובות צבאיות.
חפש בכל השפות. תן מספרים. היה קצר וחד - רק עיקר."""

    print("Calling Claude API with web search...")
    client = anthropic.Anthropic(api_key=api_key)

    # Use web search tool for real-time news, with retry on rate limit
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 15
                }],
                messages=[{"role": "user", "content": user_message}]
            )
            break
        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

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
    safe_summary = scan['summary'][:80].encode('ascii', 'replace').decode('ascii')
    print(f"Summary: {safe_summary}...")
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
