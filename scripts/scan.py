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
MAX_TOKENS = 8192
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

    system_prompt = f"""אתה אנליסט מודיעין צבאי-אסטרטגי ישראלי ברמה הגבוהה ביותר. רקע של שנים בקהילת המודיעין.
הזמן הנוכחי: {time_str} (שעון ישראל)

## המשימה:
סקירה מבצעית מקיפה ומדויקת של הסיטואציה הביטחונית סביב איראן והמזרח התיכון. חפש חדשות עדכניות ביותר ובצע ניתוח מעמיק.

## מבנה הסקירה (חובה לכתוב בדיוק בפורמט הזה):

### 🔴 מצב מבצעי נוכחי
**התקפות על איראן:**
- מה הותקף (מתקנים, בסיסים, תשתיות - פרט שמות ומיקומים)
- על ידי מי (ישראל/ארה"ב/קואליציה - פרט)
- באיזה אמצעי לחימה (טילים, מטוסים, סוג תחמושת, מספר גלים)
- סדר גודל הכוחות: כמה מטוסים, כמה טילים, כמה גלים, משך המבצע
- נזקים מאומתים vs. נטענים (הפרד בבירור!)

**תגובת איראן:**
- תגובה רשמית (הצהרות מנהיגים, משמרות המהפכה)
- תגובה צבאית (שיגורים, הפעלת שלוחים, הכנות)
- מצב ההגנה האווירית האיראנית

**התקפות איראניות בעולם:**
- שיגורים לכיוון ישראל (טילים, מל"טים, שיוט - מספרים וסוגים)
- הפעלת חיזבאללה / חות'ים / מיליציות בעיראק ובסוריה
- פעולות טרור או סייבר שמיוחסות לאיראן
- תקיפות נגד בסיסים אמריקאים באזור

### 🇮🇱 המצב בישראל
- מצב הכוננות והעורף (אזעקות, יירוטים, נפגעים)
- פעילות צה"ל (הצהרות דובר, תמרונים, גיוס מילואים)
- הצהרות מדיניות (ראש ממשלה, שר ביטחון)
- מצב הגבולות (צפון/דרום/מזרח)
- השפעה על החיים האזרחיים

### 🌍 זירה בינלאומית
- עמדת ארה"ב (הצהרות, פריסת כוחות, מעורבות)
- עמדות מעצמות (רוסיה, סין, אירופה)
- האו"ם ומוסדות בינלאומיים
- תנועות צבאיות באזור (נושאות מטוסים, בסיסים)

### 📊 תרחישים - שעה קרובה
הצג 3 תרחישים מדורגים:
**תרחיש א' (X% הסתברות): [שם התרחיש]**
- מה צפוי לקרות
- סימנים שמחזקים תרחיש זה
- מה זה אומר עבורנו

**תרחיש ב' (Y% הסתברות): [שם התרחיש]**
- ...

**תרחיש ג' (Z% הסתברות): [שם התרחיש]**
- ...

### 📈 תרחישי המשך - 24-72 שעות
הצג 2-3 תרחישי המשך:
**תרחיש המשך 1 (X%): [שם]**
- התפתחות צפויה
- נקודות מפנה אפשריות
- משמעות אסטרטגית לישראל

**תרחיש המשך 2 (Y%): [שם]**
- ...

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
7. כתוב בעברית, קצר אבל מקיף, בנקודות
8. מספרים! תמיד תן מספרים - כמה טילים, כמה מטוסים, כמה נפגעים, כמה יירוטים
9. בסוף: "מקורות עיקריים" - רשימת המקורות ששימשו עם דירוג אמינות

## שפות חיפוש:
- אנגלית: Iran attack, Iran strike, Iran military, Iran war, Israel Iran, Iran retaliation, Houthi attack, Hezbollah
- ערבית: ايران هجوم, ايران حرب, ايران ضربة, حزب الله, الحوثي
- פרסית: ایران حمله, ایران جنگ, سپاه پاسداران
- עברית: איראן תקיפה, איראן מלחמה, צה"ל איראן, כיפת ברזל, חיזבאללה, חות'ים
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

    user_message = """בצע סקירה מבצעית מקיפה. מה המצב הביטחוני עכשיו?

חפש מידע עדכני ביותר על:
1. התקפות על איראן - מה הותקף, על ידי מי, באיזה נשק, סדר גודל כוחות (מספר מטוסים, טילים, גלים)
2. התקפות של איראן בעולם - שיגורים לישראל, הפעלת חיזבאללה/חות'ים/מיליציות, תקיפות בסיסים אמריקאים
3. המצב בישראל - אזעקות, יירוטים, נפגעים, מצב העורף, הצהרות צה"ל ומדיניות
4. תגובת איראן - רשמית ולא רשמית, מצב ההגנה האווירית
5. הזירה הבינלאומית - ארה"ב, רוסיה, סין, תנועות צבאיות באזור
6. תרחישים - 3 תרחישים מדורגים לשעה קרובה עם אחוזי הסתברות, ו-2-3 תרחישי המשך ל-24-72 שעות

חפש בכל השפות: אנגלית, ערבית, פרסית, עברית.
תן מספרים! כמה טילים, כמה מטוסים, כמה נפגעים.
סקירה מקיפה עם מקורות ורמת אמינות ליד כל פריט מידע."""

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
            "max_uses": 20
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
