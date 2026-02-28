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
MAX_TOKENS = 5500
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

    # Previous scan context (last 1, generous context for continuity)
    prev_context = ""
    if history:
        last = history[0]
        t = last.get("time_str", "")
        c = last.get("content", "")[:1500]
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

    system_prompt = f"""אתה עורך מודיעין בכיר. תדריך מבצעי חד, עובדתי ומלא.

הזמן: {time_str} (ישראל)

## עקרונות:
1. אנליסט, לא עיתונאי. שורה תחתונה + עובדות מרכזיות.
2. טקסט רציף, מקורות בתוך הטקסט. סמן: ✅ מאומת | ⚠️ מקור בודד | ❓ שמועה.
3. לא ציטוטים אלא אם משנים תמונה. לא טריוויאליה (בתי חולים, PR, עיריות).
4. *** קיצור = מפתח. אמור את אותו דבר בפחות מילים. שלב מידע קשור יחד. ***

## *** כלל קריטי - פורמט: ***
- המילה הראשונה = "###". אסור הקדמות/באנרים/הערות לפני הכותרת הראשונה.
- *** כל 6 הסעיפים חייבים להופיע מלאים. סעיף שנקטע = כישלון. ***
- *** אם אתה מרגיש שנגמר המקום - קצר סעיפים קודמים, לעולם לא את התרחישים. ***

## מבנה חובה + תקציב (משפטים):

### 🔴 מצב מבצעי נוכחי
4-5 משפטים: מתי התחיל, מי-מה-באילו אמצעים, סדר גודל, נפגעים כלליים.

### 🔥 אירועים חמים
דגש על חדש *עכשיו*. 3 פסקאות קצרות (2-3 משפטים כל אחת):
- גל תקיפות אחרון / שיגורים חדשים
- חיסולים + סטטוס חמינאי (תמצית - לא כל שמועה)
- מצר הורמוז / חות'ים / אזורי (תמצית)
בסוף - חובה:
**תגובת העולם:** 2-4 משפטים בולטים על תגובות בינלאומיות (מעצמות, מדינות ערב, או"ם). אם אין חדש: "ללא התפתחות בינלאומית חדשה."

### 🇮🇱 המצב בישראל
2 פסקאות (2-3 משפטים כל אחת): אזעקות, יירוטים, נפגעים (מספרים+מיקום+חומרה), הנחיות, מרחב אווירי.
*** שמור נפגעים מסקירה קודמת + הוסף חדשים. ***

### 📊 תרחיש - שעה קרובה
*** חובה מלא - לא לקצר. *** 1-2 פסקאות: מה צפוי עכשיו, מאיפה, נקודות שבר.

### 📈 תרחיש - התפתחות המלחמה
*** חובה מלא - לא לקצר. זה הסעיף הכי חשוב ביחד עם שעה קרובה. *** 2 פסקאות: לאן המלחמה הולכת (ימים-שבועות). הסלמה, מכפילי לחץ, תחזיות.

### מקורות עיקריים
שורה אחת: 3-5 מקורות.

## חלוקת מקום (בקירוב):
מצב מבצעי ~8% | אירועים חמים ~27% | ישראל ~17% | שעה קרובה ~17% | התפתחות ~25% | מקורות ~3%
*** התרחישים ביחד = 42%. חרוג בסעיפים קודמים → קצר אותם, לא את התרחישים. ***

## למידה:
- אל תאבד מידע מסקירה קודמת (נפגעים, חיסולים). הוסף עליו, אל תחזור על כולו.

## מקורות: Epoch Times, ISW, Jane's, OSINT, Times of Israel, Jerusalem Post, Al Arabiya, Reuters, AP, i24news
## מקורות מזה"ת (חפש פעיל): Al Arabiya, Al Hadath, Sky News Arabia, TRT World, Anadolu, The National (UAE), Gulf News, Arab News. עובדות בלבד - הטיה מובנית.
## להיזהר: BBC (אנטי-ישראלי), Al Jazeera (מוטה), CNN (שטחי) - ציין הטיה
## מקורות פרסיים = מדיה ממלכתית
## *** לא Ynet. לא N12/חדשות 12/Channel 12. ***
## חפש בכל השפות: אנגלית, ערבית, פרסית, עברית, תורכית
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
    "i24news": ["i24", "i24news", "i24NEWS"],
    "Walla": ["Walla", "וואלה"],
    "Kan News": ["כאן חדשות", "Kan News", "כאן 11"],
    "Channel 13": ["חדשות 13", "Channel 13"],
    "OSINT Analysts": ["OSINT", "osint"],
    "Telegram": ["טלגרם", "Telegram", "telegram"],
    "X/Twitter": ["Twitter", "X.com"],
    "BBC": ["BBC"],
    "CNN": ["CNN"],
    "Aurora Intel": ["Aurora Intel"],
    "OSINTdefender": ["OSINTdefender"],
    "Al Hadath": ["Al Hadath", "الحدث"],
    "Sky News Arabia": ["Sky News Arabia"],
    "Asharq Al-Awsat": ["Asharq Al-Awsat", "الشرق الأوسط"],
    "TRT World": ["TRT World", "TRT"],
    "Anadolu Agency": ["Anadolu", "AA"],
    "The National (UAE)": ["The National"],
    "Gulf News": ["Gulf News"],
    "Arab News": ["Arab News"],
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

    user_message = """תדריך מבצעי מפורט. מה המצב עכשיו?

חפש חדשות עדכניות ביותר על: מלחמת איראן, תקיפות, שיגורים, המצב בישראל, תגובות צבאיות.
חפש בכל השפות (אנגלית, ערבית, פרסית, עברית, תורכית). תן מספרים מדויקים. היה מפורט - כלול כל מידע עובדתי חשוב.
דגש: מה חדש מאז הסקירה האחרונה? מה השתנה? מה ההתפתחויות החמות ביותר ברגע זה?
חשוב: הקפד לכתוב את כל הסעיפים במלואם, כולל תרחיש שעה קרובה ותרחיש התפתחות המלחמה."""

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
                    "max_uses": 20
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

    # Strip everything before the first "###" - removes AI preambles/thinking
    import re
    first_section = re.search(r'^###\s', content, re.MULTILINE)
    if first_section and first_section.start() > 0:
        stripped = content[:first_section.start()].strip()
        if stripped:
            print(f"Stripped preamble ({len(stripped)} chars): {stripped[:80]}...")
        content = content[first_section.start():]

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

    # Prepend to history, keep only 3 (sampled, not necessarily consecutive)
    history.insert(0, scan)
    if len(history) > 3:
        # Keep latest, plus 2 evenly spaced from the rest
        rest = history[1:]
        step = max(1, len(rest) // 2)
        sampled = [rest[0], rest[min(step, len(rest)-1)]]
        history = [history[0]] + sampled
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
