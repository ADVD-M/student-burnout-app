

import random
from typing import Optional

# ── Exercise Libraries ────────────────────────────────────────────────────────

BREATHING_EXERCISES = [
    (
        "🌬️ **Box Breathing (4-4-4-4)**",
        "Breathe in for 4 seconds → Hold for 4 seconds → Breathe out for 4 seconds → Hold for 4 seconds. "
        "Repeat 4–6 times. This activates your parasympathetic nervous system and quickly reduces anxiety.",
    ),
    (
        "🌿 **4-7-8 Breathing**",
        "Inhale through your nose for 4 seconds → Hold for 7 seconds → Exhale through your mouth for 8 seconds. "
        "Repeat 3–4 times. Excellent for calming panic and promoting sleep.",
    ),
    (
        "🫁 **Diaphragmatic Breathing**",
        "Place one hand on your chest and one on your belly. Breathe deeply so only your belly rises. "
        "Inhale for 4 seconds, exhale for 6 seconds. Do this for 5 minutes to lower cortisol levels.",
    ),
    (
        "💨 **Alternate Nostril Breathing**",
        "Close your right nostril with your thumb, inhale left (4 sec). Close left nostril, exhale right (4 sec). "
        "Repeat swapping sides for 5 minutes. Great for mental clarity before studying.",
    ),
]

JOURNALING_PROMPTS = [
    (
        "📓 **Gratitude Journaling**",
        "Set a timer for 5 minutes. Write 3 things you're grateful for today — they can be tiny (a hot drink, sunshine, "
        "a message from a friend). Research shows this rewires your brain toward positivity over time.",
    ),
    (
        "✍️ **Brain Dump Exercise**",
        "Open a blank page and write everything on your mind for 10 minutes without stopping. Don't judge your thoughts. "
        "This clears mental clutter and is especially useful before studying or sleeping.",
    ),
    (
        "🔍 **Cognitive Reframe Journal**",
        "Write a stressful thought you're having. Then answer: "
        "**1)** Is this thought 100% true? **2)** What's the best realistic outcome? **3)** What would I tell a friend in this situation? "
        "This is a core CBT technique.",
    ),
    (
        "🎯 **Daily Intentions Journal**",
        "Each morning, write: **1)** One thing I will accomplish today. **2)** One person I'll be kind to. "
        "**3)** One thing I'll do for my wellbeing. This gives your day purpose and structure.",
    ),
]

FOCUS_ROUTINES = [
    (
        "⏱️ **Pomodoro Technique**",
        "Work for **25 minutes** with full focus, then take a **5-minute break**. After 4 rounds, take a longer 20-minute break. "
        "This prevents mental fatigue and maintains productivity throughout study sessions.",
    ),
    (
        "🌊 **Body Scan Before Studying**",
        "Sit comfortably. Slowly scan from head to toe noticing any tension. Consciously relax each area. "
        "Takes 5–7 minutes and significantly improves concentration by releasing physical distractions.",
    ),
    (
        "📵 **Digital Detox Blocks**",
        "Put your phone in another room (or use Focus Mode) during study blocks. "
        "Studies show phone proximity alone reduces cognitive capacity, even when it's silent.",
    ),
    (
        "🎵 **Focus Music Protocol**",
        "Listen to binaural beats (40Hz gamma or alpha waves) or lo-fi instrumental music during study. "
        "Avoid music with lyrics. Apps like Brain.fm or YouTube's study music channels work well.",
    ),
]

STRESS_MANAGEMENT = [
    (
        "🚶 **5-Minute Mindful Walk**",
        "Step away from your desk. Walk slowly and notice 5 things you can see, 4 you can touch, "
        "3 you can hear, 2 you can smell, 1 you can taste. This grounds you in the present moment.",
    ),
    (
        "💪 **Progressive Muscle Relaxation**",
        "Starting from your feet, tense each muscle group tightly for 5 seconds, then release for 30 seconds. "
        "Work up through legs, abdomen, arms, shoulders, and face. Full body release takes ~10 minutes.",
    ),
    (
        "☀️ **Movement Break**",
        "Do 10 jumping jacks, 10 arm circles, and 10 shoulder rolls. Physical movement releases endorphins "
        "and breaks the cortisol cycle caused by long sitting periods.",
    ),
    (
        "🎨 **Creative Micro-break**",
        "Draw, doodle, color, or listen to one song you love. Creative activities engage different brain regions, "
        "giving your analytical mind a genuine rest. Even 5 minutes is effective.",
    ),
]

SLEEP_HYGIENE = [
    (
        "🌙 **Wind-Down Routine**",
        "1 hour before bed: dim lights, stop screens, avoid caffeine. Try: light stretching → warm shower → "
        "10 minutes of reading (physical book). Signal your brain it's time to sleep.",
    ),
    (
        "🛏️ **Sleep Scheduling**",
        "Go to bed and wake up at the same time every day — including weekends. "
        "This anchors your circadian rhythm. Even if you didn't sleep well, get up at your set time.",
    ),
    (
        "🧊 **Cold Room, Dark Room**",
        "Keep your bedroom at 65–68°F (18–20°C) and use blackout curtains or a sleep mask. "
        "Your core body temperature needs to drop 1–2°F to initiate deep sleep.",
    ),
]

CRISIS_RESOURCES = """
🆘 **If you're in distress or having thoughts of harming yourself, please reach out:**
- **iCall (India)**: 9152987821
- **Vandrevala Foundation**: 1860-2662-345 (24/7)
- **AASRA**: 91-22-27546669
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
"""

# ── Intent Detection ──────────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "breathing": ["breath", "breathing", "anxious", "anxiety", "panic", "calm", "inhale", "exhale", "hyperventilat"],
    "journaling": ["journal", "write", "writing", "thoughts", "express", "feelings", "diary", "pen"],
    "focus": ["focus", "concentrate", "study", "distracted", "productivity", "procrastinate", "attention", "assignment"],
    "stress": ["stress", "overwhelm", "pressure", "tense", "tension", "tired", "exhausted", "burnout", "cant cope"],
    "sleep": ["sleep", "insomnia", "awake", "rest", "bed", "night", "dream", "fatigue", "sleepy", "nap"],
    "crisis": ["suicide", "self harm", "hurt myself", "don't want to live", "hopeless", "end it", "no point"],
}


def detect_intent(message: str) -> Optional[str]:
    """
    Detect the most likely intent from the user's message.

    Args:
        message: Raw user message string.

    Returns:
        Intent string or None if no match.
    """
    msg_lower = message.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            return intent
    return None



GREETINGS = [
    "Hello! I'm your mental wellness companion 🌱. I'm here to share exercises and techniques to help you manage stress and academic pressure. What's on your mind today?",
    "Hi there! 👋 I'm here to support your wellbeing journey. I can suggest breathing exercises, journaling prompts, focus routines, and more. How are you feeling right now?",
    "Welcome! 🌿 Whether you're feeling overwhelmed, unfocused, or just need a little reset — I've got evidence-based techniques to help. What would you like support with?",
]

FALLBACK_BY_RISK = {
    "High": [
        "It sounds like you're going through a tough time. Let me share a quick exercise that can help right now.",
        "I hear you. High stress periods can feel overwhelming. Here's something that can bring some immediate relief:",
        "You're not alone in feeling this way. Many students go through intense pressure. Let's try something together:",
    ],
    "Medium": [
        "It's good that you're paying attention to how you feel. Here's a helpful technique for maintaining balance:",
        "Staying proactive about your wellbeing is a great habit. Here's something that might help:",
        "Let's work on keeping your mental energy steady. Try this:",
    ],
    "Low": [
        "Great that you're checking in with yourself! Here's a technique to keep your wellbeing strong:",
        "Maintaining good habits even when things feel okay is key. Here's something to add to your routine:",
        "You're doing well — let's keep it that way! Here's a simple practice for sustained wellness:",
    ],
    "Unknown": [
        "I'm here to help! Here's something you might find useful:",
        "Let me share a technique that many students find helpful:",
    ],
}

INTENT_TO_LIBRARY = {
    "breathing": BREATHING_EXERCISES,
    "journaling": JOURNALING_PROMPTS,
    "focus": FOCUS_ROUTINES,
    "stress": STRESS_MANAGEMENT,
    "sleep": SLEEP_HYGIENE,
}

DEFAULT_BY_RISK = {
    "High": BREATHING_EXERCISES + STRESS_MANAGEMENT,
    "Medium": FOCUS_ROUTINES + JOURNALING_PROMPTS,
    "Low": JOURNALING_PROMPTS + FOCUS_ROUTINES,
    "Unknown": STRESS_MANAGEMENT,
}

DISCLAIMER = (
    "\n\n---\n*⚠️ This is for general wellness information only and is not a substitute for professional "
    "mental health advice. If you're struggling, please speak to a counsellor or mental health professional.*"
)


def get_response(user_message: str, risk_level: str = "Unknown") -> str:
    """
    Generate a contextual mental health exercise response based on
    the user's message and their predicted burnout risk level.

    Args:
        user_message: The user's chat message.
        risk_level: Predicted risk level ('Low', 'Medium', 'High', or 'Unknown').

    Returns:
        Formatted response string with exercises and a safety disclaimer.
    """
    msg_lower = user_message.lower().strip()

    # ── Crisis Detection (highest priority) ──────────────────────────────────
    if detect_intent(user_message) == "crisis":
        return (
            "I'm really concerned about what you've shared. Please know that you matter and help is available. 💙\n\n"
            + CRISIS_RESOURCES
            + DISCLAIMER
        )

    # ── Greetings ─────────────────────────────────────────────────────────────
    greeting_words = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hiya", "howdy"}
    if any(w in msg_lower for w in greeting_words) and len(msg_lower.split()) <= 4:
        return random.choice(GREETINGS)

    # ── Intent-based response ─────────────────────────────────────────────────
    intent = detect_intent(user_message)
    exercise_library = INTENT_TO_LIBRARY.get(intent, DEFAULT_BY_RISK.get(risk_level, STRESS_MANAGEMENT))

    # Pick a random exercise from the relevant library
    title, description = random.choice(exercise_library)

    # Opener based on risk level
    opener = random.choice(FALLBACK_BY_RISK.get(risk_level, FALLBACK_BY_RISK["Unknown"]))

    # Build response
    response_parts = [opener, f"\n\n### {title}\n\n{description}"]

    # Add a bonus tip for High risk users
    if risk_level == "High":
        bonus_lib = [ex for ex in STRESS_MANAGEMENT + BREATHING_EXERCISES if ex[0] != title]
        if bonus_lib:
            b_title, b_desc = random.choice(bonus_lib)
            response_parts.append(
                f"\n\n💡 **Bonus exercise for you:**\n\n### {b_title}\n\n{b_desc}"
            )

    response_parts.append(DISCLAIMER)
    return "".join(response_parts)


def get_welcome_message(risk_level: str = "Unknown") -> str:
    """
    Generate a personalised welcome message based on the user's risk level.

    Args:
        risk_level: 'Low', 'Medium', 'High', or 'Unknown'.

    Returns:
        Welcome message string.
    """
    messages = {
        "High": (
            "🔴 **Your assessment suggests a High burnout risk.** I'm here to help you work through this. "
            "You can ask me about breathing exercises, stress relief, sleep tips, journaling, or anything else on your mind. "
            "You don't have to navigate this alone. What would you like help with first?"
        ),
        "Medium": (
            "🟡 **Your assessment suggests a Medium burnout risk.** Let's work together on building some healthy habits "
            "before stress builds further. I can suggest focus techniques, journaling, or relaxation exercises. "
            "What sounds most helpful right now?"
        ),
        "Low": (
            "🟢 **Your assessment suggests a Low burnout risk.** Great job maintaining your wellbeing! "
            "I can still share useful techniques for focus, journaling, and maintaining this healthy balance. "
            "What would you like to explore?"
        ),
        "Unknown": (
            "👋 **Welcome to your AI Wellness Companion!** I can share evidence-based techniques for managing "
            "stress, improving focus, better sleep, and emotional processing. What's on your mind?"
        ),
    }
    return messages.get(risk_level, messages["Unknown"])
