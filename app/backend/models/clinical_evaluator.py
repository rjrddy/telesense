"""Rule-based clinical evaluation of a session transcript.

Builds the data behind the Evaluations, Protective Factors and Missing Questions
panels from the transcript alone: protective factors are found by phrase matching,
and critical intake topics that never come up are reported as missing questions.
This is a screening aid, not a clinical assessment.
"""

import re
from typing import Dict, List, Optional


# Protective factors: category -> (phrases, rationale shown in the panel)
PROTECTIVE_FACTORS = {
    "Social support": (
        ["my family", "my wife", "my husband", "my partner", "my kids", "my children", "my mom",
         "my dad", "my friends", "my friend", "support me", "supportive", "people who care",
         "someone to talk to"],
        "Mentions supportive relationships."),
    "Treatment engagement": (
        ["my therapist", "therapy", "counseling", "counselor", "my psychiatrist", "my doctor",
         "my medication", "taking my meds", "support group", "appointment"],
        "Engaged with treatment or care."),
    "Coping skills": (
        ["meditate", "meditation", "exercise", "go for a walk", "going for walks", "journaling",
         "breathing", "yoga", "coping", "calm myself", "relax"],
        "Describes active coping strategies."),
    "Future orientation": (
        ["looking forward", "my goals", "plans for", "in the future", "next year", "hope to",
         "want to get better", "getting better", "planning to"],
        "Expresses future plans or hope."),
    "Meaning and purpose": (
        ["my job", "my work", "volunteer", "my faith", "church", "pray", "my pet", "my dog",
         "my cat", "reason to live", "responsibility"],
        "Has sources of meaning, purpose or responsibility."),
}

# Critical intake topics: question shown when missing -> phrases that count as covered
CRITICAL_TOPICS = {
    "Ask directly about suicidal ideation, intent or plan (safety)": [
        "suicid", "kill myself", "end my life", "want to die", "hurt yourself", "harm yourself",
        "thoughts of death", "better off dead", "self harm", "self-harm"],
    "Ask about homicidal ideation or thoughts of harming others (safety)": [
        "hurt someone", "harm others", "hurt anyone", "homicid", "violent", "kill someone"],
    "Ask about access to firearms or other lethal means (safety)": [
        "gun", "firearm", "weapon", "pills at home", "lethal"],
    "Ask about sleep (onset, maintenance, changes)": [
        "sleep", "insomnia", "nightmare", "wake up", "tired", "rest"],
    "Ask about appetite and weight changes (depression screen)": [
        "appetite", "eating", "weight", "hungry"],
    "Ask about substance use (alcohol, drugs, cannabis)": [
        "alcohol", "drink", "drinking", "drunk", "drugs", "weed", "marijuana", "cannabis",
        "cocaine", "substance", "smoke"],
    "Screen for psychosis or mania (hallucinations, racing thoughts)": [
        "hear voices", "hearing voices", "see things", "hallucinat", "paranoi", "racing thoughts",
        "manic", "mania", "don't need sleep"],
    "Ask about trauma history": [
        "trauma", "abuse", "assault", "ptsd", "flashback", "accident"],
    "Assess mood and anxiety symptoms (depression, anxiety)": [
        "depress", "sad", "down", "hopeless", "anxious", "anxiety", "worry", "panic", "nervous"],
    "Ask about current medications and adherence": [
        "medication", "meds", "prescri", "antidepressant", "pills"],
    "Assess social support and functional impact (work, relationships)": [
        "work", "job", "school", "friends", "family", "relationship", "lonely", "isolat"],
}


def _contains(text: str, phrase: str) -> bool:
    """
    Check whether a phrase occurs in text as a word or word prefix.
    Input:
        text - lowercased text to search
        phrase - lowercased phrase; a phrase without a trailing space also matches word prefixes

    Output:
        True if the phrase occurs, otherwise False
    """
    return re.search(r"\b" + re.escape(phrase), text) is not None


def _timestamp(seconds: float) -> str:
    """
    Format seconds as MM:SS.
    Input:
        seconds - time in seconds

    Output:
        Time string in MM:SS format
    """
    minutes, secs = divmod(int(max(seconds, 0)), 60)
    return f"{minutes:02d}:{secs:02d}"


def _matching_sentence(text: str, phrase: str) -> str:
    """
    Pick the sentence or clause of a segment that contains the matched phrase.
    Input:
        text - segment text
        phrase - lowercased phrase that matched

    Output:
        The first sentence/clause containing the phrase (the whole text if none is found)
    """
    for part in re.split(r"(?<=[.!?,;])\s+", text):
        if _contains(part.lower(), phrase):
            return part.strip(" ,;")
    return text


def find_protective_factors(segments: List[Dict]) -> List[Dict]:
    """
    Find protective factors mentioned in the transcript.
    Input:
        segments - transcript segments with start_time, end_time and text

    Output:
        List of protective factors with time_s, timestamp, category, evidence and rationale,
        at most one per category (its first mention)
    """
    factors = []
    for category, (phrases, rationale) in PROTECTIVE_FACTORS.items():
        for seg in segments:
            text = (seg.get("text") or "").strip()
            lowered = text.lower()
            matched = next((p for p in phrases if _contains(lowered, p)), None)
            if matched:
                start = float(seg.get("start_time", 0.0))
                factors.append({
                    "time_s": start,
                    "timestamp": _timestamp(start),
                    "category": category,
                    "evidence": _matching_sentence(text, matched),
                    "rationale": f"{category}: {rationale} (matched \"{matched}\")",
                })
                break
    return sorted(factors, key=lambda f: f["time_s"])


def find_missing_questions(segments: List[Dict]) -> List[str]:
    """
    List critical intake topics that never come up in the transcript.
    Input:
        segments - transcript segments with text

    Output:
        List of suggested questions for topics with no matching phrase
    """
    full_text = " ".join((seg.get("text") or "") for seg in segments).lower()
    return [question for question, phrases in CRITICAL_TOPICS.items()
            if not any(_contains(full_text, p) for p in phrases)]


def build_evaluations(segments: List[Dict], flags: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Build the transcript evaluation shown in the Evaluations, Protective Factors and
    Missing Questions panels.
    Input:
        segments - transcript segments with start_time, end_time and text
        flags - combined flags from the analysis, summarized in the notes (optional)

    Output:
        List with one evaluation dictionary (name, basis, method, notes, flags,
        protective_factors, missing_critical_questions); empty if there is no transcript
    """
    if not segments or not any((s.get("text") or "").strip() for s in segments):
        return []

    flags = flags or []
    protective = find_protective_factors(segments)
    missing = find_missing_questions(segments)
    high = sum(1 for f in flags if f.get("severity") == "high")

    notes = [
        f"{len(flags)} flags raised by the analysis ({high} high risk).",
        f"{len(protective)} protective factor categories mentioned; "
        f"{len(missing)} of {len(CRITICAL_TOPICS)} critical intake topics not discussed.",
        "Generated by phrase matching on the transcript; review before clinical use.",
    ]

    return [{
        "name": "text",
        "basis": "transcript",
        "method": "Rule-based phrase matching for protective factors and coverage of critical intake topics.",
        "notes": notes,
        "flags": [],  # flags are already shown in the flag monitors
        "protective_factors": protective,
        "missing_critical_questions": missing,
    }]
