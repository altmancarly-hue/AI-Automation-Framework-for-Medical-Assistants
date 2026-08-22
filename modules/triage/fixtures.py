"""Synthetic triage calls. No real patient data, no real audio, ever.

Fifteen calls, chosen for the things that are hard rather than the things that
are easy to write: unclear audio, a transcript with no speaker labels, an
escalation that needs a named physician, a follow-up timeframe the parser will
not recognise, a caller name that is never spoken aloud, and a model response
that tries to add advice nobody gave.

Each case carries BOTH sides of the pipeline: the transcript a transcriber would
have produced, and the JSON a model would have returned from it. That makes the
whole path -- structuring, grounding, rendering, edit-distance -- runnable
deterministically on a laptop with no GPU and no model, which is what the
regression suite README I-04 asks for actually needs to exist.

`expand_to(n)` tiles the set up to the fifty calls the README's model-version
control specifies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .capture import TranscriptSegment

__all__ = ["TriageCase", "CASES", "by_name", "expand_to", "response_for", "CALL_TIME"]

CALL_TIME = datetime(2026, 8, 22, 14, 5, tzinfo=timezone.utc)


def _blank_extraction() -> dict[str, Any]:
    return {
        "caller": {"name": None, "relationship_to_patient": None},
        "chief_complaint": "",
        "symptom_onset": None,
        "symptoms_reported_present": [],
        "symptoms_explicitly_denied": [],
        "relevant_history_mentioned": [],
        "medications_mentioned": [],
        "advice_given_by_ma": [],
        "safety_net_instructions_given": [],
        "followup_discussed": False,
        "followup_timeframe": None,
        "transcript_gaps": [],
    }


@dataclass
class TriageCase:
    name: str
    description: str
    patient_id: str
    family_id: str
    patient_label: str
    age_months: int
    segments: list[TranscriptSegment]
    extraction: dict[str, Any]
    protocol_id: str
    disposition_id: str
    supervising_professional_id: str | None = None
    diarized: bool = True
    #: What a reviewer should end up seeing. Used by tests as
    #: documentation-with-teeth and by the regression harness as the reference.
    expect: dict[str, Any] = field(default_factory=dict)

    @property
    def transcript_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "confidence": s.confidence,
            }
            for s in self.segments
        ]

    @property
    def model_response(self) -> str:
        return json.dumps(self.extraction)


def _seg(index: int, speaker: str | None, text: str, confidence: float = 0.95) -> TranscriptSegment:
    return TranscriptSegment(index * 6.0, index * 6.0 + 6.0, text, speaker, confidence)


def _case(
    name: str,
    description: str,
    *,
    label: str,
    age_months: int,
    lines: Sequence[tuple[str | None, str] | tuple[str | None, str, float]],
    extraction: Mapping[str, Any],
    protocol_id: str,
    disposition_id: str,
    supervising_professional_id: str | None = None,
    diarized: bool = True,
    expect: Mapping[str, Any] | None = None,
) -> TriageCase:
    segments = []
    for index, line in enumerate(lines):
        speaker, text = line[0], line[1]
        confidence = line[2] if len(line) > 2 else 0.95
        segments.append(_seg(index, speaker, text, confidence))
    data = _blank_extraction()
    data.update(extraction)
    return TriageCase(
        name=name,
        description=description,
        patient_id=f"p_{name}",
        family_id=f"fam_{name}",
        patient_label=label,
        age_months=age_months,
        segments=segments,
        extraction=data,
        protocol_id=protocol_id,
        disposition_id=disposition_id,
        supervising_professional_id=supervising_professional_id,
        diarized=diarized,
        expect=dict(expect or {}),
    )


def _build() -> list[TriageCase]:
    cases: list[TriageCase] = []

    cases.append(_case(
        "toddler_fever_home_care",
        "Straightforward fever call resolved with home care and safety-net advice.",
        label="Mila Torres (2 y 4 mo)",
        age_months=28,
        lines=[
            ("ma", "Hi, this is Jess at North Suburban Pediatrics returning your call about Mila."),
            ("caller", "Yes, hi, this is her mom Renata. She's had a fever since last night."),
            ("ma", "Okay. Have you taken her temperature?"),
            ("caller", "It was a hundred and one point four about an hour ago, under the arm."),
            ("ma", "Is she drinking? Any vomiting or diarrhea?"),
            ("caller", "She's drinking okay. No vomiting, no diarrhea. She's just clingy."),
            ("ma", "Any rash, any trouble breathing?"),
            ("caller", "No rash. Breathing is fine."),
            ("ma", "Has she had anything for the fever?"),
            ("caller", "I gave her children's Tylenol around noon."),
            ("ma", "That's fine. Keep offering fluids and you can repeat the Tylenol every four to six hours."),
            ("ma", "Call us back if the fever goes above a hundred and four, if she stops drinking, or if she seems much less responsive."),
            ("caller", "Okay. Should I call tomorrow either way?"),
            ("ma", "Yes, give us a call tomorrow to let us know how she's doing."),
        ],
        extraction={
            "caller": {"name": "Renata", "relationship_to_patient": "mother"},
            "chief_complaint": "fever since last night",
            "symptom_onset": "last night",
            "symptoms_reported_present": ["fever 101.4 under the arm", "clingy"],
            "symptoms_explicitly_denied": ["vomiting", "diarrhea", "rash", "trouble breathing"],
            "medications_mentioned": ["children's Tylenol around noon"],
            "advice_given_by_ma": [
                "keep offering fluids",
                "may repeat Tylenol every four to six hours",
            ],
            "safety_net_instructions_given": [
                "call back if the fever goes above 104",
                "call back if she stops drinking",
                "call back if she seems much less responsive",
            ],
            "followup_discussed": True,
            "followup_timeframe": "tomorrow",
        },
        protocol_id="PLACEHOLDER-FEVER",
        disposition_id="home_care",
        expect={"dropped": 0, "followup_task": True},
    ))

    cases.append(_case(
        "newborn_fever_ed_now",
        "Six-week-old with fever. Escalated; the record must name the physician.",
        label="Ezra Blank (6 wk)",
        age_months=1,
        lines=[
            ("ma", "This is Jess at North Suburban Pediatrics, is this Ezra's dad?"),
            ("caller", "Yes, this is Daniel. He feels really warm."),
            ("ma", "How old is Ezra again, and did you take a temperature?"),
            ("caller", "He's six weeks. Rectal was a hundred point nine."),
            ("ma", "Okay. I'm going to put you on a brief hold and get Dr. Ruiz."),
            ("ma", "Dr. Ruiz would like you to take Ezra to the emergency department now."),
            ("caller", "Right now? Okay. Lutheran General?"),
            ("ma", "Yes, Lutheran General. Go now, don't wait for a call back."),
        ],
        extraction={
            "caller": {"name": "Daniel", "relationship_to_patient": "father"},
            "chief_complaint": "fever in a six-week-old",
            "symptom_onset": None,
            "symptoms_reported_present": ["rectal temperature 100.9", "feels warm"],
            "advice_given_by_ma": [
                "take Ezra to the emergency department now",
                "go to Lutheran General",
            ],
            "safety_net_instructions_given": ["go now, do not wait for a call back"],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-NEWBORN-FEVER",
        disposition_id="ed_now",
        supervising_professional_id="dr_ruiz",
        expect={"requires_supervisor": True},
    ))

    cases.append(_case(
        "vomiting_see_today",
        "Vomiting with reduced wet diapers; appointment same day.",
        label="Nina Okonkwo (4 y)",
        age_months=48,
        lines=[
            ("ma", "Hi, calling back about Nina."),
            ("caller", "Hi, it's her grandmother, Adaeze. She's been throwing up since about six this morning."),
            ("ma", "How many times?"),
            ("caller", "Maybe seven times. She can't keep water down."),
            ("ma", "Any diarrhea, any belly pain?"),
            ("caller", "No diarrhea. She says her stomach hurts."),
            ("ma", "How many wet diapers or trips to the bathroom today?"),
            ("caller", "Only once since this morning."),
            ("ma", "I'd like to see her today. Can you come at three fifteen?"),
            ("caller", "Yes, we can."),
            ("ma", "In the meantime give small sips, a teaspoon every five minutes."),
            ("ma", "If she becomes very sleepy or you can't wake her easily, go to the emergency department."),
        ],
        extraction={
            "caller": {"name": "Adaeze", "relationship_to_patient": "grandmother"},
            "chief_complaint": "vomiting since this morning, unable to keep fluids down",
            "symptom_onset": "about six this morning",
            "symptoms_reported_present": [
                "vomiting approximately seven times",
                "stomach hurts",
                "only one void since morning",
            ],
            "symptoms_explicitly_denied": ["diarrhea"],
            "advice_given_by_ma": ["give small sips, a teaspoon every five minutes"],
            "safety_net_instructions_given": [
                "go to the emergency department if she becomes very sleepy or cannot be woken easily",
            ],
            "followup_discussed": True,
            "followup_timeframe": "today",
        },
        protocol_id="PLACEHOLDER-VOMITING",
        disposition_id="see_today",
    ))

    cases.append(_case(
        "unclear_audio_gaps",
        "Poor line. Two low-confidence spans must reach the note as gaps.",
        label="Theo Mbeki (7 y)",
        age_months=84,
        lines=[
            ("ma", "Hi, this is Jess returning your call about Theo."),
            ("caller", "He's had a cough for [inaudible] days", 0.31),
            ("ma", "Sorry, how many days?"),
            ("caller", "Four days. It's worse at night."),
            ("ma", "Any fever, any wheezing?"),
            ("caller", "No fever. Sometimes he [inaudible] after running", 0.28),
            ("ma", "Okay. Try a cool mist humidifier at night and honey if he's over one year."),
            ("ma", "Call back if he's working hard to breathe or the cough keeps him from sleeping."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "cough for four days, worse at night",
            "symptom_onset": "four days ago",
            "symptoms_reported_present": ["cough worse at night"],
            "symptoms_explicitly_denied": ["fever"],
            "advice_given_by_ma": [
                "cool mist humidifier at night",
                "honey if over one year of age",
            ],
            "safety_net_instructions_given": [
                "call back if he is working hard to breathe",
                "call back if the cough keeps him from sleeping",
            ],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-COUGH",
        disposition_id="home_care",
        expect={"gaps_min": 2},
    ))

    cases.append(_case(
        "undiarized_transcript",
        "No speaker labels available. The note must not guess who said what.",
        label="Ivy Castellanos (3 y)",
        age_months=36,
        lines=[
            (None, "Hi, calling about Ivy, she has a rash on her tummy."),
            (None, "When did it start?"),
            (None, "This morning. It's flat and pink, no fever."),
            (None, "Is she scratching, is she otherwise acting normally?"),
            (None, "She's fine, playing normally, not scratching much."),
            (None, "Let's have you send a photo through the portal and we'll take a look."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "rash on the abdomen",
            "symptom_onset": "this morning",
            "symptoms_reported_present": ["flat pink rash on the abdomen"],
            "symptoms_explicitly_denied": ["fever"],
            "advice_given_by_ma": [],
            "safety_net_instructions_given": [],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-RASH",
        disposition_id="see_tomorrow",
        diarized=False,
        expect={"diarized": False},
    ))

    cases.append(_case(
        "model_invents_advice",
        "The model adds ibuprofen advice nobody gave. Grounding must remove it.",
        label="Omar Haddad (5 y)",
        age_months=60,
        lines=[
            ("ma", "Hi, this is Jess about Omar's ear."),
            ("caller", "He's been pulling at his right ear since yesterday and crying at night."),
            ("ma", "Any fever, any drainage from the ear?"),
            ("caller", "No fever. No drainage."),
            ("ma", "Let's get him seen today. Can you come at four?"),
            ("caller", "Yes."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "right ear pain since yesterday",
            "symptom_onset": "yesterday",
            "symptoms_reported_present": ["pulling at right ear", "crying at night"],
            "symptoms_explicitly_denied": ["fever", "ear drainage"],
            "advice_given_by_ma": [
                "come in at four",
                # Never said by anyone. The grounding check must drop it.
                "give ibuprofen 10 mg per kilogram every six hours for pain",
            ],
            "safety_net_instructions_given": [],
            "followup_discussed": True,
            "followup_timeframe": "today",
        },
        protocol_id="PLACEHOLDER-EAR-PAIN",
        disposition_id="see_today",
        expect={
            "dropped_min": 1,
            # Declared, so the regression reference does not have to ask the
            # run under test what it removed.
            "expected_drops": {
                "advice_given_by_ma": [
                    "give ibuprofen 10 mg per kilogram every six hours for pain",
                ]
            },
        },
    ))

    cases.append(_case(
        "model_misattributes_advice",
        "Advice the CALLER spoke, credited to the MA. Diarized grounding catches it.",
        label="Sana Qureshi (6 y)",
        age_months=72,
        lines=[
            ("ma", "Hi, returning your call about Sana."),
            ("caller", "She bumped her head on the coffee table about an hour ago."),
            ("caller", "My sister said we should just put a bag of frozen peas on it and watch her."),
            ("ma", "Did she lose consciousness, is she vomiting, is she acting like herself?"),
            ("caller", "No blackout, no vomiting. She's acting normal."),
            ("ma", "I'm going to have Dr. Okafor speak with you."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": "parent"},
            "chief_complaint": "head bump about an hour ago",
            "symptom_onset": "about an hour ago",
            "symptoms_reported_present": ["bumped head on coffee table"],
            "symptoms_explicitly_denied": [
                "loss of consciousness",
                "vomiting",
                "behaving abnormally",
            ],
            "advice_given_by_ma": [
                # The caller's sister said this, not the MA.
                "put a bag of frozen peas on it and watch her",
            ],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-HEAD-INJURY",
        disposition_id="escalate_physician",
        supervising_professional_id="dr_okafor",
        expect={
            "dropped_min": 1,
            "expected_drops": {
                "advice_given_by_ma": [
                    "put a bag of frozen peas on it and watch her",
                ]
            },
        },
    ))

    cases.append(_case(
        "unrecognised_followup_timeframe",
        "Parent told to call 'in a little while'. No due date may be invented.",
        label="Rafi Nasser (18 mo)",
        age_months=18,
        lines=[
            ("ma", "Hi, calling about Rafi's cough."),
            ("caller", "It's a barky cough, started tonight."),
            ("ma", "Any noisy breathing when he's calm?"),
            ("caller", "Only when he's upset."),
            ("ma", "Try sitting with him in a steamy bathroom for fifteen minutes."),
            ("ma", "Give us a call back in a little while and let us know how he sounds."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "barky cough starting tonight",
            "symptom_onset": "tonight",
            "symptoms_reported_present": ["barky cough", "noisy breathing when upset"],
            "advice_given_by_ma": ["sit with him in a steamy bathroom for fifteen minutes"],
            "safety_net_instructions_given": [],
            "followup_discussed": True,
            "followup_timeframe": "in a little while",
        },
        protocol_id="PLACEHOLDER-COUGH",
        disposition_id="home_care",
        expect={"followup_without_due_date": True},
    ))

    cases.append(_case(
        "caller_name_never_spoken",
        "The model supplies a caller name that appears nowhere in the transcript.",
        label="Juno Petrov (9 y)",
        age_months=108,
        lines=[
            ("ma", "Hi, returning your call about Juno."),
            ("caller", "It's her mom. She has a sore throat and a low fever since yesterday."),
            ("ma", "Any rash, any trouble swallowing her own spit?"),
            ("caller", "No rash. She's swallowing fine."),
            ("ma", "Let's see her tomorrow morning. Nine fifteen?"),
            ("caller", "That works."),
        ],
        extraction={
            # Name invented, relationship spoken: the two halves of `caller` are
            # treated differently on purpose and this fixture exercises both.
            "caller": {"name": "Katarina Petrov", "relationship_to_patient": "mother"},
            "chief_complaint": "sore throat and low fever since yesterday",
            "symptom_onset": "yesterday",
            "symptoms_reported_present": ["sore throat", "low fever"],
            "symptoms_explicitly_denied": ["rash", "trouble swallowing saliva"],
            "advice_given_by_ma": [],
            "followup_discussed": True,
            "followup_timeframe": "tomorrow",
        },
        protocol_id="PLACEHOLDER-FEVER",
        disposition_id="see_tomorrow",
        expect={
            "caller_name_dropped": True,
            # Declared, so the regression suite scores the DROP rather than
            # scoring the fabrication as a correct extraction.
            "expected_drops": {"caller.name": ["Katarina Petrov"]},
        },
    ))

    cases.append(_case(
        "third_party_relationship_spoken",
        "The sitter calls and mentions the mother. The note claims the mother.",
        label="Ravi Anand (5 y)",
        age_months=60,
        lines=[
            ("ma", "North Suburban Pediatrics, this is Jess."),
            ("caller", "I'm watching Ravi this afternoon. His mom is at work until six."),
            ("ma", "Okay. What's going on with him?"),
            ("caller", "He's got a barky cough and he sounds hoarse."),
            ("ma", "Any trouble breathing, any noise when he breathes in?"),
            ("caller", "No trouble breathing. No noise."),
            ("ma", "Sit with him in a steamy bathroom for fifteen minutes."),
            ("ma", "Call back if he starts making a high-pitched noise breathing in."),
        ],
        extraction={
            # The word "mom" was said, but not by the caller about themselves.
            # A note documenting the informant as the mother is a wrong-informant
            # record: it is who the practice will say it took the history from.
            "caller": {"name": None, "relationship_to_patient": "mother"},
            "chief_complaint": "barky cough and hoarseness",
            "symptom_onset": "this afternoon",
            "symptoms_reported_present": ["barky cough", "hoarse voice"],
            "symptoms_explicitly_denied": ["trouble breathing", "stridor"],
            "advice_given_by_ma": ["sit with him in a steamy bathroom for fifteen minutes"],
            "safety_net_instructions_given": [
                "call back if he starts making a high-pitched noise breathing in"
            ],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-COUGH",
        disposition_id="home_care",
        expect={"relationship_flagged": True},
    ))

    cases.append(_case(
        "medication_heavy",
        "Several medications named; all must survive grounding.",
        label="Bea Lindqvist (11 y)",
        age_months=132,
        lines=[
            ("ma", "Hi, calling about Bea's stomach."),
            ("caller", "She's had cramps for two days. She takes omeprazole and a daily multivitamin."),
            ("ma", "Anything new started recently?"),
            ("caller", "She started ibuprofen for cramps three days ago."),
            ("ma", "Let's stop the ibuprofen for now and see how she does."),
            ("ma", "Call back if the pain moves to the lower right side or she starts vomiting."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "abdominal cramps for two days",
            "symptom_onset": "two days ago",
            "symptoms_reported_present": ["abdominal cramps"],
            "medications_mentioned": ["omeprazole", "daily multivitamin", "ibuprofen"],
            "advice_given_by_ma": ["stop the ibuprofen for now"],
            "safety_net_instructions_given": [
                "call back if the pain moves to the lower right side",
                "call back if she starts vomiting",
            ],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-VOMITING",
        disposition_id="home_care",
        expect={"medications": 3},
    ))

    cases.append(_case(
        "very_short_call",
        "A twenty-second call. The note must still be complete enough to defend.",
        label="Kit Alvarez (14 y)",
        age_months=168,
        lines=[
            ("ma", "Hi, you asked about the rash on Kit's arm."),
            ("caller", "It's gone now, it was just from the grass. Sorry to bother you."),
            ("ma", "No bother at all. Call us if it comes back or spreads."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "rash on the arm, resolved",
            "symptom_onset": None,
            "symptoms_reported_present": ["rash on arm, now resolved"],
            "advice_given_by_ma": [],
            "safety_net_instructions_given": ["call if it comes back or spreads"],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-RASH",
        disposition_id="home_care",
    ))

    cases.append(_case(
        "see_now_disposition",
        "Difficulty breathing; MA brings them in immediately, physician named.",
        label="Arlo Fenn (3 y)",
        age_months=36,
        lines=[
            ("ma", "Hi, this is Jess about Arlo."),
            ("caller", "He's breathing fast and his ribs are sucking in."),
            ("ma", "Is he able to speak in full sentences?"),
            ("caller", "Only a few words at a time."),
            ("ma", "Bring him in right now, I'm telling the front desk you're coming."),
            ("ma", "Dr. Ruiz will see him as soon as you arrive."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "fast breathing with retractions",
            "symptom_onset": None,
            "symptoms_reported_present": [
                "breathing fast",
                "ribs sucking in",
                "only a few words at a time",
            ],
            "advice_given_by_ma": ["bring him in right now"],
            "safety_net_instructions_given": [],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-COUGH",
        disposition_id="escalate_physician",
        supervising_professional_id="dr_ruiz",
    ))

    cases.append(_case(
        "denials_only",
        "A reassurance call: almost everything is a negative.",
        label="Wren Adeyemi (2 y)",
        age_months=24,
        lines=[
            ("ma", "Hi, returning your call about Wren."),
            ("caller", "She fell off the couch. She's not crying now, no bruise, no vomiting."),
            ("ma", "Did she land on her head, is she moving both arms normally?"),
            ("caller", "She landed on her bottom. Both arms are fine, she's walking around."),
            ("ma", "Watch her for the next few hours."),
            ("ma", "Call back if she vomits, becomes very sleepy, or won't use an arm or leg."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "fell off the couch",
            "symptom_onset": None,
            "symptoms_reported_present": ["fell off the couch, landed on bottom"],
            "symptoms_explicitly_denied": [
                "crying now",
                "bruise",
                "vomiting",
                "head impact",
                "abnormal arm movement",
            ],
            "advice_given_by_ma": ["watch her for the next few hours"],
            "safety_net_instructions_given": [
                "call back if she vomits",
                "call back if she becomes very sleepy",
                "call back if she will not use an arm or a leg",
            ],
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-HEAD-INJURY",
        disposition_id="home_care",
    ))

    cases.append(_case(
        "history_mentioned",
        "Relevant history surfaces mid-call and must be captured.",
        label="Dario Sosa (8 y)",
        age_months=96,
        lines=[
            ("ma", "Hi, calling about Dario's wheeze."),
            ("caller", "He's wheezing again. He has asthma, diagnosed at four."),
            ("ma", "Has he used his albuterol?"),
            ("caller", "Twice today. It helps for maybe an hour."),
            ("ma", "Let's see him today. I have two thirty."),
            ("caller", "We'll take it."),
            ("ma", "If he needs albuterol again before then and it doesn't help, call 911."),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "recurrent wheeze",
            "symptom_onset": "today",
            "symptoms_reported_present": ["wheezing", "relief lasting about an hour"],
            "relevant_history_mentioned": ["asthma diagnosed at age four"],
            "medications_mentioned": ["albuterol, used twice today"],
            "advice_given_by_ma": ["come in at two thirty"],
            "safety_net_instructions_given": [
                "call 911 if he needs albuterol again before then and it does not help",
            ],
            "followup_discussed": True,
            "followup_timeframe": "today",
        },
        protocol_id="PLACEHOLDER-COUGH",
        disposition_id="see_today",
        expect={"history": 1},
    ))

    cases.append(_case(
        "silent_line",
        "Nothing intelligible was captured. The note must say so, not invent.",
        label="Case Whitlock (5 y)",
        age_months=60,
        lines=[
            ("ma", "Hello? Can you hear me?", 0.2),
            (None, "[unintelligible]", 0.1),
        ],
        extraction={
            "caller": {"name": None, "relationship_to_patient": None},
            "chief_complaint": "call could not be completed; audio unintelligible",
            "symptom_onset": None,
            "followup_discussed": False,
        },
        protocol_id="PLACEHOLDER-FEVER",
        disposition_id="home_care",
        expect={"gaps_min": 2},
    ))

    return cases


CASES: list[TriageCase] = _build()


def by_name(name: str) -> TriageCase:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(f"no triage fixture named {name!r}")


def response_for(case: TriageCase) -> str:
    return case.model_response


def expand_to(count: int) -> list[TriageCase]:
    """Tile the case list up to `count`, for the 50-call regression suite.

    Tiling rather than generating: a synthetic call produced by a generator
    tests the generator. These are hand-written transcripts, and running the
    same fifteen through the pipeline repeatedly is what a version-bump check
    actually needs -- deterministic inputs and a known-good answer.
    """
    if count <= 0:
        return []
    out: list[TriageCase] = []
    index = 0
    while len(out) < count:
        base = CASES[index % len(CASES)]
        if index < len(CASES):
            out.append(base)
        else:
            clone = TriageCase(
                name=f"{base.name}__r{index // len(CASES)}",
                description=base.description,
                patient_id=base.patient_id,
                family_id=base.family_id,
                patient_label=base.patient_label,
                age_months=base.age_months,
                segments=list(base.segments),
                extraction=dict(base.extraction),
                protocol_id=base.protocol_id,
                disposition_id=base.disposition_id,
                supervising_professional_id=base.supervising_professional_id,
                diarized=base.diarized,
                expect=dict(base.expect),
            )
            out.append(clone)
        index += 1
    return out
