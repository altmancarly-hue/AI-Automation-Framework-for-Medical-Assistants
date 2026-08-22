"""I-03 — Pre-visit chart preparation.

The rules engine produces the checklist; the model produces the narrative
context. README I-03 says they "are not interchangeable", and this package is
organised so that they cannot be confused for one another:

  * `periodicity.py` and `growth.py` contain no model call and never will --
    `make lint` asserts it. Screenings due and percentile crossings are lookup
    tables and arithmetic.
  * `narrative.py` contains the only inference, constrained by Appendix A.4's
    schema and four post-conditions, chief among them: every item must cite an
    encounter date that was actually supplied, and the claim must appear in that
    encounter's note.
  * `brief.py` keeps the two apart on screen, because a reader who cannot tell
    computed content from generated content has a document that reads better
    than the evidence behind it.
  * `feedback.py` measures whether the narrative is helping at all, and tracks
    WRONG separately from NOT USEFUL so that three useful ratings cannot cancel
    a clinician who was pointed at something untrue.

Typical wiring:

    from modules.previsit import (
        GrowthReference, PeriodicitySchedule, NarrativeSynthesizer,
        FeedbackLog, run_batch,
    )

    batch = run_batch(
        tomorrow,
        clinic_date=clinic_date,
        schedule=PeriodicitySchedule.load(),
        reference=GrowthReference(),
        synthesizer=NarrativeSynthesizer(client, audit=audit),
        generated_utc=now,
        feedback=FeedbackLog(db, audit=audit),
    )
"""

from .batch import BriefBatch, PatientDay, run_batch
from .brief import (
    AI_MARKER,
    MAX_LINES,
    BriefSection,
    OpenThread,
    PreVisitBrief,
    assemble,
    render_text,
)
from .feedback import FEEDBACK_SCHEMA, BriefFeedback, FeedbackLog, Verdict
from .growth import (
    MAJOR_CHANNELS,
    ChannelCrossing,
    GrowthPoint,
    GrowthReference,
    Indicator,
    Measurement,
    NotComparable,
    OutOfRange,
    bmi,
    channel_crossing,
    lms_z,
    percentile_to_z,
    z_to_percentile,
)
from .narrative import (
    MAX_ENCOUNTERS,
    NARRATIVE_SCHEMA,
    NARRATIVE_SECTIONS,
    NARRATIVE_SYSTEM_PROMPT,
    ClinicalJudgementLeak,
    Encounter,
    NarrativeContext,
    NarrativeItem,
    NarrativeSynthesizer,
    assert_no_judgement_fields,
)
from .periodicity import (
    CompletedScreening,
    PeriodicitySchedule,
    ScheduleNotReviewed,
    ScreeningDefinition,
    ScreeningStatus,
    Status,
)

__all__ = [
    "AI_MARKER", "BriefBatch", "BriefFeedback", "BriefSection", "ChannelCrossing",
    "ClinicalJudgementLeak", "CompletedScreening", "Encounter", "FEEDBACK_SCHEMA",
    "FeedbackLog", "GrowthPoint", "GrowthReference", "Indicator", "MAJOR_CHANNELS",
    "MAX_ENCOUNTERS", "MAX_LINES", "Measurement", "NARRATIVE_SCHEMA",
    "NARRATIVE_SECTIONS", "NARRATIVE_SYSTEM_PROMPT", "NarrativeContext",
    "NarrativeItem", "NarrativeSynthesizer", "NotComparable", "OpenThread",
    "OutOfRange", "PatientDay", "PeriodicitySchedule", "PreVisitBrief",
    "ScheduleNotReviewed", "ScreeningDefinition", "ScreeningStatus", "Status",
    "Verdict", "assemble", "assert_no_judgement_fields", "bmi",
    "channel_crossing", "lms_z", "percentile_to_z", "render_text", "run_batch",
    "z_to_percentile",
]
