"""I-04 — Telephone triage documentation.

The machine transcribes and writes down. The medical assistant assesses, selects
the protocol, determines the disposition and gives the advice. That division is
not caution — it is what Illinois delegation law requires, and README I-04 calls
it "the single most important design constraint in the entire document".

Where the constraint is enforced, in ascending order of how much it protects
anyone:

  1. `structure.NOTE_SCHEMA` has no field for a disposition, protocol or
     diagnosis (README Appendix A.1, verbatim).
  2. `structure.assert_no_clinical_judgement_fields` refuses any schema whose
     key names read as clinical judgement, at import time and per call.
  3. Every extracted string is checked against the transcript; advice claimed
     for the MA must match something the MA actually said.
  4. `protocols.ProtocolRegistry` offers a searchable list and nothing else —
     no suggestion, no ranking, no reordering by plausibility.

Typical wiring:

    from modules.triage import (
        ConsentRegistry, Recorder, ScriptedTranscriber, NoteStructurer,
        ProtocolRegistry, NoteReviewer, AudioLifecycle, TriageService,
    )

    service = TriageService(
        db,
        consent=ConsentRegistry(db, audit=audit),
        recorder=Recorder(audio_dir),
        transcriber=FasterWhisperTranscriber(),
        structurer=NoteStructurer(client, audit=audit),
        registry=ProtocolRegistry.load(),
        reviewer=NoteReviewer(audit=audit, registry=registry),
        lifecycle=AudioLifecycle(db, audit=audit),
    )
"""

from .capture import (
    LOW_CONFIDENCE_THRESHOLD,
    AudioRecording,
    FasterWhisperTranscriber,
    Recorder,
    ScriptedTranscriber,
    Transcriber,
    Transcript,
    TranscriptSegment,
    silent_wav,
)
from .consent import (
    DISCLOSURE_SCRIPT,
    ConsentBasis,
    ConsentRegistry,
    DeclinedRecording,
    DisclosureDelivery,
    RecordingAuthorisation,
    RecordingNotAuthorised,
)
from .encounter import (
    EncounterState,
    IllegalTransition,
    TriageEncounter,
    TriageService,
)
from .lifecycle import (
    HARD_RETENTION_CEILING,
    AudioLifecycle,
    RetentionBreach,
)
from .protocols import (
    Disposition,
    MATaps,
    Protocol,
    ProtocolRegistry,
    TapsIncomplete,
)
from .regression import (
    ModelNotValidated,
    ModelPin,
    RegressionResult,
    run_regression,
)
from .render import (
    FollowUpTask,
    build_followup_task,
    render_draft,
    render_note,
    signature_block,
)
from .review import (
    POOR_DRAFT_ALARM_RATIO,
    RUBBER_STAMP_ALARM_RATIO,
    NoteReviewer,
    SignedNote,
    UnreviewedDropError,
)
from .structure import (
    A1_KEYS,
    BANNED_FIELD_TOKENS,
    NOTE_SCHEMA,
    STRUCTURING_SYSTEM_PROMPT,
    ClinicalJudgementLeak,
    NoteStructurer,
    StructuredNote,
    assert_matches_a1_shape,
    assert_no_clinical_judgement_fields,
    assert_output_keys_are_a1,
)

__all__ = [
    "A1_KEYS",
    "AudioLifecycle",
    "AudioRecording",
    "BANNED_FIELD_TOKENS",
    "ClinicalJudgementLeak",
    "ConsentBasis",
    "ConsentRegistry",
    "DISCLOSURE_SCRIPT",
    "DeclinedRecording",
    "Disposition",
    "DisclosureDelivery",
    "EncounterState",
    "FasterWhisperTranscriber",
    "FollowUpTask",
    "HARD_RETENTION_CEILING",
    "IllegalTransition",
    "LOW_CONFIDENCE_THRESHOLD",
    "MATaps",
    "ModelNotValidated",
    "ModelPin",
    "NOTE_SCHEMA",
    "NoteReviewer",
    "NoteStructurer",
    "POOR_DRAFT_ALARM_RATIO",
    "Protocol",
    "ProtocolRegistry",
    "RUBBER_STAMP_ALARM_RATIO",
    "Recorder",
    "RecordingAuthorisation",
    "RecordingNotAuthorised",
    "RegressionResult",
    "RetentionBreach",
    "STRUCTURING_SYSTEM_PROMPT",
    "ScriptedTranscriber",
    "SignedNote",
    "StructuredNote",
    "TapsIncomplete",
    "Transcriber",
    "Transcript",
    "TranscriptSegment",
    "TriageEncounter",
    "TriageService",
    "UnreviewedDropError",
    "assert_matches_a1_shape",
    "assert_no_clinical_judgement_fields",
    "assert_output_keys_are_a1",
    "build_followup_task",
    "render_draft",
    "render_note",
    "run_regression",
    "signature_block",
    "silent_wav",
]
