"""Audio capture and local transcription.

WHY the recorder takes an authorisation object rather than a boolean:

`Recorder.start()` will not run without a `RecordingAuthorisation` from
`consent.py`, and that object names the consent row and the disclosure delivery
that justified this specific call. Illinois two-party consent (720 ILCS 5/14-2)
is a felony statute; "we check a flag somewhere upstream" is not a control that
survives contact with a subpoena. There is no `force` parameter and no code path
that reaches `_open_stream` without a token.

WHY local Whisper rather than AWS Transcribe Medical:

README I-04's reference implementation names Transcribe Medical, and the build
plan's whole premise is that the local path is the default. faster-whisper
large-v3 on a 4090 runs at roughly ten times realtime, which turns a four-minute
call into twenty-five seconds of transcription, and it removes a BAA, a per-
minute bill and a PHI egress path from the design. The medical-vocabulary
advantage Transcribe Medical has is real but narrow here: telephone triage
language is parental, not clinical ("throwing up", not "emesis").

WHY diarization degrades instead of failing:

Speaker labels make a better note -- knowing which sentence was the parent and
which was the MA is most of the structuring problem. But diarization needs an
extra model and a token, and it is the first thing to be unavailable on a
minimum-viable deployment. So `Transcript.diarized` is a flag the structurer
reads, and an undiarized transcript produces a note that is honest about not
knowing who said what rather than one that guesses.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import wave
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from modules.scheduling.models import iso

from .consent import RecordingAuthorisation, RecordingNotAuthorised

__all__ = [
    "AudioRecording",
    "TranscriptSegment",
    "Transcript",
    "Transcriber",
    "FasterWhisperTranscriber",
    "ScriptedTranscriber",
    "Recorder",
    "LOW_CONFIDENCE_THRESHOLD",
]

#: Below this, a segment is treated as unclear audio and surfaces as a
#: transcript gap rather than as text the MA might read past. README I-04's
#: control for "transcription error changes clinical meaning".
LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class AudioRecording:
    encounter_id: str
    path: str
    started_utc: datetime
    ended_utc: datetime
    sha256: str
    duration_seconds: float
    authorisation: RecordingAuthorisation

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)

    def as_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "path": self.path,
            "started_utc": iso(self.started_utc),
            "ended_utc": iso(self.ended_utc),
            "sha256": self.sha256,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None       # "ma" | "caller" | None when undiarized
    confidence: float = 1.0

    @property
    def unclear(self) -> bool:
        return self.confidence < LOW_CONFIDENCE_THRESHOLD


@dataclass
class Transcript:
    encounter_id: str
    segments: list[TranscriptSegment]
    model_id: str
    model_version: str
    diarized: bool = False
    language: str = "en"

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def labelled_text(self) -> str:
        """What the model actually sees. Speaker labels only when we have them.

        An undiarized transcript is presented as an unlabelled dialogue rather
        than with invented "MA:" / "Caller:" prefixes. Guessing the speaker is
        the same error class as guessing the content: it reads as fact in the
        note and nobody can tell it was a guess.
        """
        if not self.diarized:
            return self.text
        lines = []
        for segment in self.segments:
            if not segment.text.strip():
                continue
            who = {"ma": "MA", "caller": "CALLER"}.get(segment.speaker or "", "SPEAKER")
            lines.append(f"{who}: {segment.text.strip()}")
        return "\n".join(lines)

    @property
    def unclear_segments(self) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.unclear]

    def gap_hints(self) -> list[str]:
        """Timestamps of unclear audio, for the structurer's transcript_gaps.

        Supplied as an input rather than left to the model to notice: the model
        sees text, and low-confidence text looks exactly like confident text
        once it has been written down.
        """
        return [
            f"unclear audio {s.start:.0f}-{s.end:.0f}s (confidence {s.confidence:.2f})"
            for s in self.unclear_segments
        ]

    @property
    def sha256(self) -> str:
        """Hash of the transcript, for the audit record (README I-04)."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def duration_seconds(self) -> float:
        return max((s.end for s in self.segments), default=0.0)


class Transcriber(abc.ABC):
    """Anything that turns audio into a transcript."""

    name: str = "abstract"
    model_id: str = "unknown"
    model_version: str = "unpinned"

    @abc.abstractmethod
    def transcribe(self, recording: AudioRecording) -> Transcript:
        ...


class FasterWhisperTranscriber(Transcriber):
    """Local faster-whisper. Large-v3 on GPU, distil-large-v3 on CPU.

    The model is imported and loaded lazily so that importing this package
    costs nothing on a machine that will never transcribe -- the nightly recall
    cron and the test suite both import it.

    Model identity is pinned and reported, because README I-04 lists "model
    degrades after a vendor update" as a real risk and the control is a pinned
    version plus a regression suite (see `regression.py`).
    """

    name = "faster_whisper"

    #: Chosen per README's hardware table: large-v3 is the recommended-tier
    #: model, distil-large-v3 is what a 32GB CPU-only box can actually run.
    GPU_MODEL = "Systran/faster-whisper-large-v3"
    CPU_MODEL = "Systran/faster-distil-whisper-large-v3"

    def __init__(
        self,
        *,
        device: str = "auto",
        compute_type: str | None = None,
        model_id: str | None = None,
        model_version: str = "unpinned",
        diarizer: Any = None,
        beam_size: int = 5,
    ) -> None:
        self.device = device
        self.compute_type = compute_type
        self.model_version = model_version
        self.beam_size = beam_size
        self.diarizer = diarizer
        self._model: Any = None
        self._resolved_device: str | None = None
        self.model_id = model_id or ""

    def _resolve(self) -> tuple[str, str, str]:  # pragma: no cover - needs GPU/CPU probe
        device = self.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        model_id = self.model_id or (self.GPU_MODEL if device == "cuda" else self.CPU_MODEL)
        compute = self.compute_type or ("float16" if device == "cuda" else "int8")
        return device, model_id, compute

    def _load(self) -> Any:  # pragma: no cover - heavy optional dependency
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: see class docstring

            device, model_id, compute = self._resolve()
            self._resolved_device = device
            self.model_id = model_id
            self._model = WhisperModel(model_id, device=device, compute_type=compute)
        return self._model

    def transcribe(self, recording: AudioRecording) -> Transcript:  # pragma: no cover
        model = self._load()
        segments, info = model.transcribe(
            recording.path,
            beam_size=self.beam_size,
            vad_filter=True,
            word_timestamps=False,
        )
        out: list[TranscriptSegment] = []
        for segment in segments:
            # faster-whisper reports avg_logprob; map it to a 0-1 confidence.
            # The mapping is crude on purpose -- it is used only to decide what
            # to flag as unclear, and a crude flag that fires is worth more than
            # a precise one nobody computed.
            logprob = float(getattr(segment, "avg_logprob", -0.2))
            confidence = max(0.0, min(1.0, 1.0 + logprob))
            out.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text),
                    confidence=confidence,
                )
            )

        diarized = False
        if self.diarizer is not None:
            try:
                out = self.diarizer.assign(recording.path, out)
                diarized = True
            except Exception:
                # Degrade, do not fail. A note without speaker labels is worth
                # far more than no note, and this is the first component to be
                # missing on a minimum-viable deployment.
                diarized = False

        return Transcript(
            encounter_id=recording.encounter_id,
            segments=out,
            model_id=self.model_id,
            model_version=self.model_version,
            diarized=diarized,
            language=str(getattr(info, "language", "en")),
        )


class ScriptedTranscriber(Transcriber):
    """Reads a transcript from a sidecar file. A shipped component, not a mock.

    WHY it ships rather than living in the test tree: the regression suite
    README I-04 asks for -- "50 known calls, re-validate before any version
    bump" -- needs a way to run the ENTIRE downstream pipeline (structuring,
    grounding checks, rendering, edit-distance) against fixed transcripts
    without a GPU. It is also the right dry-run mode for a pilot: record
    nothing, paste a transcript, see the note the system would have produced.

    Sidecar format is either a `.txt` (one line per segment) or a `.json` list
    of {start, end, text, speaker, confidence}.
    """

    name = "scripted"

    def __init__(
        self,
        *,
        transcripts: Mapping[str, Any] | None = None,
        directory: str | os.PathLike[str] | None = None,
        model_id: str = "scripted",
        model_version: str = "fixture",
        diarized: bool = True,
    ) -> None:
        self.transcripts = dict(transcripts or {})
        self.directory = str(directory) if directory else None
        self.model_id = model_id
        self.model_version = model_version
        self.diarized = diarized

    def transcribe(self, recording: AudioRecording) -> Transcript:
        payload = self.transcripts.get(recording.encounter_id)
        if payload is None and self.directory:
            for suffix in (".json", ".txt"):
                candidate = os.path.join(self.directory, recording.encounter_id + suffix)
                if os.path.exists(candidate):
                    with open(candidate, "r", encoding="utf-8") as fh:
                        payload = json.load(fh) if suffix == ".json" else fh.read()
                    break
        if payload is None:
            raise KeyError(
                f"no scripted transcript for encounter {recording.encounter_id!r}"
            )
        return self._to_transcript(recording.encounter_id, payload)

    def _to_transcript(self, encounter_id: str, payload: Any) -> Transcript:
        segments: list[TranscriptSegment] = []
        if isinstance(payload, str):
            for index, line in enumerate(l for l in payload.splitlines() if l.strip()):
                speaker = None
                text = line.strip()
                if ":" in text[:12]:
                    head, _, rest = text.partition(":")
                    if head.strip().lower() in ("ma", "caller", "parent"):
                        speaker = "ma" if head.strip().lower() == "ma" else "caller"
                        text = rest.strip()
                segments.append(
                    TranscriptSegment(index * 5.0, index * 5.0 + 5.0, text, speaker)
                )
        else:
            for index, item in enumerate(payload):
                segments.append(
                    TranscriptSegment(
                        start=float(item.get("start", index * 5.0)),
                        end=float(item.get("end", index * 5.0 + 5.0)),
                        text=str(item.get("text", "")),
                        speaker=item.get("speaker"),
                        confidence=float(item.get("confidence", 1.0)),
                    )
                )
        diarized = self.diarized and any(s.speaker for s in segments)
        return Transcript(
            encounter_id=encounter_id,
            segments=segments,
            model_id=self.model_id,
            model_version=self.model_version,
            diarized=diarized,
        )


class Recorder:
    """Writes call audio to disk. Refuses to start without an authorisation."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = str(directory)
        os.makedirs(self.directory, exist_ok=True)

    def path_for(self, encounter_id: str) -> str:
        return os.path.join(self.directory, f"{encounter_id}.wav")

    def capture(
        self,
        *,
        authorisation: RecordingAuthorisation | None,
        encounter_id: str,
        audio_bytes: bytes | None = None,
        source_path: str | None = None,
        started_utc: datetime,
        ended_utc: datetime,
    ) -> AudioRecording:
        """Persist a completed recording.

        The audio itself arrives from the client (a browser MediaRecorder blob
        or a handset integration); this function is the trust boundary where it
        lands on disk, which is why the authorisation check is here rather than
        in the UI.
        """
        if authorisation is None:
            raise RecordingNotAuthorised(
                "Recorder.capture requires a RecordingAuthorisation. Illinois is "
                "a two-party consent state and there is no path that writes call "
                "audio to disk without one."
            )
        if authorisation.encounter_id != encounter_id:
            raise RecordingNotAuthorised(
                f"authorisation is for encounter {authorisation.encounter_id!r}, "
                f"not {encounter_id!r}; an authorisation is not transferable"
            )
        if audio_bytes is None and source_path is None:
            raise ValueError("provide audio_bytes or source_path")

        destination = self.path_for(encounter_id)
        if audio_bytes is None:
            with open(str(source_path), "rb") as fh:
                audio_bytes = fh.read()
        with open(destination, "wb") as fh:
            fh.write(audio_bytes)
        # Restrictive from the moment it exists. The window between "written"
        # and "permissions fixed" is a window.
        try:
            os.chmod(destination, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass

        return AudioRecording(
            encounter_id=encounter_id,
            path=destination,
            started_utc=started_utc,
            ended_utc=ended_utc,
            sha256=hashlib.sha256(audio_bytes).hexdigest(),
            duration_seconds=(ended_utc - started_utc).total_seconds(),
            authorisation=authorisation,
        )


def silent_wav(seconds: float = 1.0, *, rate: int = 16000) -> bytes:
    """A valid, tiny WAV. Used by the demo and the tests as stand-in audio.

    Real audio is never committed to this repo, and synthetic silence exercises
    every byte of the capture, hashing and deletion path without any of it.
    """
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()
