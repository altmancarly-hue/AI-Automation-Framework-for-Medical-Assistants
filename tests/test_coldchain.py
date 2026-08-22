"""I-08 — vaccine cold chain telemetry.

The module with no model in it. Several of these tests exist to prove that, and
`make lint` runs them.
"""

from __future__ import annotations

import ast
import os
from datetime import date, datetime, timedelta, timezone

import pytest

from modules.coldchain import (
    DISPOSITION_ADVISORY,
    Alert,
    AlertKind,
    ColdChainConfig,
    ColdChainMonitor,
    LotDisposition,
    Reading,
    ScriptedFeed,
    Severity,
    StorageUnit,
    UnknownUnitType,
    UnreviewedThresholds,
    VaccineLot,
    build_compliance_report,
    open_excursion,
)
from modules.coldchain import monitor as monitor_module
from modules.coldchain.fixtures import (
    INVENTORY,
    MANUFACTURER_CONTACTS,
    NOW,
    UNITS,
    build_feed,
    unit,
)
from nsp_core.audit import AuditLog

_MODULE_DIR = os.path.dirname(os.path.abspath(monitor_module.__file__))


@pytest.fixture
def config():
    loaded = ColdChainConfig.load()
    loaded.review["owner"] = "test-owner"
    return loaded


@pytest.fixture
def monitor(config):
    return ColdChainMonitor(
        config=config, units={u.unit_id: u for u in UNITS}, feed=build_feed()
    )


def kinds(alerts, unit_id=None):
    return {a.kind for a in alerts if unit_id is None or a.unit_id == unit_id}


# ==========================================================================
# the discipline of not using AI
# ==========================================================================


def test_there_is_no_model_anywhere_in_this_module():
    """README I-08: "Does this need an LLM? No. Emphatically no... Include this
    initiative in an AI program specifically to demonstrate the discipline of
    not using AI where it does not belong.\""""
    for name in sorted(os.listdir(_MODULE_DIR)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_MODULE_DIR, name), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "llm" not in node.module, (name, node.module)
                assert node.module not in ("openai", "anthropic"), name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("openai", "anthropic"), name


def test_no_cloud_sdk_and_no_network_at_module_scope():
    for name in sorted(os.listdir(_MODULE_DIR)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_MODULE_DIR, name), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in (
                        "boto3", "botocore", "requests", "httpx", "paho", "socket"
                    ), name


# ==========================================================================
# the gate
# ==========================================================================


def test_thresholds_need_a_named_clinical_owner():
    """These numbers decide whether $20,000 of vaccine is thrown away or given
    to a child."""
    shipped = ColdChainConfig.load()
    assert not shipped.has_clinical_owner
    with pytest.raises(UnreviewedThresholds, match="no named owner"):
        ColdChainMonitor(config=shipped, units={}, feed=build_feed()).evaluate(now=NOW)


def test_a_unit_of_unknown_type_is_refused_at_construction(config):
    """A freezer scored against the refrigerator band screams permanently; a
    refrigerator scored against the freezer band is permanently silent."""
    with pytest.raises(UnknownUnitType, match="not a configured unit type"):
        ColdChainMonitor(
            config=config,
            units={"x": StorageUnit("x", "X", "wine_cooler")},
            feed=build_feed(),
        )


def test_an_inverted_band_is_refused_at_load(tmp_path):
    import yaml

    data = yaml.safe_load(open("config/coldchain.yaml", encoding="utf-8"))
    data["unit_types"]["refrigerator"]["min_c"] = 9.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="inverted"):
        ColdChainConfig.load(path)


def test_a_warning_margin_that_swallows_the_range_is_refused(tmp_path):
    """Every in-range reading would warn, which is the same as not warning."""
    import yaml

    data = yaml.safe_load(open("config/coldchain.yaml", encoding="utf-8"))
    data["unit_types"]["refrigerator"]["warning_margin_c"] = 4.0   # range is 6
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="every in-range reading would warn"):
        ColdChainConfig.load(path)


# ==========================================================================
# the five conditions
# ==========================================================================


def test_a_healthy_practice_produces_almost_no_alerts(config):
    """An alert that fires on everything fails the same way as no alert."""
    quiet = ColdChainMonitor(
        config=config,
        units={u.unit_id: u for u in UNITS},
        feed=build_feed("all_quiet"),
    )
    alerts = quiet.evaluate(now=NOW)
    # Only the two genuine calibration findings, which are real.
    assert kinds(alerts) == {AlertKind.CALIBRATION_DUE}


def test_a_dead_logger_is_an_alarm(monitor):
    """THE important one. README I-08 rates it HIGH: "a dead logger is worse
    than no logger because it creates false confidence.\""""
    alerts = monitor.evaluate(now=NOW)
    offline = [
        a for a in alerts
        if a.kind == AlertKind.SENSOR_OFFLINE and a.unit_id == "fridge_3"
    ]
    assert offline
    assert offline[0].severity == Severity.CRITICAL
    assert offline[0].duration_minutes > 30
    assert "NOT being monitored" in offline[0].detail


def test_a_unit_with_no_readings_at_all_is_offline_not_healthy(config):
    """The failure that has no data to react to."""
    lonely = StorageUnit("empty", "Never reported", "refrigerator")
    alerts = ColdChainMonitor(
        config=config, units={"empty": lonely}, feed=ScriptedFeed(data={})
    ).evaluate(now=NOW)
    assert kinds(alerts) == {AlertKind.SENSOR_OFFLINE}


def test_a_silent_unit_still_reports_the_excursion_it_was_in(config):
    """A unit can be both silent now and have been out of range when it last
    spoke, and the excursion is the half that puts vaccine at risk."""
    stopped = NOW - timedelta(hours=4)
    feed = ScriptedFeed(data={
        "u": [
            Reading("u", stopped - timedelta(minutes=m * 5), temperature_c=17.0)
            for m in range(12, 0, -1)
        ]
    })
    alerts = ColdChainMonitor(
        config=config,
        units={"u": StorageUnit("u", "U", "refrigerator")},
        feed=feed,
    ).evaluate(now=NOW)
    assert AlertKind.SENSOR_OFFLINE in kinds(alerts)
    assert AlertKind.SUSTAINED_EXCURSION in kinds(alerts)


def test_a_sustained_excursion_escalates_further_than_a_brief_one(config):
    """A door opened for ninety seconds and a compressor that died are the same
    reading and very different events."""
    def alerts_for(minutes_out: float):
        readings = []
        for step in range(24):
            taken = NOW - timedelta(minutes=step * 5)
            readings.append(
                Reading("u", taken, temperature_c=14.0 if
                        (NOW - taken).total_seconds() / 60 <= minutes_out else 4.5)
            )
        return ColdChainMonitor(
            config=config,
            units={"u": StorageUnit("u", "U", "refrigerator")},
            feed=ScriptedFeed(data={"u": readings}),
        ).evaluate(now=NOW)

    brief = alerts_for(6)
    assert AlertKind.EXCURSION in kinds(brief)
    assert AlertKind.SUSTAINED_EXCURSION not in kinds(brief)

    long = alerts_for(45)
    assert AlertKind.SUSTAINED_EXCURSION in kinds(long)
    escalation = [a for a in long if a.kind == AlertKind.SUSTAINED_EXCURSION][0]
    assert "physician_partner" in escalation.notify
    assert "practice_owner" in escalation.notify


def test_a_duration_bounded_by_the_window_says_at_least(monitor):
    """Saying "1440 minutes" about a compressor that died sixty hours ago
    understates it by a factor of two and a half, and the number goes on a
    document."""
    narrow = monitor.evaluate(now=NOW, window=timedelta(hours=24))
    bounded = [a for a in narrow if a.unit_id == "fridge_2" and a.inventory_at_risk][0]
    assert "at least" in bounded.detail

    wide = monitor.evaluate(now=NOW, window=timedelta(hours=96))
    measured = [a for a in wide if a.unit_id == "fridge_2" and a.inventory_at_risk][0]
    assert "at least" not in measured.detail
    assert measured.duration_minutes == pytest.approx(60 * 60, abs=10)


def test_a_door_left_open_alerts_and_the_clock_does_not_restart(monitor, config):
    """Measured from the FIRST consecutive open reading. From the last one, the
    three-minute rule would never fire."""
    alerts = monitor.evaluate(now=NOW)
    door = [a for a in alerts if a.kind == AlertKind.DOOR_OPEN]
    assert door and door[0].unit_id == "fridge_1"
    assert door[0].duration_minutes >= 3

    # A door open for two minutes is under the three-minute limit. Someone
    # taking a vial out is not an alert.
    readings = [
        Reading("u", NOW - timedelta(minutes=m), temperature_c=4.5,
                door_open=m <= 2)
        for m in range(60, -1, -1)
    ]
    quiet = ColdChainMonitor(
        config=config, units={"u": StorageUnit("u", "U", "refrigerator")},
        feed=ScriptedFeed(data={"u": readings}),
    ).evaluate(now=NOW)
    assert AlertKind.DOOR_OPEN not in kinds(quiet)


def test_the_bands_are_per_unit_type(config):
    """-20C is a healthy freezer and a catastrophic refrigerator."""
    readings = [
        Reading("u", NOW - timedelta(minutes=m), temperature_c=-20.0)
        for m in range(60, -1, -5)
    ]
    for unit_type, expect_excursion in (("freezer", False), ("refrigerator", True)):
        alerts = ColdChainMonitor(
            config=config,
            units={"u": StorageUnit("u", "U", unit_type)},
            feed=ScriptedFeed(data={"u": readings}),
        ).evaluate(now=NOW)
        assert (AlertKind.SUSTAINED_EXCURSION in kinds(alerts)) is expect_excursion


def test_a_low_battery_alerts_before_the_logger_dies(monitor):
    alerts = monitor.evaluate(now=NOW)
    battery = [a for a in alerts if a.kind == AlertKind.LOW_BATTERY]
    assert battery and battery[0].unit_id == "freezer_1"


def test_a_lapsed_calibration_is_critical_not_a_reminder(monitor):
    """A lapsed NIST certificate makes every reading since unusable as evidence
    -- a compliance failure with no symptom until an auditor asks."""
    alerts = monitor.evaluate(now=NOW)
    lapsed = [
        a for a in alerts
        if a.kind == AlertKind.CALIBRATION_DUE and a.unit_id == "fridge_3"
    ]
    assert lapsed and lapsed[0].severity == Severity.CRITICAL
    assert "LAPSED" in lapsed[0].detail

    upcoming = [
        a for a in alerts
        if a.kind == AlertKind.CALIBRATION_DUE and a.unit_id == "fridge_2"
    ]
    assert upcoming and upcoming[0].severity == Severity.WARNING


def test_every_notification_is_audited(config, tmp_path):
    audit = AuditLog(tmp_path / "a.sqlite3", hmac_key=b"k")
    monitor = ColdChainMonitor(
        config=config, units={u.unit_id: u for u in UNITS},
        feed=build_feed(), audit=audit,
    )
    sent = monitor.notifications(now=NOW)
    rows = audit.query("SELECT * FROM event WHERE initiative_id = ?", ("I-08",))
    assert len(rows) == len(sent)


def test_a_standing_condition_is_not_re_announced_on_every_poll(config, tmp_path):
    """A five-minute poll announces every standing condition 288 times a day. A
    calibration date thirty days out would page the office manager 8,640 times
    before it came due -- the alert-fatigue failure this module's own config
    comments warn about."""
    audit = AuditLog(tmp_path / "a.sqlite3", hmac_key=b"k")
    monitor = ColdChainMonitor(
        config=config, units={u.unit_id: u for u in UNITS},
        feed=build_feed(), audit=audit,
    )
    true_at_any_moment = len(monitor.evaluate(now=NOW))
    assert true_at_any_moment >= 6

    sent = 0
    for poll in range(288):
        sent += len(
            monitor.notifications(now=NOW - timedelta(minutes=5 * (287 - poll)))
        )
    # Without suppression this is 288 x the number of live conditions.
    assert sent < true_at_any_moment * 30, sent
    assert sent >= true_at_any_moment, "each condition is announced at least once"
    assert len(audit.query("SELECT * FROM event", ())) == sent


def test_a_condition_that_clears_and_returns_is_announced_again(config):
    """An excursion that recurs after being fixed is news."""
    unit_obj = StorageUnit("u", "U", "refrigerator", has_door_sensor=False)

    def monitor_with(temp: float, at: datetime):
        return [
            Reading("u", at - timedelta(minutes=m), temperature_c=temp)
            for m in range(60, -1, -5)
        ]

    monitor = ColdChainMonitor(config=config, units={"u": unit_obj}, feed=ScriptedFeed())
    monitor.feed.data["u"] = monitor_with(17.0, NOW)
    assert monitor.notifications(now=NOW)
    # Still true five minutes later: not re-announced.
    assert not monitor.notifications(now=NOW + timedelta(minutes=5))
    # Fixed.
    monitor.feed.data["u"] = monitor_with(4.5, NOW + timedelta(hours=2))
    assert not monitor.notifications(now=NOW + timedelta(hours=2))
    # Broken again: announced.
    monitor.feed.data["u"] = monitor_with(17.0, NOW + timedelta(hours=3))
    assert monitor.notifications(now=NOW + timedelta(hours=3))


# ==========================================================================
# the excursion workflow
# ==========================================================================


def test_an_excursion_record_names_every_lot_in_the_unit(monitor):
    """The exposure happened to the APPLIANCE. Which products tolerate it is the
    manufacturer's question, and narrowing the list here would be answering it."""
    alert = next(
        a for a in monitor.evaluate(now=NOW, window=timedelta(hours=96))
        if a.kind == AlertKind.SUSTAINED_EXCURSION
    )
    record = open_excursion(
        alert, unit(alert.unit_id),
        build_feed().readings(alert.unit_id, since=NOW - timedelta(hours=96), until=NOW),
        INVENTORY, excursion_id="EXC-1",
        manufacturer_contacts=MANUFACTURER_CONTACTS,
    )
    assert {lot.lot_number for lot in record.affected_lots} == {
        lot.lot_number for lot in INVENTORY if lot.unit_id == alert.unit_id
    }
    assert all(lot.disposition == LotDisposition.QUARANTINED for lot in record.affected_lots)


def test_an_excursion_record_cannot_be_opened_for_a_non_excursion(monitor):
    alert = next(
        a for a in monitor.evaluate(now=NOW) if a.kind == AlertKind.DOOR_OPEN
    )
    with pytest.raises(ValueError, match="does not put inventory at risk"):
        open_excursion(alert, unit(alert.unit_id), [], INVENTORY, excursion_id="X")


def test_the_module_never_decides_whether_exposed_vaccine_is_usable(monitor):
    """A wrong "fine" gives children ineffective vaccine; a wrong "discard"
    throws away $20,000 and recalls patients who did not need it."""
    record = _excursion(monitor)
    text = record.quarantine_label().lower()
    for forbidden in ("probably fine", "likely usable", "safe to use", "discard all"):
        assert forbidden not in text
    assert "only the manufacturer can say" in text


def test_an_excursion_cannot_be_closed_with_lots_still_quarantined(monitor):
    record = _excursion(monitor)
    record.corrective_action = "Unit taken out of service."
    with pytest.raises(ValueError, match="no recorded disposition"):
        record.close(closed_by="ma_jess", now=NOW)


def test_an_excursion_cannot_be_closed_without_a_corrective_action(monitor):
    record = _excursion(monitor)
    for lot in record.affected_lots:
        record.record_disposition(
            lot.lot_number, disposition=LotDisposition.DISCARDED,
            decided_by="ma_jess", note="Called Sanofi ref#1, discard.",
        )
    with pytest.raises(ValueError, match="No corrective action"):
        record.close(closed_by="ma_jess", now=NOW)


def test_a_disposition_names_who_obtained_it_and_what_they_were_told(monitor):
    record = _excursion(monitor)
    lot = record.affected_lots[0]
    with pytest.raises(ValueError, match="names the person"):
        record.record_disposition(
            lot.lot_number, disposition=LotDisposition.RELEASED,
            decided_by="  ", note="fine",
        )
    with pytest.raises(ValueError, match="what the manufacturer said"):
        record.record_disposition(
            lot.lot_number, disposition=LotDisposition.RELEASED,
            decided_by="ma_jess", note="",
        )


def _excursion(monitor):
    alert = next(
        a for a in monitor.evaluate(now=NOW, window=timedelta(hours=96))
        if a.kind == AlertKind.SUSTAINED_EXCURSION
    )
    return open_excursion(
        alert, unit(alert.unit_id),
        build_feed().readings(alert.unit_id, since=NOW - timedelta(hours=96), until=NOW),
        INVENTORY, excursion_id="EXC-1",
        manufacturer_contacts=MANUFACTURER_CONTACTS,
    )


# ==========================================================================
# the monthly record
# ==========================================================================


def test_the_daily_min_max_is_computed_not_transcribed(config):
    """README I-08 target state 4: "Automatic min/max daily records -- no human
    transcription, no initials, no paper." The transcription is the part that
    produces the audit finding."""
    report = build_compliance_report(
        month="2026-08", units=UNITS, feed=build_feed(), config=config,
        start=date(2026, 8, 23), end=date(2026, 8, 23), now=NOW,
    )
    fridge_1 = [d for d in report.days if d.unit_id == "fridge_1"][0]
    assert fridge_1.min_c == 4.4
    assert fridge_1.max_c == 4.4
    assert fridge_1.reading_count == 288       # 5-minute logging, 24 hours


def test_the_report_declares_its_own_gaps(config):
    """A report that says 100% compliant every month is a report nobody checks."""
    report = build_compliance_report(
        month="2026-08", units=UNITS, feed=build_feed(), config=config,
        start=date(2026, 8, 23), end=date(2026, 8, 24), now=NOW,
    )
    findings = report.findings()
    assert any("fridge_3" in f and "calibration lapsed" in f for f in findings)
    assert any("unmonitored" in f for f in findings)
    assert report.coverage < 1.0


def test_an_unclosed_excursion_is_a_finding(config, monitor):
    record = _excursion(monitor)
    report = build_compliance_report(
        month="2026-08", units=UNITS, feed=build_feed(), config=config,
        start=date(2026, 8, 23), end=date(2026, 8, 23), now=NOW,
        excursions=[record],
    )
    assert any("still open" in f for f in report.findings())
    assert report.unclosed_excursions


def test_a_day_missing_one_poll_in_a_hundred_is_still_monitored(config):
    """A cellular logger that drops a poll has not stopped monitoring the fridge,
    and flagging every such day trains the reader to ignore the list."""
    day = date(2026, 8, 23)
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    readings = [
        Reading("u", start + timedelta(minutes=5 * i), temperature_c=4.5)
        for i in range(288)
        if i % 100                                   # drop ~1%
    ]
    report = build_compliance_report(
        month="2026-08",
        units=[StorageUnit("u", "U", "refrigerator")],
        feed=ScriptedFeed(data={"u": readings}), config=config,
        start=day, end=day, now=NOW,
    )
    assert report.days[0].complete


# ==========================================================================
# adversarial-review regressions
# ==========================================================================


def test_r01_a_probe_that_dies_while_the_radio_lives_is_an_alarm(config):
    """#1 Silence was measured from the last PACKET. A logger whose thermistor
    fails but whose radio keeps transmitting sent `temperature_c=None` every
    five minutes: the unit looked perfectly monitored, and the threshold checks
    -- which filter on `temperature_c is not None` -- ran over nothing. Zero
    alerts, indefinitely, on a fridge holding $21,000 of vaccine."""
    unit_obj = StorageUnit("u", "U", "refrigerator", has_door_sensor=False)
    failed_at = NOW - timedelta(hours=12)
    readings = [
        Reading(
            "u", NOW - timedelta(minutes=5 * step),
            temperature_c=None if NOW - timedelta(minutes=5 * step) > failed_at else 4.5,
            battery_percent=88.0,
        )
        for step in range(288, -1, -1)
    ]
    alerts = ColdChainMonitor(
        config=config, units={"u": unit_obj}, feed=ScriptedFeed(data={"u": readings})
    ).evaluate(now=NOW)
    offline = [a for a in alerts if a.kind == AlertKind.SENSOR_OFFLINE]
    assert offline, "a probe that reports nothing is an unmonitored fridge"
    assert offline[0].duration_minutes >= 60 * 11
    assert "probe has failed, not the radio" in offline[0].detail


def test_r02_a_null_door_sample_does_not_cancel_the_alarm(config):
    """#2 `door_open=None` means the channel said NOTHING, not "closed". One
    null cancelled a ninety-minute open-door alarm, and a null in the middle of
    the stretch cut the reported duration to a quarter of the truth."""
    unit_obj = StorageUnit("u", "U", "refrigerator")
    opened = NOW - timedelta(minutes=90)

    def readings(null_at=None):
        out = []
        for step in range(120, -1, -1):
            taken = NOW - timedelta(minutes=step)
            state = taken >= opened
            if null_at is not None and taken == null_at:
                state = None
            out.append(Reading("u", taken, temperature_c=4.5, door_open=state))
        return out

    for label, null_at in (
        ("clean", None),
        ("newest sample null", NOW),
        ("null mid-stretch", NOW - timedelta(minutes=30)),
    ):
        alerts = ColdChainMonitor(
            config=config, units={"u": unit_obj},
            feed=ScriptedFeed(data={"u": readings(null_at)}),
        ).evaluate(now=NOW)
        door = [a for a in alerts if a.kind == AlertKind.DOOR_OPEN]
        assert door, label
        assert door[0].duration_minutes == pytest.approx(90, abs=1), label


def test_r02_a_dead_door_channel_is_reported(config):
    """`has_door_sensor` was declared on the unit and read by nothing, so a door
    channel that died produced no alert of any kind -- the same false confidence
    as a dead thermometer, on the most common cause of an excursion."""
    with_sensor = StorageUnit("u", "U", "refrigerator", has_door_sensor=True)
    without = StorageUnit("v", "V", "refrigerator", has_door_sensor=False)
    silent = {
        unit_id: [
            Reading(unit_id, NOW - timedelta(minutes=m), temperature_c=4.5)
            for m in range(60, -1, -5)
        ]
        for unit_id in ("u", "v")
    }
    alerts = ColdChainMonitor(
        config=config, units={"u": with_sensor, "v": without},
        feed=ScriptedFeed(data=silent),
    ).evaluate(now=NOW)
    assert any(
        a.unit_id == "u" and a.kind == AlertKind.SENSOR_OFFLINE for a in alerts
    )
    assert not any(a.unit_id == "v" for a in alerts)


def test_r03_a_day_with_a_real_contiguous_gap_is_not_complete(config):
    """#3 Completeness was a COUNT. At a five-minute interval, 90% tolerates 28
    missing samples -- which may be one contiguous block. A logger unplugged
    from 20:40 to 23:00 during a service call produced 260 of 288 readings,
    reported `complete`, and the report declared 100% coverage with no findings
    for a day with 2h20m of no data."""
    day = date(2026, 8, 23)
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    gap_from = start + timedelta(hours=20, minutes=40)
    gap_to = start + timedelta(hours=23)
    readings = [
        Reading("u", start + timedelta(minutes=5 * i), temperature_c=4.5)
        for i in range(288)
        if not (gap_from <= start + timedelta(minutes=5 * i) < gap_to)
    ]
    assert len(readings) == 260                       # 90.3% of 288
    report = build_compliance_report(
        month="2026-08", units=[StorageUnit("u", "U", "refrigerator")],
        feed=ScriptedFeed(data={"u": readings}), config=config,
        start=day, end=day, now=NOW,
    )
    record = report.days[0]
    assert not record.complete
    assert record.longest_gap_minutes == pytest.approx(140, abs=6)
    assert report.coverage == 0.0
    assert any("140 minutes" in f for f in report.findings())


def test_r03_scattered_drops_are_still_a_monitored_day(config):
    """The other half. A cellular logger that drops the odd poll has not stopped
    monitoring the fridge, and flagging every such day trains the reader to
    ignore the list."""
    day = date(2026, 8, 23)
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    readings = [
        Reading("u", start + timedelta(minutes=5 * i), temperature_c=4.5)
        for i in range(288)
        if i % 11                                     # ~9% scattered
    ]
    assert len(readings) < 288 * 0.92
    report = build_compliance_report(
        month="2026-08", units=[StorageUnit("u", "U", "refrigerator")],
        feed=ScriptedFeed(data={"u": readings}), config=config,
        start=day, end=day, now=NOW,
    )
    assert report.days[0].complete


def test_r13_a_cycling_thermostat_still_escalates(config):
    """#6 The duration walk-back measured the current run of consecutive
    out-of-range readings and reset at any in-range sample. A failing thermostat
    cycling 11.5C / 6.0C reported "5 minutes" after eight cumulative hours above
    8C, and the 3am escalation never fired."""
    unit_obj = StorageUnit("u", "U", "refrigerator", has_door_sensor=False)
    readings = [
        Reading(
            "u", NOW - timedelta(minutes=5 * step),
            temperature_c=11.5 if step % 2 else 6.0,
        )
        for step in range(144, -1, -1)
    ]
    alerts = ColdChainMonitor(
        config=config, units={"u": unit_obj}, feed=ScriptedFeed(data={"u": readings})
    ).evaluate(now=NOW)
    sustained = [a for a in alerts if a.kind == AlertKind.SUSTAINED_EXCURSION]
    assert sustained, "cumulative exposure must escalate, not only a long run"
    assert sustained[0].duration_minutes > 120
    assert "physician_partner" in sustained[0].notify
    assert "cumulative" in sustained[0].detail


def test_r06_the_quarantine_label_makes_no_usability_claim(config, monitor):
    """Mutation testing found the previous test was a blocklist of four exact
    phrases, so any other wording passed. This asserts the label is ONLY the
    template lines plus data -- a whitelist."""
    record = _excursion(monitor)
    allowed_starts = (
        "*** DO NOT USE", "Unit:", "Excursion:", "Exposure:", "Duration:",
        "Doses affected:", "Manufacturer contacts:", "Affected lots:",
        "These doses MUST NOT be administered until the manufacturer has",
        "been contacted and a disposition recorded. Stability after an",
        "excursion is product-specific and only the manufacturer can say.",
        "",
    )
    for line in record.quarantine_label().splitlines():
        stripped = line.strip()
        if line.startswith("  "):
            continue                                  # a contact or a lot row
        assert line.startswith(allowed_starts), line


def test_r08_a_live_excursion_is_critical(config):
    """Mutation testing: nothing asserted the severity of a live excursion, so
    it could be downgraded to a warning and the suite stayed green."""
    unit_obj = StorageUnit("u", "U", "refrigerator", has_door_sensor=False)
    readings = [
        Reading("u", NOW - timedelta(minutes=m), temperature_c=14.0 if m <= 5 else 4.5)
        for m in range(60, -1, -5)
    ]
    alerts = ColdChainMonitor(
        config=config, units={"u": unit_obj}, feed=ScriptedFeed(data={"u": readings})
    ).evaluate(now=NOW)
    live = [a for a in alerts if a.kind == AlertKind.EXCURSION]
    assert live and live[0].severity == Severity.CRITICAL
    assert live[0].inventory_at_risk


def test_r07_the_gap_minutes_are_computed_from_the_timestamps(config):
    """Mutation testing: `gap_minutes` could be hard-wired to 0.0 and nothing
    noticed, because the only assertion was `any("unmonitored" in f)` -- which
    the string "(0 minutes unmonitored)" satisfies."""
    day = date(2026, 8, 23)
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    readings = [
        Reading("u", start + timedelta(minutes=5 * i), temperature_c=4.5)
        for i in range(288)
        if not (100 <= i < 124)                       # a 120-minute block
    ]
    report = build_compliance_report(
        month="2026-08", units=[StorageUnit("u", "U", "refrigerator")],
        feed=ScriptedFeed(data={"u": readings}), config=config,
        start=day, end=day, now=NOW,
    )
    assert report.days[0].gap_minutes == pytest.approx(115, abs=6)
    assert report.days[0].longest_gap_minutes == pytest.approx(115, abs=6)


def test_r02_the_open_door_duration_is_asserted_by_value(monitor):
    """Mutation testing: the original test asserted `duration >= 3` against a
    fixture whose newest reading was already 5 minutes stale, so deleting the
    walk-back entirely still passed."""
    door = [a for a in monitor.evaluate(now=NOW) if a.kind == AlertKind.DOOR_OPEN]
    assert door
    assert door[0].duration_minutes == pytest.approx(10, abs=1)


def test_the_quarantine_advisory_is_fixed_policy_text():
    """Mutation C6: the previous test blocked four phrases — "probably fine",
    "likely usable", "safe to use", "discard all" — and a novel fifth sentence
    walked straight past it. Allowlists beat blocklists: the label's prose is
    one named constant, pinned here verbatim, so changing what a quarantine
    label claims about potency requires changing this test."""
    assert DISPOSITION_ADVISORY == (
        "These doses MUST NOT be administered until the manufacturer has",
        "been contacted and a disposition recorded. Stability after an",
        "excursion is product-specific and only the manufacturer can say.",
    )


def test_the_label_prose_comes_only_from_that_constant(monitor):
    """The other half of the allowlist: nothing composes prose at the call
    site either. Every line of the label is a data line, a heading, or one of
    the advisory lines."""
    record = _excursion(monitor)
    label = record.quarantine_label()
    assert "\n".join(DISPOSITION_ADVISORY) in label

    data_prefixes = (
        "*** DO NOT USE", "Unit:", "Excursion:", "Exposure:", "Duration:",
        "Doses affected:", "Manufacturer contacts:", "Affected lots:", "  ",
    )
    prose = [
        line for line in label.splitlines()
        if line.strip() and not line.startswith(data_prefixes)
    ]
    assert prose == list(DISPOSITION_ADVISORY), prose
