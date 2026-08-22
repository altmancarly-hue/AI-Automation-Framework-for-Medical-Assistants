# I-08 — Vaccine Cold Chain Telemetry

**The initiative with no model in it, on purpose.**

> README I-08: *"Include this initiative in an 'AI program' specifically to
> demonstrate the discipline of not using AI where it does not belong. A
> proposal that reaches for a language model in every section is not a
> technology strategy; it is a shopping list."*

A pediatric practice holds roughly $40,000–$60,000 of vaccine inventory in two
refrigerators and a freezer. The failure mode is not dramatic: a door left ajar
on a Friday afternoon, a compressor that dies at 21:00 on a Saturday, a
thermometer nobody has calibrated since 2021. The loss is discovered on Monday,
and by then the question is not "is this vaccine cold" but "can we prove it
was."

## How much of this is a model?

**Zero.** `make lint` asserts that no file in `modules/coldchain` imports
`nsp_core.llm`, `openai` or `anthropic`, and `test_there_is_no_model_anywhere`
walks the AST of every module in the package to make the same assertion in the
test suite. This is not an oversight to be corrected later. Temperature is a
number; a threshold is a comparison; a duration is arithmetic. Every one of
those is a case where hard constraint 5 — *deterministic logic is implemented in
Python* — says a model is the wrong tool.

## Files

| File | Responsibility |
| --- | --- |
| `monitor.py` | `Reading`, `StorageUnit`, `ColdChainConfig`, `ColdChainMonitor`. Evaluates the live state and decides who hears about it. |
| `excursion.py` | `open_excursion()`, `VaccineLot`, `LotDisposition`, `build_compliance_report()`. The record that survives the event. |
| `fixtures.py` | Four synthetic units and named scenarios (`all_quiet`, `monday_morning`, …). |
| `demo.py` | `make demo-i08`. |

Configuration lives in `config/coldchain.yaml`: unit types, ranges, sustained
thresholds, re-notify intervals, routing by role, calibration cadence. Nothing
in this module hard-codes 2–8 °C.

## The two things that are easy to get wrong

**1. `evaluate()` reports state; `notifications()` decides what is sent.**

`evaluate(now=…)` answers *"what is true about these units right now"* — every
condition, every poll. It is a state read and it must not lie about persistence.
`notifications(now=…)` is the thing with a memory: a condition is announced when
it first appears and then no more often than its configured re-notify interval,
and its entry is dropped when the condition clears, so a recurrence is news
again. The audit row is written there, not in `evaluate()`.

The difference is not cosmetic. Polling four units every five minutes for a day:

```
scenario 'monday_morning': 288 polls over 24 h, four units
   conditions true (evaluate) : 1231
   alerts sent (notifications):   36
```

An alert that fires on everything fails the same way as no alert. Route the
output of `notifications()`. Do not route the output of `evaluate()`.

**2. Silence is measured from the last reading that carried a temperature.**

A sensor whose battery is dying can keep sending heartbeats with a null
temperature field. Measuring "time since we last heard from the unit" against
those heartbeats reports a healthy unit that has told you nothing about its
contents for nine hours. `_last_temperature_reading()` walks back to the last
reading that actually carried a value, and the offline check is written against
that; the heartbeat only proves the radio works. The same applies to the door
channel: `has_door_sensor` gates a separate "the door channel has gone quiet"
alert, and the open-door clock steps over `None` samples rather than being reset
by them.

Both are instances of the standing rule: **a safety threshold must never widen
in response to missing data.**

## Excursions and the disposition question

`open_excursion()` produces an `ExcursionRecord` with the temperature series,
the affected lots, the duration, and a quarantine label. It does **not** decide
whether the vaccine is usable. That call belongs to the manufacturer, reached by
telephone, and `test_the_module_never_decides_whether_exposed_vaccine_is_usable`
asserts that no wording in the module offers an opinion about potency — a
reassuring sentence in a quarantine label is a clinical judgement wearing the
costume of a status message.

`build_compliance_report()` answers the auditor's question. `DailyRecord.complete`
is a **distribution** test, not a count: a day is complete when the longest gap
between readings is under `max_gap_minutes`, because 280 readings clustered in
the morning and nothing after noon is not a complete day, and a count-based
threshold reads it as one. The report declares its own gaps rather than
presenting a total.

## Commands

```bash
make test-coldchain    # 39 tests
make demo-i08          # four units, a weekend excursion, the compliance report
make lint              # includes: NO MODEL AT ALL in the cold chain
python3 -m modules.coldchain.demo
```

## Wiring

```python
from modules.coldchain import ColdChainConfig, ColdChainMonitor
from modules.coldchain.fixtures import UNITS, build_feed

config = ColdChainConfig.load()          # config/coldchain.yaml
config.review["owner"] = "dr_alvarez"    # refuses to run unowned

monitor = ColdChainMonitor(
    config=config,
    units={u.unit_id: u for u in UNITS},
    feed=build_feed("monday_morning"),   # swap for a real SensorFeed
    audit=audit_log,
)

state = monitor.evaluate(now=now)                          # what is true
for alert in monitor.notifications(now=now, alerts=state): # what to send
    page(alert.notify, alert.summary)
```

`SensorFeed` is the integration seam. `ScriptedFeed` is the shipped test double;
a real deployment implements `readings(unit_id, since, until)` against whatever
the sensor vendor provides. Nothing above that interface knows the vendor.
