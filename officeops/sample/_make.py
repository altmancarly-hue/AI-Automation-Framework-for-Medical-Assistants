"""Generate the bundled sample exports. Synthetic; no real patient data.

Deliberately messy: mixed date formats, a blank cell, a header spelled the way
one real EHR spells it. Sample data that is already clean tests nothing, and the
first thing a practice discovers is that their export is not clean.
"""

from __future__ import annotations

import csv
import datetime as dt
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = dt.date(2026, 8, 24)


def _write(name: str, headers: list[str], rows: list[list]) -> None:
    with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    tomorrow = TODAY + dt.timedelta(days=1)
    # -- schedule: note the vendor-ish headers and the mixed time formats
    _write(
        "schedule.csv",
        ["Chart No", "Patient Name", "Appt Date/Time", "Provider", "Appt Type",
         "Confirmation Status", "Home Phone", "Preferred Language", "DOB"],
        [
            ["10231", "Torres, Mila", f"{tomorrow:%m/%d/%Y} 08:20", "Dr. Alvarez",
             "Well child", "Confirmed", "847-555-0142", "English", "04/12/2024"],
            ["10877", "Nowak, Filip", f"{tomorrow:%m/%d/%Y} 08:40", "Dr. Alvarez",
             "Sick", "", "847-555-0198", "Polish", "09/03/2019"],
            ["11004", "Adeyemi, Chidi", f"{tomorrow:%m/%d/%Y} 09:00", "Dr. Osei",
             "Well child", "Not confirmed", "847-555-0177", "English", "01/22/2015"],
            ["11220", "Reyes, Sofia", f"{tomorrow:%m/%d/%Y} 09:20", "Dr. Osei",
             "Follow-up", "confirmed", "847-555-0110", "Spanish", "06/30/2021"],
            ["11455", "Reyes, Mateo", f"{tomorrow:%m/%d/%Y} 14:10", "Dr. Osei",
             "Sick", "no", "847-555-0110", "Spanish", "06/30/2021"],
            ["11901", "Petrov, Juno", f"{tomorrow:%m/%d/%Y} 15:40", "Dr. Alvarez",
             "Sports physical", "N", "847-555-0163", "English", "11/08/2011"],
            ["12007", "Kim, Aera", f"{TODAY:%m/%d/%Y} 10:00", "Dr. Osei",
             "Well child", "Y", "847-555-0121", "Korean", "07/19/2013"],
        ],
    )

    # -- roster for recall
    _write(
        "roster.csv",
        ["MRN", "Patient", "Date of Birth", "Last Well Visit", "Phone"],
        [
            ["10231", "Torres, Mila", "2024-04-12", "2026-02-10", "847-555-0142"],
            ["10877", "Nowak, Filip", "2019-09-03", "2024-11-02", "847-555-0198"],
            ["11004", "Adeyemi, Chidi", "2015-01-22", "", "847-555-0177"],
            ["11220", "Reyes, Sofia", "2021-06-30", "2025-07-01", "847-555-0110"],
            ["11901", "Petrov, Juno", "2011-11-08", "2026-06-14", "847-555-0163"],
            ["09115", "Bell, Marcus", "2003-02-19", "2021-03-04", "847-555-0155"],
        ],
    )

    # -- fridge log: a real excursion, plus a day with only one reading
    rows = []
    start = dt.datetime(2026, 8, 17, 8, 0)
    temps = {
        (0, 8): 4.1, (0, 16): 4.4, (1, 8): 4.0, (1, 16): 3.8,
        (2, 8): 4.2,  # only one reading on day 2 -- the log gap
        (3, 8): 4.3, (3, 16): 9.6, (4, 8): 10.4, (4, 16): 4.5,
        (5, 8): 4.2, (5, 16): 4.0, (6, 8): 4.1, (6, 16): 4.3,
    }
    for (day, hour), temp in sorted(temps.items()):
        when = start + dt.timedelta(days=day, hours=hour - 8)
        rows.append([when.strftime("%Y-%m-%d %H:%M"), "Fridge 1", f"{temp:.1f}"])
    _write("fridge_log.csv", ["Date/Time", "Storage Unit", "Temp (C)"], rows)

    # -- vaccine inventory
    _write(
        "vaccine_inventory.csv",
        ["Vaccine", "Lot Number", "Expiration Date", "Doses On Hand", "Funding Source"],
        [
            ["DTaP (Infanrix)", "AJ4471", "2026-08-01", "6", "VFC"],
            ["MMR (M-M-R II)", "TT0912", "2026-09-15", "12", "Private"],
            ["HPV (Gardasil 9)", "GG8820", "2026-10-02", "9", "VFC"],
            ["Influenza (Fluzone)", "FZ2266", "2027-06-30", "80", "Private"],
            ["Hib (ActHIB)", "AH1190", "2026-08-20", "3", "VFC"],
            ["Varicella (Varivax)", "VV5510", "2028-01-31", "15", "Private"],
        ],
    )

    # -- orders: labs and referrals
    _write(
        "orders.csv",
        ["MRN", "Patient", "Order Date", "Order Type", "Description", "Result Date",
         "Report Date", "Provider"],
        [
            ["10231", "Torres, Mila", "2026-08-18", "lab", "Hemoglobin", "2026-08-19", "", "Dr. Alvarez"],
            ["10877", "Nowak, Filip", "2026-07-02", "lab", "Lead level", "", "", "Dr. Alvarez"],
            ["11004", "Adeyemi, Chidi", "2026-06-11", "lab", "Lipid panel", "", "", "Dr. Osei"],
            ["11220", "Reyes, Sofia", "2026-03-14", "referral", "Allergy", "", "", "Dr. Osei"],
            ["11901", "Petrov, Juno", "2026-08-20", "referral", "Orthopedics", "", "", "Dr. Alvarez"],
            ["09115", "Bell, Marcus", "2026-05-02", "referral", "Dermatology", "", "2026-05-20", "Dr. Osei"],
        ],
    )

    # -- QC log with a hole
    rows = []
    for offset in range(20):
        day = TODAY - dt.timedelta(days=offset)
        for test in ("strep", "flu", "urine"):
            if test == "flu" and offset in (3, 4, 5):
                continue  # three days with no flu control
            result = "fail" if (test == "urine" and offset == 7) else "pass"
            rows.append([day.isoformat(), test, "control 1", result])
    _write("qc_log.csv", ["Date", "Test", "Control", "Result"], rows)

    # -- crash cart and sample closet
    _write(
        "crash_cart.csv",
        ["Item", "Lot", "Expiration", "Location", "Qty"],
        [
            ["Epinephrine 1:1000 ampoule", "EP2201", "2026-08-10", "Crash cart drawer 1", "4"],
            ["Albuterol nebules", "AB7742", "2026-09-30", "Crash cart drawer 2", "10"],
            ["Diphenhydramine injection", "DP9911", "2027-02-28", "Crash cart drawer 1", "2"],
            ["Oral glucose gel", "OG3310", "2026-08-30", "Crash cart drawer 3", "3"],
        ],
    )
    _write(
        "sample_closet.csv",
        ["Item", "Lot", "Expiration", "Location", "Qty"],
        [
            ["Amoxicillin 400mg/5mL suspension", "AM1120", "2026-07-15", "Sample closet", "6"],
            ["Fluticasone nasal spray", "FL8830", "2027-11-01", "Sample closet", "4"],
        ],
    )

    # -- credentials
    _write(
        "credentials.csv",
        ["Staff Member", "Title", "Credential", "License Number", "Expiration Date"],
        [
            ["Jess Alvarado", "Medical Assistant", "BLS/CPR", "AHA-88121", "2026-08-15"],
            ["Jess Alvarado", "Medical Assistant", "HIPAA training", "", "2026-11-30"],
            ["Dr. Ines Alvarez", "Physician", "IL Medical License", "036-119284", "2027-07-31"],
            ["Dr. Kwame Osei", "Physician", "IL Medical License", "036-224417", "2026-09-30"],
            ["Marta Silva", "LPN", "IL LPN License", "043-556120", "2026-10-15"],
            ["Marta Silva", "LPN", "OSHA bloodborne pathogens", "", "2026-08-01"],
            ["Dana Whitfield", "Front Desk", "HIPAA training", "", "2027-04-01"],
        ],
    )

    # -- standing orders
    _write(
        "standing_orders.csv",
        ["Staff Member", "Title", "Standing Order", "Signed Date", "Supervising Physician"],
        [
            ["Jess Alvarado", "Medical Assistant", "Vaccine administration per ACIP",
             "2026-01-15", "Dr. Ines Alvarez"],
            ["Jess Alvarado", "Medical Assistant", "Point-of-care strep testing",
             "2024-02-01", "Dr. Ines Alvarez"],
            ["Marta Silva", "LPN", "Vaccine administration per ACIP", "2026-03-02", ""],
            ["Dana Whitfield", "Front Desk", "Vision screening", "", "Dr. Kwame Osei"],
        ],
    )

    # -- records index
    _write(
        "records_index.csv",
        ["MRN", "Patient", "Date of Birth", "Last Activity"],
        [
            ["04412", "Hensley, Robert", "1988-05-02", "2011-09-14"],
            ["05219", "Okafor, Ada", "2001-03-30", "2019-06-01"],
            ["06330", "Lindqvist, Eva", "2005-08-11", "2023-04-19"],
            ["09115", "Bell, Marcus", "2003-02-19", "2021-03-04"],
        ],
    )

    # -- recipients + template
    _write(
        "recipients.csv",
        ["Patient Name", "Address", "City", "State", "Zip", "Last Well Visit"],
        [
            ["Nowak, Filip", "14 Kingsbury Ct", "Buffalo Grove", "IL", "60089", "2024-11-02"],
            ["Adeyemi, Chidi", "902 Weiland Rd", "Buffalo Grove", "IL", "60089", ""],
            ["Bell, Marcus", "77 Checker Dr", "Buffalo Grove", "IL", "60089", "2021-03-04"],
        ],
    )
    with open(os.path.join(HERE, "recall_letter.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "{{patient_name}}\n{{address}}\n{{city}}, {{state}} {{zip}}\n\n"
            "Dear parent or guardian of {{patient_name}},\n\n"
            "Our records show the last well-child visit was on {{last_well_visit}}.\n"
            "Please call the office to schedule the next one.\n\n"
            "North Suburban Pediatrics\n"
        )

    # -- visits and charges
    _write(
        "visits.csv",
        ["MRN", "Patient", "Date of Service", "Provider", "Visit Type"],
        [
            ["10231", "Torres, Mila", "2026-08-10", "Dr. Alvarez", "Well child"],
            ["10877", "Nowak, Filip", "2026-08-11", "Dr. Alvarez", "Sick"],
            ["11004", "Adeyemi, Chidi", "2026-08-12", "Dr. Osei", "Well child"],
            ["11220", "Reyes, Sofia", "2026-08-12", "Dr. Osei", "Sick"],
            ["11901", "Petrov, Juno", "2026-08-13", "Dr. Alvarez", "Sports physical"],
        ],
    )
    _write(
        "charges.csv",
        ["MRN", "Patient", "Date of Service", "CPT", "Amount"],
        [
            ["10231", "Torres, Mila", "2026-08-10", "99392", "185.00"],
            ["10877", "Nowak, Filip", "2026-08-11", "99213", "120.00"],
            ["11220", "Reyes, Sofia", "2026-08-12", "99213", "120.00"],
        ],
    )

    # -- denials
    rows = []
    for offset, (payer, code, desc, amount) in enumerate(
        [
            ("BCBS IL", "CO-197", "Precertification absent", "185.00"),
            ("BCBS IL", "CO-27", "Coverage terminated", "240.00"),
            ("Aetna", "CO-16", "Missing information", "60.00"),
            ("Aetna", "CO-16", "Missing information", "60.00"),
            ("Meridian", "CO-16", "Missing information", "45.00"),
            ("Meridian", "CO-97", "Bundled service", "32.00"),
            ("BCBS IL", "CO-16", "Missing information", "60.00"),
            ("Aetna", "CO-27", "Coverage terminated", "310.00"),
            ("Meridian", "CO-16", "Missing information", "45.00"),
            ("BCBS IL", "CO-16", "Missing information", "60.00"),
        ]
    ):
        rows.append(
            [(TODAY - dt.timedelta(days=offset * 5)).isoformat(), payer, code, desc, amount]
        )
    _write("denials.csv", ["Remit Date", "Payer", "Reason Code", "Reason Description", "Amount"], rows)
    print(f"sample data written to {HERE}")


if __name__ == "__main__":
    main()
