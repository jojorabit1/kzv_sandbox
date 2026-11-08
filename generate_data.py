"""Erzeugt einen synthetischen Uebungsdatensatz (angelehnt an zahnaerztliche Kassenabrechnung).

Alle Werte sind FIKTIV. Ausgabe:
  - sandbox.db                      (SQLite)
  - csv/<tabelle>.csv               (fuer Power BI)
  - fehler_loesung/fehler_log.csv   (Protokoll der eingebauten Fehler - erst spaeter ansehen!)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PRAXEN = 300
N_FAELLE = 50_000
ERR_RATE = 0.004  # pro Fehlerart und Tabelle -> 8 Fehlerarten ~ 2 % Gesamtfehlerquote

BASE = Path(__file__).parent
CSV_DIR = BASE / "csv"
LOSUNG_DIR = BASE / "fehler_loesung"
DB_PATH = BASE / "sandbox.db"

QUARTALE = [f"{y}Q{q}" for y in (2024, 2025) for q in (1, 2, 3, 4)]
BEREICHE = ["KCH", "KFO", "PAR", "ZE", "KB"]
KASSENARTEN = ["AOK", "BKK", "IKK", "vdek", "KBS"]

# (bema_nr, bezeichnung, leistungsbereich, punkte) - alle Nummern und Punkte fiktiv
BEMA_KATALOG = [
    ("K01", "Beratung", "KCH", 8), ("K02", "Eingehende Untersuchung", "KCH", 18),
    ("K03", "Zahnsteinentfernung", "KCH", 10), ("K04", "Roentgen Einzelaufnahme", "KCH", 9),
    ("K05", "Fuellung einflaechig", "KCH", 40), ("K06", "Fuellung zweiflaechig", "KCH", 55),
    ("K07", "Fuellung dreiflaechig", "KCH", 70), ("K08", "Vitalitaetspruefung", "KCH", 6),
    ("K09", "Lokale Anaesthesie", "KCH", 12), ("K10", "Leitungsanaesthesie", "KCH", 20),
    ("K11", "Extraktion einwurzelig", "KCH", 30), ("K12", "Extraktion mehrwurzelig", "KCH", 45),
    ("K13", "Wurzelkanalaufbereitung", "KCH", 65), ("K14", "Wurzelfuellung", "KCH", 50),
    ("K15", "Pulpotomie", "KCH", 28), ("K16", "Fissurenversiegelung", "KCH", 22),
    ("K17", "Fluoridierung", "KCH", 7), ("K18", "Kontrolle nach Eingriff", "KCH", 5),
    ("K19", "Hypersensibilitaet Behandlung", "KCH", 11), ("K20", "Notfallbehandlung", "KCH", 35),
    ("O01", "KFO Beratung", "KFO", 15), ("O02", "KFO Befundaufnahme", "KFO", 60),
    ("O03", "Behandlungsplan KFO", "KFO", 45), ("O04", "Aktive Behandlung Quartal", "KFO", 90),
    ("O05", "Retainer Kontrolle", "KFO", 20), ("O06", "Bracketkontrolle", "KFO", 25),
    ("P01", "Parodontaler Befund", "PAR", 38), ("P02", "Geschlossene Kuerettage", "PAR", 55),
    ("P03", "Offene Kuerettage", "PAR", 95), ("P04", "Nachkontrolle PAR", "PAR", 20),
    ("P05", "Mundhygieneunterweisung", "PAR", 14),
    ("Z01", "Krone Befund", "ZE", 80), ("Z02", "Abformung", "ZE", 35),
    ("Z03", "Provisorium", "ZE", 42), ("Z04", "Eingliederung Krone", "ZE", 70),
    ("Z05", "Brueckenglied", "ZE", 60),
    ("B01", "Aufbissbehelf", "KB", 75), ("B02", "Schienenkontrolle", "KB", 18),
    ("B03", "Funktionsanalyse", "KB", 50), ("B04", "Einschleifen", "KB", 24),
]

# Leistungsmix je Fachrichtung (Reihenfolge = BEREICHE)
BEREICH_P = {
    "Allgemein":    [0.70, 0.00, 0.10, 0.12, 0.08],
    "KFO":          [0.25, 0.70, 0.00, 0.00, 0.05],
    "Oralchirurgie": [0.80, 0.00, 0.05, 0.05, 0.10],
}
ALTER_P = {  # 0-17, 18-39, 40-64, 65+
    "Allgemein":    [0.15, 0.25, 0.35, 0.25],
    "KFO":          [0.85, 0.13, 0.02, 0.00],
    "Oralchirurgie": [0.05, 0.25, 0.40, 0.30],
}


def saison_gewichte() -> dict:
    """Relative Fallhaeufigkeit je Quartal (wird anschliessend normiert).

    Gibt {"2024Q1": gewicht, ..., "2025Q4": gewicht} zurueck.
    """
    basis = {1: 100, 2: 105, 3: 88, 4: 112}
    jahresfaktor = {2024: 1.00, 2025: 1.03}
    return {
        f"{jahr}Q{q}": basis[q] * jahresfaktor[jahr]
        for jahr in jahresfaktor
        for q in range(1,5)
    }


def quartal_grenzen():
    starts, tage = {}, {}
    for q in QUARTALE:
        p = pd.Period(q, freq="Q")
        s, e = p.start_time.normalize(), p.end_time.normalize()
        starts[q] = s
        tage[q] = (e - s).days + 1
    return starts, tage


def gen_praxen(rng):
    ids = [f"P{i:04d}" for i in range(1, N_PRAXEN + 1)]
    return pd.DataFrame({
        "praxis_id": ids,
        "bezirk": rng.choice([f"Bezirk_{i:02d}" for i in range(1, 9)], N_PRAXEN),
        "fachrichtung": rng.choice(list(BEREICH_P), N_PRAXEN, p=[0.82, 0.10, 0.08]),
        "gruendungsjahr": rng.integers(1975, 2023, N_PRAXEN),
        "anzahl_zahnaerzte": rng.choice([1, 2, 3, 4, 5], N_PRAXEN, p=[0.50, 0.28, 0.12, 0.07, 0.03]),
    })


def gen_katalog():
    return pd.DataFrame(BEMA_KATALOG, columns=["bema_nr", "bezeichnung", "leistungsbereich", "punkte"])


def gen_faelle(rng, praxen):
    w = rng.lognormal(0, 0.6, N_PRAXEN)
    praxis = rng.choice(praxen["praxis_id"], N_FAELLE, p=w / w.sum())
    sw = saison_gewichte()
    qw = np.array([sw[q] for q in QUARTALE], dtype=float)
    fachr = praxen.set_index("praxis_id").loc[praxis, "fachrichtung"].to_numpy()

    alter = np.empty(N_FAELLE, dtype=object)
    for f, p in ALTER_P.items():
        m = fachr == f
        alter[m] = rng.choice(["0-17", "18-39", "40-64", "65+"], m.sum(), p=p)

    faelle = pd.DataFrame({
        "fall_id": [f"F{i:06d}" for i in range(1, N_FAELLE + 1)],
        "praxis_id": praxis,
        "quartal": rng.choice(QUARTALE, N_FAELLE, p=qw / qw.sum()),
        "kassenart": rng.choice(KASSENARTEN, N_FAELLE, p=[0.36, 0.18, 0.07, 0.34, 0.05]),
        "altersgruppe": alter,
        "geschlecht": rng.choice(["m", "w", "d"], N_FAELLE, p=[0.48, 0.51, 0.01]),
    })
    return faelle, fachr


def gen_leistungen(rng, faelle, fachr, katalog):
    n_l = rng.choice([1, 2, 3, 4, 5, 6], N_FAELLE, p=[0.28, 0.28, 0.20, 0.13, 0.07, 0.04])
    idx = np.repeat(np.arange(N_FAELLE), n_l)
    total = len(idx)
    fachr_l = fachr[idx]

    bereich = np.empty(total, dtype=object)
    for f, p in BEREICH_P.items():
        m = fachr_l == f
        bereich[m] = rng.choice(BEREICHE, m.sum(), p=p)

    bema = np.empty(total, dtype=object)
    for b in BEREICHE:
        m = bereich == b
        if not m.any():
            continue
        nrs = katalog.loc[katalog["leistungsbereich"] == b, "bema_nr"].to_numpy()
        w = 1.0 / np.arange(1, len(nrs) + 1)  # haeufige Positionen vorne
        bema[m] = rng.choice(nrs, m.sum(), p=w / w.sum())

    starts, tage = quartal_grenzen()
    q = faelle["quartal"].to_numpy()[idx]
    start = np.array([starts[x] for x in q], dtype="datetime64[D]")
    ndays = np.array([tage[x] for x in q])
    datum = start + (rng.random(total) * ndays).astype(int).astype("timedelta64[D]")

    return pd.DataFrame({
        "leistung_id": [f"L{i:07d}" for i in range(1, total + 1)],
        "fall_id": faelle["fall_id"].to_numpy()[idx],
        "bema_nr": bema,
        "anzahl": rng.choice([1, 2, 3, 4], total, p=[0.70, 0.20, 0.07, 0.03]),
        "leistungsdatum": pd.to_datetime(datum),
    })


def gen_punktwerte():
    basis = {"KCH": 1.20, "KFO": 1.15, "PAR": 1.10, "ZE": 1.30, "KB": 1.25}
    faktor = {"AOK": 1.0, "BKK": 1.01, "IKK": 0.995, "vdek": 1.015, "KBS": 0.99}
    rows = [(b, k, q, round(basis[b] * faktor[k] * (1 + 0.004 * i), 4))
            for b in BEREICHE for k in KASSENARTEN for i, q in enumerate(QUARTALE)]
    return pd.DataFrame(rows, columns=["leistungsbereich", "kassenart", "quartal", "punktwert_eur"])


def inject_errors(rng, faelle, leistungen):
    """Baut Fehler ein und liefert (faelle, leistungen, log)."""
    log = []
    fq = dict(zip(faelle["fall_id"], faelle["quartal"]))  # echtes Quartal je Fall (vor Formatfehlern)
    starts, tage = quartal_grenzen()

    # ---- leistungen: 6 Fehlerarten auf disjunkten Zeilen ----
    n = int(len(leistungen) * ERR_RATE)
    perm = rng.permutation(len(leistungen))
    sl = {name: perm[i * n:(i + 1) * n] for i, name in enumerate(
        ["null_fk", "neg", "outlier", "bema", "datum", "dup"])}
    leistungen = leistungen.astype({"anzahl": "int64"}).copy()

    def put(table, key_col, rows, col, art, new_vals, df):
        for pos, nv in zip(rows, new_vals):
            old = df.iat[pos, df.columns.get_loc(col)]
            log.append((table, df.iat[pos, df.columns.get_loc(key_col)], col, art, old, nv))
            df.iat[pos, df.columns.get_loc(col)] = nv

    put("leistungen", "leistung_id", sl["null_fk"], "fall_id", "NULL-Fremdschluessel",
        [None] * n, leistungen)
    put("leistungen", "leistung_id", sl["neg"], "anzahl", "negative Anzahl",
        -rng.integers(1, 4, n), leistungen)
    put("leistungen", "leistung_id", sl["outlier"], "anzahl", "Ausreisser",
        rng.integers(50, 300, n), leistungen)
    put("leistungen", "leistung_id", sl["bema"], "bema_nr", "ungueltige bema_nr",
        rng.choice(["X99", "K00", "Q01", "999", "k05 "], n), leistungen)

    # Datum ausserhalb des Quartals (Verschiebung >= 100 Tage > Quartalslaenge)
    shift = rng.integers(100, 200, n) * rng.choice([-1, 1], n)
    old_dates = leistungen.iloc[sl["datum"]]["leistungsdatum"]
    new_dates = old_dates + pd.to_timedelta(shift, unit="D")
    for pos, od, nd in zip(sl["datum"], old_dates, new_dates):
        log.append(("leistungen", leistungen.iat[pos, 0], "leistungsdatum",
                    "Datum ausserhalb Quartal", od.date().isoformat(), nd.date().isoformat()))
        leistungen.iat[pos, leistungen.columns.get_loc("leistungsdatum")] = nd

    # Duplikate: Kopie mit NEUER leistung_id (Original unveraendert)
    dups = leistungen.iloc[sl["dup"]].copy()
    next_id = len(leistungen) + 1
    dups["leistung_id"] = [f"L{next_id + i:07d}" for i in range(n)]
    for orig, new in zip(leistungen.iloc[sl["dup"]]["leistung_id"], dups["leistung_id"]):
        log.append(("leistungen", new, "(ganze Zeile)", "Duplikat", orig, new))
    leistungen = pd.concat([leistungen, dups], ignore_index=True)
    leistungen = leistungen.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ---- faelle: 2 Fehlerarten ----
    nf = int(len(faelle) * ERR_RATE)
    permf = rng.permutation(len(faelle))
    rows_fk, rows_q = permf[:nf], permf[nf:2 * nf]
    faelle = faelle.copy()
    put("faelle", "fall_id", rows_fk, "praxis_id", "NULL-Fremdschluessel", [None] * nf, faelle)

    def fmt(q):
        y, k = q[:4], q[-1]
        return rng.choice([f"{y}-Q{k}", f"Q{k}/{y}", f"{y} Q{k}", f"{y[2:]}Q{k}"])

    put("faelle", "fall_id", rows_q, "quartal", "uneinheitliches Quartalsformat",
        [fmt(faelle.iat[r, faelle.columns.get_loc("quartal")]) for r in rows_q], faelle)

    log_df = pd.DataFrame(log, columns=["tabelle", "primaerschluessel", "spalte",
                                        "fehlerart", "originalwert", "fehlerwert"])
    return faelle, leistungen, log_df


def main():
    rng = np.random.default_rng(SEED)
    praxen = gen_praxen(rng)
    katalog = gen_katalog()
    faelle, fachr = gen_faelle(rng, praxen)
    leistungen = gen_leistungen(rng, faelle, fachr, katalog)
    punktwerte = gen_punktwerte()

    faelle, leistungen, log = inject_errors(rng, faelle, leistungen)
    leistungen["leistungsdatum"] = leistungen["leistungsdatum"].dt.strftime("%Y-%m-%d")

    tabellen = {"praxen": praxen, "bema_katalog": katalog, "faelle": faelle,
                "leistungen": leistungen, "punktwerte": punktwerte}

    CSV_DIR.mkdir(exist_ok=True)
    LOSUNG_DIR.mkdir(exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        for name, df in tabellen.items():
            df.to_sql(name, con, index=False)
            df.to_csv(CSV_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    log.to_csv(LOSUNG_DIR / "fehler_log.csv", index=False, encoding="utf-8-sig")

    for name, df in tabellen.items():
        print(f"{name:14s} {len(df):>8,} Zeilen")
    gesamt = sum(len(tabellen[t]) for t in ("faelle", "leistungen"))
    print(f"Fehler im Log: {len(log):,} ({len(log) / gesamt:.2%} der Fall-/Leistungszeilen)")


if __name__ == "__main__":
    main()
