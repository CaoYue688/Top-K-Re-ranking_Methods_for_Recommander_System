[中文](RESULTS.md) | [English](RESULTS.en.md) | [Deutsch](RESULTS.de.md)

# Historische Baseline-Ergebnisse (nicht für die finale Masterarbeit)

> Diese Seite dokumentiert einen frühen Lauf mit nur einem Seed und einer überholten Evaluationsdefinition. Für die finale Masterarbeit sind [RESEARCH_RESULTS.de.md](RESEARCH_RESULTS.de.md) und
> `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/` maßgeblich; die Werte dieser Seite dürfen nicht als finale Ergebnisse zitiert werden.

Laufkonfiguration: Zufallsseed `2026`, Embedding-Dimension `64` und `1 Epoche` auf dem vollständigen Trainingsumfang. Dies sind die tatsächlich gespeicherten Ausgaben dieses historischen Laufs. Für ein formales Experiment ist das Produktionsverfahren in [README.de.md](README.de.md) mit der dokumentierten Konfiguration zu verwenden; Checkpoints dürfen ausschließlich anhand von Validation-Metriken ausgewählt werden.

## Daten

| Bestandteil | Wert |
|---|---:|
| Rohinteraktionen | 20.000.263 |
| Interaktionen nach iterativem 5-Core | 19.984.024 |
| Nutzer | 138.493 |
| Items | 18.345 |
| Training | 15.932.772 (79,73 %) |
| Validierung | 1.940.306 (9,71 %) |
| Test | 2.110.946 (10,56 %) |
| Evaluationskandidaten | 1 Positives + 100 Negative je Nutzer |

## Rankingmetriken auf dem gesampelten Testdatensatz

| Modell | HR@10 | NDCG@10 | MRR@10 | HR@20 | NDCG@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|
| BPR-MF | 0,7965 | 0,5063 | 0,4159 | 0,9178 | 0,5372 | 0,4246 |

Diese Metriken wurden auf einer festen Menge von 101 Kandidaten berechnet und dürfen nicht direkt mit Vollkatalog-Rankingmetriken verglichen werden.

## Listenqualität nach dem Re-Ranking von Top 100 auf Top 20

Die Re-Ranking-Gewichte sind Relevanz `0,70`, Kalibrierung `0,15` und Diversität `0,15`. Calibration ist die Jensen-Shannon-Ähnlichkeit; ILD ist die mittlere paarweise Kosinusdistanz zwischen Genre-Multi-Hot-Vektoren.

| Modell | Calibration | ILD |
|---|---:|---:|
| BPR-MF | 0,8738 | 0,7159 |
