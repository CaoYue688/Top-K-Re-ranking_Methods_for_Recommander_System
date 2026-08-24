[中文](RESEARCH_RESULTS.md) | [English](RESEARCH_RESULTS.en.md) | [Deutsch](RESEARCH_RESULTS.de.md)

# Finale Experimentergebnisse der Masterarbeit (auditiert)

Datenlabel: `thesis_pos4_neg2_traincore5_dataseed2026`. Die Auditprüfung bestätigte 686 vollständige Konfigurationszeilen, keine doppelten zusammengesetzten Schlüssel, keine fehlenden Kernmetriken für das jeweils gültige K und identische Lambda=0-Baselines für alle drei Methoden.

## Primäre Baseline (Test, Mittelwert über drei Seeds)

| Metrik | Wert |
|---|---:|
| Recall@10 | 0,060897 |
| NDCG@10 | 0,058480 |
| ILD@10 | 0,669365 |
| Calibration@10 | 0,819102 |
| Subtopic Recall@10 | 0,862119 |
| Catalog Coverage@10 | 0,482772 |
| Exposure Gini@10 | 0,962414 |
| Long-tail share@10 | 0,016793 |
| Candidate Recall@100 | 0,276126 |

## Methodenübergreifende Auswahl nach Validation-Budget

Für alle vier Budgets wird MMR ausgewählt:

| Validation-Budget | Lambda | Relative Änderung Test-NDCG | Relative Änderung Test-ILD |
|---:|---:|---:|---:|
| 1 % | 0,25 | -0,84 % | +8,85 % |
| 3 % | 0,35 | -2,49 % | +13,67 % |
| 5 % | 0,40 | -3,80 % | +16,30 % |
| 10 % | 0,55 | -8,92 % | +22,84 % |

Am zentralen 5-%-Arbeitspunkt betragen die Mittelwerte NDCG@10=0,056257, ILD@10=0,778464, Calibration@10=0,841053, Catalog Coverage@10=0,497370 und Long-tail share@10=0,020364. Die konservativen Vorzeichentests und Holm-korrigierten Richtungsaussagen sind signifikant; die mittleren Cohen-dz-Werte für NDCG und ILD liegen bei ungefähr -0,049 beziehungsweise 1,386.

## Methodenunterschiede innerhalb des 5-%-Budgets

- **MMR Lambda=0,40**: höchste paarweise Genre-ILD, +16,30 %; NDCG -3,80 %.
- **xQuAD Lambda=0,80**: kleinerer ILD-Gewinn, aber profilgewichteter Subtopic Recall von ungefähr 0,943; am besten für die Abdeckung von Interessenaspekten geeignet.
- **Kalibrierung Lambda=0,85**: Calibration von ungefähr 0,899 und geringster NDCG-Verlust; am besten zur Anpassung an historische Interessenanteile geeignet.

Es gibt daher keinen zielunabhängigen „besten Algorithmus“: MMR, xQuAD und Kalibrierung optimieren unterschiedliche Diversitätskonstrukte.

## Robustheit

MMR unter Seed 2026 und dem 5-%-Validation-Budget:

| Einstellung | Relative Änderung Test-NDCG | Relative Änderung Test-ILD |
|---|---:|---:|
| N=50, K=10 | -2,84 % | +13,44 % |
| N=100, K=10 | -4,06 % | +16,34 % |
| N=200, K=10 | -2,45 % | +13,02 % |
| N=100, K=5 | -4,16 % | +26,05 % |
| N=100, K=20 | -3,43 % | +10,20 % |

Auf dem tag-vollständigen Kandidatenpool erzielt Genre-MMR einen Feature-ILD-Gewinn von 16,32 %, Tag-Genome-SVD64-MMR dagegen 13,47 %; beide verursachen ungefähr 4,1 % Test-NDCG-Verlust. Die qualitative Schlussfolgerung ist stabil, die Effektgröße hängt jedoch vom Merkmalsraum ab.

## Nutzergruppen- und Katalogeffekte

- In allen drei Aktivitätsgruppen liegen die ILD-Gewinne bei ungefähr 16,1-16,5 %.
- Focused-Profile haben einen höheren ILD-Gewinn (18,27 %), aber auch den höchsten NDCG-Verlust (5,16 %).
- Broad-Profile erreichen 14,77 % ILD-Gewinn bei 3,04 % NDCG-Verlust.
- Catalog Coverage steigt von 48,28 % auf 49,74 %, Long-tail share von 1,68 % auf 2,04 %.
- Exposure Gini verändert sich nur von 0,962414 auf 0,962382. Lokale Listendiversität kann daher kein globales Ziel für faire Exposition ersetzen.

## Prüfbare Dateien

- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/all_thesis_results.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/validation_budget_selections.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/test_budget_results.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/experiment_manifest.json`

Die frühere Datei `RESULTS.md` und ältere Ausgaben sind historische Baselines und ersetzen diese finalen auditierten Ergebnisse nicht.
