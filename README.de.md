[English](README.md) | [Deutsch](README.de.md) | [中文](README.zh-CN.md)

# Diversitätsorientiertes Re-Ranking für Empfehlungssysteme

Dieses Repository enthält die reproduzierbare Implementierung der Masterarbeit zum diversitätsorientierten Top-K-Re-Ranking auf MovieLens 20M. Verglichen werden MMR, xQuAD und Kalibrierung unter expliziten Budgets für den zulässigen Genauigkeitsverlust.

> Die Ergebnisse der Masterarbeit werden mit `recsys20m.thesis_pipeline` erzeugt. Die ältere `recsys20m.pipeline` bleibt ausschließlich als historischer technischer Basislauf erhalten und bildet nicht die Evidenzgrundlage der Arbeit.

## Implementierter Umfang

- chronologische Aufteilung in Training, Validierung und Test mit iterativem 5-Core ausschließlich auf den Trainingsdaten,
- BPR-MF-Training mit drei festgelegten Zufallsseeds,
- Aufbau eines Tag-Genome-Merkmalsraums mittels 64-dimensionaler SVD,
- deterministische Erzeugung der Top-N-Kandidaten,
- Re-Ranking mit MMR, xQuAD und Kalibrierung,
- Evaluation von Accuracy, Diversität, Kalibrierung, Popularitätsbias und Laufzeit,
- Accuracy-Loss-Budgets von 1 %, 3 %, 5 % und 10 %,
- Robustheitsanalysen für unterschiedliche Kandidaten- und Empfehlungslistenlängen,
- gepaarte Bootstrap-Konfidenzintervalle und Holm-korrigierte konfirmatorische Tests.

## Finale Experimentkonfiguration

| Bestandteil | Einstellung |
|---|---|
| Datensatz | MovieLens 20M |
| Positives Feedback | Bewertung >= 4,0 |
| Explizit negatives Feedback | Bewertung <= 2,0 |
| Nutzer-/Item-Filterung | iteratives 5-Core, ausschließlich auf Trainingsdaten angepasst |
| Aufteilung | chronologisch in Training/Validierung/Test |
| Trainingsseeds | 2026, 2027, 2028 |
| BPR-MF | 10 Epochen, 64 latente Faktoren, Batch-Größe 8192 |
| Negative Sampling | 50 % explizite Negative, sofern verfügbar |
| Merkmalsraum | MovieLens Tag Genome, SVD-Dimension 64 |
| Retrieval-Pool | intern Top 200; primäre Evaluation mit N = 100 |
| Empfehlungslistenlänge | primär K = 10 |
| Re-Ranker | MMR, xQuAD, Kalibrierung |
| Lambda-Raster | primär Schrittweite 0,05; Robustheit 0,10 |
| Accuracy-Loss-Budgets | 1 %, 3 %, 5 %, 10 % |
| Bootstrap | 200 Stichproben primär, 100 für Robustheit |
| Robustheitsraster | N in {50, 100, 200}; K in {5, 10, 20} |
| Rechenziel | standardmäßig CUDA-GPU; CPU möglich, aber deutlich langsamer |

Das vollständige Verfahren und die Metrikdefinitionen stehen in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## Voraussetzungen und Installation

Benötigt werden:

- Python 3.11 oder neuer,
- der offizielle MovieLens-20M-Datensatz,
- für die vorgesehene Laufzeit eine CUDA-fähige GPU; eine Ausführung auf der CPU ist möglich.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Falls die automatisch installierte PyTorch-Version nicht zur lokalen CUDA-Umgebung passt, muss zuerst die für das jeweilige System geeignete PyTorch-Version installiert und anschließend die editierbare Installation erneut ausgeführt werden. `Pillow` ist als Abhängigkeit angegeben, weil das öffentliche Skript zur Abbildungserzeugung die Bibliothek importiert.

## Datenvorbereitung

MovieLens 20M ist von der offiziellen GroupLens-Quelle herunterzuladen und zu entpacken. Der Datensatz wird in diesem Repository nicht weitergegeben.

Die Dateien werden unter folgendem Pfad erwartet:

```text
data/raw/ml-20m/
├── ratings.csv
├── movies.csv
├── genome-scores.csv
└── genome-tags.csv
```

Erwartete Prüfsumme des Archivs:

```text
SHA256 ml-20m.zip
96F1322B342E074A2B251BB4C1E1990AB58082C228A430029A258A4E4393F51A
```

## Reproduktion der finalen Experimente

Aus dem Wurzelverzeichnis des Repositorys unter Windows:

```powershell
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root .
```

Ausführung ausschließlich auf der CPU:

```powershell
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root . --device cpu
```

Für einen vollständigen Neulauf, bei dem zwischengespeicherte Resultate ersetzt werden, ist `--force` zu ergänzen.

Nach Abschluss des Experiments werden die aggregierten Resultate zusammengefasst und die Abbildungen neu erzeugt:

```powershell
.\.venv\Scripts\python.exe scriptssummarize_thesis_results.py
.\.venv\Scripts\python.exe scriptsgenerate_thesis_figures.py
```

Unter Linux/macOS ist `.\.venv\Scripts\python.exe` durch `.venv/bin/python` zu ersetzen.

## Zentrale Ausgabedateien

Die finalen aggregierten Resultate werden unter `outputs/thesis/aggregate/` abgelegt.

| Ausgabe | Zweck |
|---|---|
| `summary_seed_level.csv` | aggregierte Metriken je Seed |
| `budget_selection_seed_level.csv` | ausgewählte Konfigurationen je Accuracy-Loss-Budget |
| `accuracy_comparison_seed_level.csv` | Accuracy-Vergleich mit der BPR-MF-Baseline |
| `robustness_summary_seed_level.csv` | Robustheitsergebnisse für verschiedene N und K |
| `robustness_budget_selection_seed_level.csv` | Robustheitsauswahl unter den Budgets |
| `runtime_seed_level.csv` | gemessene Laufzeiten |
| `figures/` | erzeugte Abbildungen der Masterarbeit |

Generierte Daten, Checkpoints und Experimentausgaben werden wegen ihrer Größe und Reproduzierbarkeit nicht in Git gespeichert.

## Tests

Vollständige öffentliche Testsuite unter Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Forschungsdokumentation

- Experimentprotokoll: [Deutsch](RESEARCH_PROTOCOL.de.md) | [English](RESEARCH_PROTOCOL.en.md) | [中文](RESEARCH_PROTOCOL.md)
- Auditierte finale Ergebnisse: [Deutsch](RESEARCH_RESULTS.de.md) | [English](RESEARCH_RESULTS.en.md) | [中文](RESEARCH_RESULTS.md)
- [RESULTS.md](RESULTS.md): Ausgabe des historischen Basislaufs; sie darf nicht mit den finalen Ergebnissen verwechselt werden.

## Repository-Struktur

```text
src/recsys20m/                 Kernimplementierung
scripts/                       reproduzierbare Zusammenfassung und Abbildungserzeugung
tests/                         öffentliche automatisierte Tests
RESEARCH_PROTOCOL*.md          Experimentprotokoll auf Chinesisch, Englisch und Deutsch
RESEARCH_RESULTS*.md           finaler Ergebnisbericht auf Chinesisch, Englisch und Deutsch
pyproject.toml                 Paketmetadaten und kanonische Abhängigkeiten
requirements.txt               praktische direkte Abhängigkeitsliste
LICENSE                        MIT-Lizenz
```
## Historischer Basislauf

Der frühere technische Basislauf kann weiterhin folgendermaßen ausgeführt werden:

```powershell
.\.venv\Scripts\python.exe -m recsys20m.pipeline --root .
```

Dieser Pfad verwendet eine kleinere Konfiguration, darunter drei Trainingsepochen, eine gesampelte Evaluation und Top-20-Ausgaben. Er bleibt aus Gründen der Nachvollziehbarkeit und für Tests erhalten. Für Aussagen über das finale Experiment der Masterarbeit dürfen weder `RESULTS.md` noch dieser historische Lauf als Quelle verwendet werden.

## Archivierter Abgabestand

Der unveränderliche Softwarestand der eingereichten Masterarbeit wird als annotierter Tag und GitHub Release [`thesis-v1.0`](https://github.com/CaoYue688/Top-K-Re-ranking_Methods_for_Recommander_System/releases/tag/thesis-v1.0) veröffentlicht. Anhang B der Arbeit dokumentiert zusätzlich die vollständige 40-stellige Commit-Kennung. Reproduktionsangaben sollen diesen Commit oder das Release und nicht den veränderlichen Branch `main` zitieren.

## Lizenz

Der Quellcode steht unter der [MIT-Lizenz](LICENSE). Für die MovieLens-Daten gelten unabhängig davon die Nutzungsbedingungen von GroupLens.
## Nutzungsbedingungen der Daten

Für MovieLens gelten die Nutzungsbedingungen von GroupLens. Der Datensatz ist aus der offiziellen Quelle zu beziehen; die dort genannten Nutzungs- und Zitieranforderungen sind einzuhalten.
