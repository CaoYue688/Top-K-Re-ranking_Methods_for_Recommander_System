[中文](RESEARCH_PROTOCOL.md) | [English](RESEARCH_PROTOCOL.en.md) | [Deutsch](RESEARCH_PROTOCOL.de.md)

# Finales Experimentprotokoll der Masterarbeit

Dieses Protokoll gehört zu `recsys20m.thesis_pipeline` und zum Datenlabel
`thesis_pos4_neg2_traincore5_dataseed2026`. Der frühere Plan A und die K=20-Ergebnisse mit nur einem Seed gehören nicht zur Evidenzgrundlage der finalen Masterarbeit.

## Forschungsfragen

- Welcher Diversitätsgewinn im Testdatensatz ist unter Validation-NDCG-Verlustbudgets von 1 %, 3 %, 5 % und 10 % erreichbar?
- Wie wirken sich MMR, xQuAD und Kalibrierungs-Re-Ranking auf ILD, die Abdeckung der Nutzerinteressen und die Übereinstimmung mit historischen Interessenanteilen aus?
- Sind die Schlussfolgerungen gegenüber Trainingsseed, Kandidatenmenge N, Listenlänge K, Nutzergruppe und Merkmalsraum robust?
- Verbessert lokale Listendiversität die Katalogabdeckung, die Long-Tail-Exposition und die globale Expositionskonzentration?

## Daten und Vermeidung von Leakage

- MovieLens 20M; `rating >= 4` gilt als positives Feedback, `rating <= 2` als explizit negatives Feedback und `2.5-3.5` als neutral.
- Das positive Feedback jedes Nutzers wird chronologisch ungefähr im Verhältnis 80/10/10 in Training, Validierung und Test aufgeteilt.
- Das iterative 5-Core wird **ausschließlich aus der vorab festgelegten, chronologisch früheren Trainingspartition berechnet**, damit zukünftige Interaktionen die Trainingspopulation nicht bestimmen können.
- Nutzer-Genreprofile verwenden ausschließlich positive Trainingsinteraktionen.
- Beim Validation-Retrieval werden alle im Trainingszeitfenster bewerteten Items ausgeschlossen; beim Test zusätzlich das Validation-Zeitfenster.
- Mit Wahrscheinlichkeit 0,5 verwendet das BPR-Negative-Sampling ein explizit negatives Item aus der Trainingsperiode des Nutzers; andernfalls ein tatsächlich unbewertetes Item.

Finaler Umfang: 134.703 Nutzer, 11.851 Items und 9.952.928 positive Interaktionen; davon 7.908.519 Training, 939.551 Validierung und 1.104.858 Test.

## Modell und Kandidaten

- BPR-MF mit 64-dimensionalen Embeddings, 10 Epochen, Batch-Größe 8.192 und CUDA.
- Trainingsseeds: 2026, 2027 und 2028; der Datenseed ist auf 2026 festgelegt.
- Hauptexperiment: N=100 und K=10; Robustheitsprüfungen: N in {50, 100, 200} und K in {5, 10, 20}.
- Candidate Recall wird separat berichtet, um die Retrieval-Obergrenze von Verlusten durch das Re-Ranking zu unterscheiden.

## Re-Ranking-Methoden

- MMR: Relevanz im Ausgleich mit der Genre-Kosinusdistanz zum ähnlichsten bereits ausgewählten Item.
- xQuAD: belohnt noch nicht ausreichend abgedeckte Interessenaspekte auf Basis des historischen Genrepriors des Nutzers.
- Kalibrierung: gleicht Relevanz und Jensen-Shannon-Ähnlichkeit zwischen historischer Genreverteilung und Verteilung der Empfehlungsliste aus.
- Primäres Lambda-Raster: 0,00, 0,05, ..., 1,00; Schrittweite der Robustheitsanalyse: 0,10.

## Auswahl und Statistik

Für Budget b und Methode m werden ausschließlich Validation-Konfigurationen beibehalten, die
`NDCG_m(lambda) >= (1-b) * NDCG_baseline` erfüllen. Unter ihnen wird die Konfiguration mit der höchsten ILD gewählt; Gleichstände werden durch höheren NDCG und anschließend kleineres Lambda aufgelöst. Testdaten werden nie für die Auswahl verwendet.

- 200 nutzerbezogene gepaarte Bootstrap-Stichproben je primärer Konfiguration; 100 für Robustheitskonfigurationen.
- Berichtet werden 95-%-Konfidenzintervalle, der gepaarte Vorzeichentest und Cohen's dz.
- Aussagen über mehrere Budgets werden nach Holm korrigiert; seedübergreifend werden Mittelwert, Seed-Standardabweichung und der konservativste p-Wert verwendet.

## Metriken

- Accuracy: NDCG@K, Recall@K, HR@K, MRR@K und Candidate Recall@N.
- Liste/Profil: ILD, Feature-ILD, Calibration, JS-Distanz, Genreentropie, Genreanzahl und profilgewichteter Subtopic Recall.
- System: Catalog Coverage, Genre Coverage, Exposure Gini, Long-tail share, Laufzeit und Python traced-memory peak.
- Gruppen: geringe/mittlere/hohe Aktivität und focused/medium/broad Profil-Terzile.

## Tag-Genome-Sensitivität

Auf das MovieLens Tag Genome wird eine GPU-randomisierte unzentrierte SVD64 angewendet. Sie deckt 9.864 von 11.851 Items ab und erhält 89,24 % der Frobenius-Energie. Genre- und Tag-Genome-Variante verwenden denselben tag-vollständigen Kandidatenpool, damit fehlende Merkmale nicht als Störfaktor wirken.

## Ausführung und Artefakte

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Die aggregierten Artefakte liegen unter:

`outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/`

`experiment_manifest.json` dokumentiert sämtliche Parameter, `all_thesis_results.csv` enthält 686 vollständige Konfigurationszeilen, und `validation_budget_selections.csv` bleibt von `test_budget_results.csv` getrennt, damit Modellauswahl und finale Berichterstattung nicht vermischt werden.
