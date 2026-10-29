# Metriche

Per ogni run registrare ID/data, revisione codice, ambiente/versione, configurazione/seed, policy, task, episodi/azioni, runtime, esito e motivo del fallimento.

Metriche iniziali possibili: successo, progresso, azioni e tempo al goal, errori di esecuzione, latenza, rifiuti safety gate, esperimenti per identificare controlli, accuratezza delle previsioni, recovery e costo compute.

Definire la baseline prima del tuning; confrontare condizioni equivalenti; separare tuning e valutazione; riportare campione e variabilità. Un singolo successo non prova apprendimento affidabile. Scegliere uno storage semplice e versionato; evitare commit di catture grezze grandi.

Per la ricerca di strategie registrare anche numero di candidati generati, seed condivisi di training, seed indipendenti di holdout, reward medio, vittorie/sconfitte/timeout, guadagni e regressioni appaiate, tempo per vittoria e decisione di promozione. Non sovrascrivere la baseline quando un candidato mostra regressioni appaiate.

## Benchmark simulato corrente

Eseguire `python -m simulated_game.benchmark --scenario all --episodes 40 --sequences 1`.

- `static`: target fisso e inseguitore disabilitato, per isolare memoria, controllo e navigazione.
- `dynamic`: target mobile e inseguitore attivo.
- Entrambe le modalità confrontano seed appaiati con memoria volatile e database persistente temporaneo.
- Report: vittorie/sconfitte/timeout/stalli espliciti, turni, azioni bloccate e senza movimento, stalli di 8 turni, osservazioni del target, relocazioni, cambi di posizione del target inferiti, copertura, progresso verso la posizione del target pre-azione e accuratezza delle previsioni quando disponibili.
- Il progresso verso il target è una metrica diagnostica calcolata dall'evaluator, non è fornita all'agente.
- Aumentare `--sequences` per testare più ordini; documentare sempre numero di seed e ordini usati.
- Ogni sequenza riusa lo stesso insieme di seed `0`–`episodes-1` e ne cambia l'ordine. Più sequenze misurano sensibilità all'ordine e non aumentano il numero di scenari indipendenti.

## Retrieval episodico

Il report del benchmark include `queries`, `queries_with_matches`, `matched_cases`, `guided_decisions` e `mean_available_cases` per modalità.

- `matched_cases` conta riferimenti a casi recuperati per tutte le query, non casi unici.
- Riportare separatamente task statico e dinamico, seed/ordini, limite turni e condizione volatile/persistente.
- Per stimare l'effetto dei casi episodici, mantenere identiche tra condizioni le prove persistenti dei controlli e confrontare gli stessi seed. Una semplice comparazione volatile/persistente non separa i due tipi di memoria.
- Registrare soglia di similarità, supporto minimo e versione dello schema insieme alla revisione del codice.
