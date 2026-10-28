# Strategia di test

## Livelli
- Unit: contratti, stato, validazione azioni e incertezza.
- Contract: comportamento coerente degli adapter.
- Integration: Central Agent con adapter toy/fake.
- End-to-end: avvio, task e chiusura.
- Evaluation: metriche su più run.

## Casi critici
Osservazione invalida/stale, timeout, disconnessione, errore modello, azione non supportata, budget superato, stop durante azione, eccezione con input attivo, errore storage e transizione inattesa.

## Invarianti
Stop rispettato entro limite documentato; input rilasciati in caso di errore; nessuna azione bypassa il safety gate; dati stale non trattati come validi; cleanup garantito; log senza segreti; servizi esterni disattivati senza opt-in.

Confrontare con baseline, ripetere run, riportare campione/variabilità e misurare anche latency e risorse.

## Prove specifiche correnti

- Il core delega una decisione opaca all'adapter iniettato e non interpreta nomi o regole del gioco.
- L'adapter a griglia viene verificato separatamente; le sue coordinate e azioni non compaiono nei test del core.
- Obiettivo osservato e poi nascosto: la posizione recente è riutilizzata finché non viene osservata vuota.
- Osservazione vuota sul target noto: la memoria invalida l'ipotesi.
- Transizione bloccata: il planner evita temporaneamente il passo e trova un percorso alternativo quando disponibile.
- Obiettivo osservato in più posizioni: registrare un'ipotesi di mobilità e non trattare la coordinata precedente come certa.
- Effetti di controllo: accuratezza calcolata solo sulle predizioni disponibili e confrontata con lo spostamento osservato.
- Benchmark: separare target statico e dinamico per non confondere la qualità della navigazione con la frequenza di movimento del target.
- Ricerca strategica: tutti i candidati condividono seed di training; selezione e tuning non accedono agli seed held-out; nessun candidato viene promosso senza superare il confronto appaiato e i vincoli di regressione.
