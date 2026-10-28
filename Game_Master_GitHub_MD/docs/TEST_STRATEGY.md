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
