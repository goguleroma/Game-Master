Leggi `AGENTS.md` e i documenti `docs/` pertinenti prima di modificare il codice.

Game Master è un progetto Python greenfield per un agente videoludico ibrido. Ciclo previsto: osservare → interpretare → aggiornare stato → decidere/esplorare → agire → verificare → registrare → apprendere.

Preservare Central Agent e interfacce sostituibili. La scoperta dei controlli usa esperimenti limitati, non soltanto mapping manuali. Ispezionare il repository reale. Preferire un vertical slice a scaffolding inutilizzato. Aggiungere test e riportare solo quelli eseguiti. Prevedere stop, timeout e cleanup degli input. Dati locali per default. Non implementare bypass anti-cheat. Aggiornare i documenti quando cambiano contratti, architettura, dipendenze o stato.
