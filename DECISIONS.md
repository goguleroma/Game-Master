# DECISIONS.md

## Decisione 1 — Python come base del progetto

**Decisione:** mantenere Python come lingua principale per la prima fase del progetto.

**Motivazione:** velocità di prototyping, flessibilità, ecosistema di librerie, facilità di iterazione e compatibilità con il lavoro iniziale.

**Conseguenza:** Rust verrà considerato solo come ottimizzazione di un collo di bottiglia reale, non come soluzione iniziale predefinita.

## Decisione 2 — Architettura ibrida

**Decisione:** usare un orchestratore centrale più moduli specializzati.

**Motivazione:** separazione di responsabilità, modularità e maggiore controllabilità della pipeline.

**Conseguenze:** necessità di contratti dati chiari tra i moduli e di test di integrazione.

## Decisione 3 — Apprendimento autonomo dei controlli

**Decisione:** l'agente deve scoprire i controlli e i loro effetti attraverso osservazione e sperimentazione.

**Motivazione:** costruire un agente generalista, non un bot scritto ad hoc per titolo.

**Conseguenze:** servono esperimenti controllati, log e criteri di confidenza delle relazioni scoperte.

## Decisione 4 — Fokus sul ciclo operativo minimo

**Decisione:** prima ottenere un loop minimale funzionante, poi aumentare la complessità.

**Motivazione:** evitare di costruire un sistema teoricamente avanzato ma non verificabile.

**Conseguenze:** i benchmark fittizi sono essenziali e servono come base di validazione.

## Decisione 5 — Valutare prima la percezione reale

**Decisione:** prima di qualsiasi decisione su Rust o modelli più avanzati, bisogna misurare dove si perde tempo e dove si sta bloccando il sistema.

**Motivazione:** evitare di ottimizzare il componente sbagliato.

**Conseguenze:** le performance vanno misurate separatamente per acquisizione, percezione, decisione e input.

## Decisione 6 — Gestione dell'incertezza

**Decisione:** distinguere tra osservazioni, ipotesi, conoscenze verificate e regole consolidate.

**Motivazione:** evitare che un errore o un risultato sporadico diventi credenza permanente.

**Conseguenze:** la memoria è la base per l'apprendimento affidabile.
