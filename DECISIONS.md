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

## Decisione 7 — Generazione e promozione di varianti strategiche

**Decisione:** iniziare con ricerca evolutiva interpretabile su parametri bounded forniti dalla policy dell'ambiente. Valutare ogni variante sugli stessi seed della baseline e validare il candidato su seed held-out separati prima di promuoverlo.

**Motivazione:** permette variazioni autonome e confrontabili con pochi dati, senza assumere che una rete neurale o una ricerca libera siano già utili.

**Conseguenze:** la prima integrazione riguarda soltanto due pesi della policy di esplorazione nella simulazione a griglia. Un mancato miglioramento lascia intatta la strategia salvata; il risultato non implica generalizzazione.

## Decisione 8 — Retrieval episodico conservativo nell'adapter

**Decisione:** registrare contesto osservabile, direzione pianificata ed esito delle azioni a griglia in una memoria episodica locale al gioco. Applicare il punteggio recuperato solo come tie-break tra alternative di costo equivalente; richiedere supporto ripetuto e mantenere limiti alla cache e alla finestra di retrieval.

**Motivazione:** migliorare l'uso dell'esperienza e della memoria prima di introdurre ulteriori modalità di percezione o orchestrazione parallela, riducendo il rischio di sovrascrivere una pianificazione più corta con evidenza debole.

**Conseguenze:** l'implementazione non modifica il core né l'encoder neurale e non generalizza tra giochi. Il run esplorativo corrente è piccolo e confonde i casi episodici con la persistenza delle prove di controllo; prima di valutare un beneficio occorre una ablation dedicata e una replica più ampia.

## Decisione 9 — Branch di esplorazione per obiettivo dinamico

**Decisione:** quando la memoria spaziale segnala un obiettivo dinamico e l'obiettivo non è visibile, il planner imposta i target a lista vuota, delegando la scelta alla policy di esplorazione invece di inseguire le posizioni ricordate.

**Motivazione:** nell'ambiente dinamico l'obiettivo si sposta frequentemente e spesso era fuori visuale; inseguire posizioni memorizzate produceva blocchi e copertura insufficiente.

**Conseguenze:** il ramo resta confinato all'adapter simulato e collegato al flag `goal_is_dynamic`; l'A/B vecchio planner vs nuovo planner ha mostrato copertura 0,311 contro 0,731 ma più azioni bloccate, quindi il comportamento va rivalutato su più ordini di seed.

## Decisione 10 — Ritardo dell'ambiente reale

**Decisione:** non procedere con l'integrazione di un ambiente reale in questa fase; rimandare la decisione alla luce dei risultati simulati.

**Motivazione:** la percezione reale, l'OCR e l'input reale restano non implementati e il progetto non ha ancora un ambiente target operativo.

**Conseguenze:** le priorità attuali restano sul simulatore; le funzionalità mancanti restano in backlog.

## Decisione 11 — Branch di esplorazione per obiettivo dinamico confermato

**Decisione:** mantenere il branch di esplorazione per obiettivo dinamico: quando la memoria spaziale segnala un obiettivo dinamico e l'obiettivo non è visibile, il planner imposta i target a lista vuota invece di inseguire la posizione memorizzata.

**Motivazione:** l'A/B della Fase 2 su 5 sequenze del benchmark dinamico ha confermato il miglioramento introdotto nel commit `2a2055b`: la sequenza 1 (ordine 9204) ha riprodotto i numeri del nuovo planner della Fase 1 e le sequenze 2-5 hanno confermato il trend (vittorie persistenti 25-29 contro 22 del vecchio planner, timeout 10-15, stalli ≤1, mean_turns 103-119).

**Conseguenze:** il ramo resta confinato all'adapter simulato e collegato al flag `goal_is_dynamic`; il verdict KEEP giustifica il proseguimento alla Fase 3 (wiring delle stop conditions e prossimo miglioramento).

## Decisione 12 - Attribution dell'improvement dinamico al ranking delle frontiere

**Decisione:** l'improvement misurato nella Fase 2 sul target dinamico e' attribuibile al ranking delle frontiere introdotto nel commit `2a2055b`; il ramo di esplorazione per obiettivo dinamico e' logicamente equivalente al ramo precedente sotto la strategia di default.

**Motivazione:** l'A/B del ranking (vecchio vs nuovo) con la logica del ramo dinamico mantenuta ha mostrato una coverage volatile da 0,302 a 0,731 e una delta di wins persistenti; l'equivalenza vecchio/nuovo planner sotto strategia di default ha escluso il ramo dinamico come fonte unica dell'improvement.

**Conseguenze:** il ramo di esplorazione per obiettivo dinamico va mantenuto per strategie con soglia di cambio obiettivo superiore alla soglia di default; l'attribuzione dell'improvement passa al ranking delle frontiere come variabile principale da valutare nella fase successiva.

## Decisione 13 - Attribution finale dell'improvement dinamico al percorso guidato dalla memoria

**Decisione:** l'improvement misurato nella Fase 2 sul target dinamico e' attribuibile al percorso decisionale guidato dalla memoria (retrieval episodico) introdotto nel commit `2a2055b`; il ramo di esplorazione per obiettivo dinamico risulta equivalente al ramo precedente sotto strategia di default.

**Motivazione:** l'ablazione retrieval-on/off con il nuovo planner ha mostrato risultati coincidenti con il vecchio planner quando il retrieval e' disattivato (coverage volatile 0,311, wins persistenti 22), mentre con retrieval attivo (flag on) il nuovo planner produce coverage 0,731 e wins 25.

**Conseguenze:** la validazione sperimentale resta sul planner nuovo con retrieval attivo; la fase successiva deve replicare l'ablazione su piu' ordini e valutare l'integrazione del ranking delle frontiere nello stesso percorso.
