# ROADMAP.md

## Stato verificato al 2026-10-30

- Core generico con policy/esecutore iniettati; agent e memoria spaziale della griglia confinati nell'adapter simulato.
- Stop conditions del loop: `GameEngine` con `max_steps` e `should_stop`, verificati con test dedicati.
- Miglioramento esplorazione per obiettivo dinamico: con obiettivo dinamico e non visibile, il planner imposta i target a lista vuota e attiva l'esplorazione. A/B vecchio planner vs nuovo planner sullo scenario dinamico, ordine 9204: copertura 0,311 contro 0,731; con memoria persistente il nuovo planner ha portato a +3 vittorie, -2 timeout, -7,67 mean_turns e stallo 0 contro 1. Dettagli in Esperimento 33.
- Validazione Fase 2 dell'improvement esplorazione per obiettivo dinamico: A/B su 5 sequenze (`--sequences 5`) del benchmark dinamico, 40 seed e max-turns 200, nuovo planner vs trend del vecchio planner; improvement confermato e riprodotto (sequenza 1), trend confermato nelle sequenze 2-5, verdict KEEP, procedere alla Fase 3. Dettagli in Esperimento 34.
- Analisi del ranking delle frontiere (Fase 3): A/B vecchio vs nuovo ranking ha mostrato coverage volatile 0,302 contro 0,731 e delta di wins persistenti; equivalenza vecchio/nuovo planner sotto strategia di default; l'improvement e' attribuibile al ranking delle frontiere del commit `2a2055b`. Dettagli in Esperimento 36.
- Ablazione retrieval episodico (Fase 3): benchmark dinamico con e senza retrieval ha mostrato che l'improvement e' dovuto al percorso guidato dalla memoria; dettagli in Esperimento 37.
- Ciclo end-to-end simulato, apprendimento di sei controlli anonimi, memoria persistente e test dell'adapter: presenti.
- Task statico: 40/40 vittorie con memoria volatile e persistente nei tre ordini valutati; la persistenza riduce i turni medi.
- Task dinamico: in un ordine da 40 seed, 25 vittorie volatile contro 22 persistenti; non è dimostrato un beneficio di vittorie.
- Ricerca strategica: 14 varianti valutate; il migliore ha fatto 10/20 vittorie held-out contro 8/20 baseline, ma con 6 regressioni appaiate, quindi non promosso.
- Policy neurale: intent BFS e reward di progresso sono testati; su tre holdout il checkpoint progress ha ottenuto 23/40 vs 21/40, 27/40 vs 26/40 e 24/40 vs 24/40 baseline. Ci sono state regressioni appaiate in tutti i confronti; fine-tuning DQN ha abbassato validation da 11/20 a 0/20.
- Memoria episodica dell'adapter: retrieval contestuale per gioco con soglia di supporto/similarità e tie-break solo tra percorsi di costo uguale. Dieci ordini degli stessi 10 seed statici hanno dato 60/100 vittorie volatile e 58/100 persistenti; i blocchi e i turni medi calano con persistenza, ma non è isolato l'effetto dei casi dagli effetti appresi dei controlli. Vedi Esperimenti 30–31.
- Replica statica più ampia: su 40 seed e 200 turni, volatile 40/40 vittorie contro 39/40 persistenti; persistente riduce blocchi e turni medi, ma la modalità combina retrieval episodico e apprendimento dei controlli. Vedi Esperimento 32.
- Prossimo obiettivo: Fase 3 — wiring delle stop conditions e prossimo miglioramento; isolare il contributo del retrieval episodico dalle prove dei controlli persistenti resta il confronto da effettuare.
- Non implementati: acquisizione/percezione/input reali e trasferimento tra giochi.

## Fase 0 — Fondazione e struttura

Obiettivo: mettere in piedi la base del progetto in modo coerente.

Criteri di completamento:
- repository strutturato;
- documentazione base presente;
- moduli iniziali creati;
- principi e roadmap definiti.

## Fase 1 — Loop operativo minimo

Obiettivo: creare il ciclo osservazione → decisione → azione → verifica.

Componenti coinvolti:
- GameEngine
- Observation
- GameState
- Action
- ActionResult
- Policy iniettata
- Esecutore di azioni iniettato
- Memoria generica d'esperienza

Criterio di completamento:
- il sistema compie un ciclo completo con dati fittizi e registra un risultato.

## Fase 2 — Adapter di test e test deterministici

Obiettivo: validare la logica del sistema senza un gioco reale.

Criteri di completamento:
- test del core che accertano la delega a policy ed esecutore senza nomi d'azione specifici;
- test della griglia confinati al pacchetto di simulazione;
- registrazione dei risultati attesi;
- pipeline di test automatizzati.

## Fase 3 — Percezione e acquisizione

Obiettivo: collegare il sistema a sorgenti reali di osservazione.

Criteri di completamento:
- capture di frame reali;
- preprocessing di base;
- parsing dello stato in una struttura coerente;
- gestione dei casi in cui i dati non sono disponibili.

## Fase 4 — Apprendimento dei controlli

Obiettivo: far acquisire all'agente le relazioni input-effetto.

Criteri di completamento:
- registrazione delle relazioni input-effetto;
- verifiche con esperimenti controllati;
- conservazione di conoscenze con confidenza e incertezza.

## Fase 5 — Memoria e stato del mondo

Obiettivo: approcciare la gestione di conoscenze e contesto.

Criteri di completamento:
- la memoria distingue osservazione, ipotesi e conoscenze verificate;
- lo stato del gioco è aggiornato in modo coerente;
- gli errori non vengono automaticamente trasformati in regole.
- retrieval episodico limitato all'adapter simulato, con contesto versionato, soglia di supporto e effetto conservativo sulle scelte a costo equivalente; resta da misurare il beneficio con ablation e benchmark estesi.

## Fase 6 — Pianificazione

Obiettivo: trasformare le azioni in una strategia coerente con gli obiettivi.

Criteri di completamento:
- definizione di obiettivi e criteri di successo;
- scelta tra azioni immediate e piani a breve termine;
- gestione dei casi di blocco o mancanza di progresso.

## Fase 7 — Generalizzazione

Obiettivo: verificare se le conoscenze acquisite possono esser riutilizzate in giochi diversi.

Criteri di completamento:
- test su ambiente differente dal primo gioco di riferimento;
- analisi del trasferimento di competenze;
- separazione delle capacità generali da quelle specifiche del gioco.

## Fase 8 — Ottimizzazione e maturazione

Obiettivo: migliorare prestazioni, affidabilità e stabilità.

Criteri di completamento:
- benchmark reali delle performance;
- ottimizzazione del collo di bottiglia;
- log, diagnostica e gestione di regressioni;
- versione stabile del progetto.
