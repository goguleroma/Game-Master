# ROADMAP.md

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
- Planner
- InputAdapter
- MemoryStore

Criterio di completamento:
- il sistema compie un ciclo completo con dati fittizi e registra un risultato.

## Fase 2 — Benchmark fittizi e test deterministici

Obiettivo: validare la logica del sistema senza un gioco reale.

Criteri di completamento:
- casi di test per menu, stato vuoto, obiettivo move_left, oggetti sconosciuti;
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
