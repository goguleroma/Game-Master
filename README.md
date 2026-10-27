# Game-Master

Game Master è un progetto Python in fase di costruzione, pensato per creare un agente AI capace di osservare videogiochi, comprenderne le dinamiche, apprendere i controlli e migliorare autonomamente attraverso l'esperienza.

## Obiettivo

L'obiettivo finale è costruire un agente in grado di:
- acquisire dati dal gioco in tempo reale
- interpretare immagini, UI e oggetti visibili
- comprendere lo stato del mondo di gioco
- scegliere azioni coerenti con gli obiettivi
- eseguire input in modo controllato
- verificare gli effetti delle azioni
- registrare esperienze e apprendere continuamente

## Principi guida

- Incrementalità: sviluppare in passi piccoli, verificabili e reversibili.
- Modularità: separare percezione, decisione, controllo e memoria.
- Osservabilità: loggare eventi, errori e risultati.
- Robustezza: gestire incertezza, timeout e fallimenti.
- Generalizzazione: costruire capacità riutilizzabili tra giochi diversi.
- Verifica: ogni miglioramento deve essere supportato da evidenze o test.

## Struttura del repository

```text
game_master/
├── app/
│   └── main.py
├── core/
│   ├── engine.py
│   ├── memory.py
│   ├── logger.py
│   ├── types.py
│   └── __init__.py
├── perception/
│   ├── capture.py
│   ├── state_extractor.py
│   └── __init__.py
├── decision/
│   ├── planner.py
│   └── __init__.py
├── control/
│   ├── input_adapter.py
│   └── __init__.py
├── utils/
│   ├── config.py
│   └── __init__.py
├── tests/
│   └── test_engine.py
├── requirements.txt
├── .gitignore
├── ROADMAP.md
├── PROJECT_STATUS.md
├── DECISIONS.md
├── README.md
└── .env.example
```

## MVP iniziale

Il primo prototipo è un ciclo operativo minimo composto da:
1. cattura della schermata di gioco
2. estrazione dello stato visibile
3. scelta di una semplice azione
4. esecuzione dell'input
5. verifica del risultato
6. registrazione dell'esperienza

## Fasi iniziali

- Audit del repo e definizione della struttura.
- Implementazione del loop base osservazione-decisione-azione.
- Definizione dei tipi di dato e delle azioni.
- Verifica del flusso operativo.
- Introduzione di memoria e logging.
- Apprendimento dei controlli e delle conseguenze.

## Documentazione di riferimento

- ROADMAP.md: roadmap e priorità.
- PROJECT_STATUS.md: stato del progetto.
- DECISIONS.md: decisioni tecniche e architetturali.

## Requisiti base

- Python 3.11+
- pytest
- PyAutoGUI o strumenti equivalenti per input e screenshot (se necessario)
- OpenCV (opzionale per rilevamento visivo iniziale)

## Primi passi pratici

- creare la struttura del repo
- definire il loop operativo minimo
- implementare i moduli base di percezione, pianificazione e input
- verificare il comportamento con un test controllato
- solo dopo estendere con memoria e apprendimento

## Nota

Il progetto è inizialmente pensato per crescere in modo incrementale e verificabile. L'obiettivo non è costruire subito un sistema generale completo, ma creare una base reale e testabile che possa evolvere in modo coerente.
