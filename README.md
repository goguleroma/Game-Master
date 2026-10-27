# Game-Master

Game Master è un progetto di intelligenza artificiale in Python dedicato a creare un agente capace di osservare videogiochi, comprenderne lo stato, imparare i controlli, prendere decisioni in base agli obiettivi e migliorare attraverso l'esperienza.

## Obiettivo

L'obiettivo finale non è costruire un bot ad hoc per un solo gioco, ma un agente generalista che:
- osserva il gioco in tempo reale;
- interpreta immagini, UI, oggetti, personaggi e eventi;
- ricostruisce una rappresentazione dello stato del mondo di gioco;
- scopre controlli, effetti e dinamiche tramite esperimenti controllati;
- pianifica azioni coerenti con gli obiettivi;
- verifica i risultati ed aggiorna la memoria;
- trasferisce conoscenze tra giochi diversi quando è possibile;
- migliora con l'esperienza, senza dipendere da una sequenza rigida di comandi.

## Principi chiave

- Incrementalità: sviluppare in passi piccoli, verificabili e reversibili.
- Modularità: separare percezione, decisione, controllo e memoria.
- Osservabilità: registrare eventi, errori, risultati e metriche.
- Robustezza: gestire incertezza, timeout, dati mancanti e fallimenti.
- Generalizzazione: preferire capacità riutilizzabili rispetto a soluzioni specifiche.
- Sperimentazione controllata: imparare i controlli osservando effetti verificabili.
- Verifica: ogni miglioramento va supportato da test o prove misurabili.

## Architettura iniziale

Il progetto è pensato attorno a una struttura ibrida:
- orchestratore centrale (Central Agent);
- moduli specializzati per percezione, controllo, memoria, pianificazione e apprendimento;
- sistemi di logging, stato, test e benchmark;
- loop operativo osserva -> decide -> agisce -> verifica -> apprende.

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
│   ├── test_engine.py
│   └── test_scenarios.py
├── AGENTS.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── PROJECT_STATUS.md
├── DECISIONS.md
├── ACCEPTANCE_TESTS.md
├── EXPERIMENT_LOG.md
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
```

## Stato corrente

Il repository ha una base iniziale coerente ma ancora molto minimale:
- struttura di progetto impostata;
- loop base operativo definito;
- tipi dati iniziali e memoria semplice;
- planner e input adapter mockati per il primo test di logica;
- benchmark fittizi per validare il flusso senza un gioco reale;
- assenza di percezione reale e di un gioco target operativo.

Questo è il punto corretto per continuare: validare il nucleo logico prima di aggiungere percezione di alto livello o tecnologie più pesanti.

## Come usare il progetto

### Installazione

```bash
cd Game-Master
python -m pip install -r requirements.txt
```

### Test

```bash
python -m pytest
```

### Test specifici

```bash
python -m pytest tests/test_engine.py
python -m pytest tests/test_scenarios.py
```

## Documentazione principale

- `AGENTS.md` — regole operative per l'agente e la guida di sviluppo.
- `ARCHITECTURE.md` — architettura dell'applicazione e contratti dati.
- `ROADMAP.md` — roadmap e priorità di sviluppo.
- `PROJECT_STATUS.md` — stato reale del progetto.
- `DECISIONS.md` — decisioni architetturali e tecnico-pratiche.
- `ACCEPTANCE_TESTS.md` — criteri di accettazione e casi di test.
- `EXPERIMENT_LOG.md` — log degli esperimenti e risultati verificati.

## Prossimi passi

1. Validare i benchmark fittizi e il loop base.
2. Aggiungere un primo livello di stato osservabile reale.
3. Definire un gioco target o ambiente controllato di prova.
4. Collegare capture, percezione e input reale.
5. Introdurre i primi elementi di memoria, apprendimento e strategia.
6. Rivedere la scelta tecnologica solo dopo aver misurato dove si perde tempo.

## Nota tecnica

Rust potrebbe essere utile in un secondo momento, ma non è la scelta iniziale corretta. Per il primo prototipo Python è la soluzione più rapida e più coerente con il progetto e con il tipo di sviluppo richiesto.

Il progetto è pensato per crescere in modo incrementale e verificabile. Non si punta subito a un sistema perfetto, ma a un nucleo che funziona, viene testato e può evolvere.
