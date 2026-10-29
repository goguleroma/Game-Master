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
│   └── __init__.py
├── control/
│   └── __init__.py
├── simulated_game/
│   ├── agent/       # Grid-only policy, model, memory and control adapter
│   ├── game.py      # Isolated test environment
│   └── runtime.py   # Composition of the simulator adapters with core
├── utils/
│   ├── config.py
│   └── __init__.py
├── tests/
│   ├── test_engine.py
│   ├── test_exploration.py
│   └── test_simulated_game.py
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

## Confine tra core e ambienti

`core.GameEngine` coordina lo scambio tra stato normalizzato, policy, esecutore di azioni e memoria d'esperienza. Policy ed esecutore sono obbligatoriamente iniettati; un adapter di osservazione può essere fornito per convertire dati di un ambiente in `GameState`. Il core non sceglie mosse di griglia, non conosce obiettivi o controlli del simulatore e non ha un comportamento di gioco predefinito.

Il codice specifico della griglia è confinato in `simulated_game/agent/` e `simulated_game/`. È un adapter di test sostituibile, non una dipendenza del core. Per provarlo:

```powershell
python -m simulated_game --seed 4
```

Il test adapter include una prima ricerca evolutiva che genera e confronta varianti dei parametri di esplorazione su seed condivisi e le verifica su holdout indipendenti:

```powershell
python -m simulated_game.strategy_search --scenario dynamic
```

È un esperimento limitato alla policy della griglia, non una capacità generalista né una rete neurale. La strategia locale viene aggiornata solo quando supera il criterio held-out documentato nel README della simulazione.

### Esperimento neurale isolato

La policy DQN è opzionale e confinata al simulatore. La baseline resta la modalità predefinita; il modello neurale non viene promosso né sostituisce la strategia baseline automaticamente. La codifica include un'indicazione di intent del planner (direzione BFS verso target/frontiera) e il reward misura il progresso rispetto a tale intent anche con target nascosto. Su tre holdout indipendenti il best checkpoint per imitazione ha avuto piccoli vantaggi aggregati nei primi due e un pareggio nelle vittorie nel terzo, ma regressioni appaiate in tutti; il fine-tuning DQN ha degradato la validation, quindi non è stato promosso. Sono aiuti specifici dell'adapter a griglia, non capacità generaliste. La codifica cambia dimensione: i vecchi checkpoint non sono caricabili dalla nuova policy. Per installare le dipendenze aggiuntive, addestrare un candidato e confrontarlo su seed held-out appaiati:

```powershell
python -m pip install -r requirements-neural.txt
python -m simulated_game.neural_experiment --train-episodes 300 --holdout-episodes 40
```

Il training riprende automaticamente il checkpoint `.gm-neural-training.pt` se presente; ogni avvio aggiunge gli episodi indicati a quelli precedenti e salva pesi, optimizer, replay buffer, RNG e contatori. Valuta ogni 50 episodi sullo stesso set di validation seed; il test finale usa seed separati e mai riutilizzati nelle riprese successive. Il miglior checkpoint secondo la validation viene conservato in `.gm-neural-training.best.pt`; il checkpoint latest permette di continuare il training. Per cominciare un esperimento indipendente, scegliere un percorso `--model` diverso e passare `--fresh`.

Il report confronta vittorie, sconfitte, timeout, stalli, score e regressioni appaiate. La baseline non viene promossa o sostituita automaticamente. Il checkpoint migliore si può provare esplicitamente con `python -m simulated_game --policy neural --model .gm-neural-training.best.pt`. I vecchi checkpoint di soli pesi restano caricabili in inferenza, ma per riprendere da un nuovo training usare un percorso di checkpoint distinto. È un esperimento di reinforcement learning sul simulatore, non prova di ragionamento generale o trasferimento tra giochi.

Se il training da zero non dà risultati sufficienti, la modalità sperimentale di warm-start imita prima le decisioni della baseline e poi continua con DQN:

```powershell
python -m simulated_game.neural_experiment --model .gm-neural-imitation.pt --fresh --initialization baseline --train-episodes 300 --imitation-episodes 100
```

Il warm-start è un confronto separato e non modifica il modello baseline.

Non esiste ancora un adapter per un gioco reale. `python -m app.main` lo segnala esplicitamente invece di avviare un agente simulato o di inventare controlli.

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
python -m pytest tests/test_simulated_game.py
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

1. Continuare a sviluppare il core tramite contratti e componenti neutrali rispetto all'ambiente.
2. Migliorare il test della ricerca di target dinamici esclusivamente nell'adapter simulato.
3. Scegliere un ambiente reale consentito prima di implementare i relativi adapter di osservazione e input.
4. Dimostrare il trasferimento tra ambienti con misure riproducibili prima di generalizzare le strategie.
5. Valutare reti neurali soltanto dopo aver misurato un limite che i meccanismi attuali non risolvono.

## Nota tecnica

Rust potrebbe essere utile in un secondo momento, ma non è la scelta iniziale corretta. Per il primo prototipo Python è la soluzione più rapida e più coerente con il progetto e con il tipo di sviluppo richiesto.

Il progetto è pensato per crescere in modo incrementale e verificabile. Non si punta subito a un sistema perfetto, ma a un nucleo che funziona, viene testato e può evolvere.
