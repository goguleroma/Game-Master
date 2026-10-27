# ARCHITECTURE.md

## Obiettivo

Definire l'architettura tecnica di Game Master in modo coerente con la missione del progetto: costruire un agente capace di osservare, scegliere azioni, verificare risultati e apprendere in modo incrementale.

## Architettura generale

Il sistema è progettato come un'architettura ibrida composta da:

- Central Agent / Orchestrator;
- Perception Manager;
- World Model;
- Memory Manager;
- Decision and Planning;
- Input Adapter;
- Experiment Manager;
- Learning Manager;
- modelli neurali specializzati, dove utili.

Il loop operativo base è:

```text
Osservazione -> Stato -> Decisione -> Azione -> Verifica -> Esperienza -> Apprendimento
```

## Moduli principali

### Central Agent

Responsabile del coordinamento del ciclo decisionale e della gestione delle interazioni tra moduli. Non deve essere un monolite, ma un coordinatore di componenti specializzati.

### Perception Manager

Acquisisce dati osservabili, immagini, valori di HUD, oggetti, eventi e informazioni contestuali. La percezione è il primo livello di elaborazione e base per il progetto.

### World Model

Rappresenta lo stato del gioco in un dato momento, aggrega informazioni osservate, gestisce incertezza e collega eventi recenti alle decisioni future.

### Memory Manager

Gestisce:
- memoria di lavoro;
- memoria di eventi recenti;
- memoria delle esperienze;
- conoscenze verificate;
- competenze e strategie riutilizzabili.

### Decision and Planning

Sceglie azioni in base alla situazione, agli obiettivi e al livello di incertezza. Inizialmente può essere molto semplice, poi si evolverà in pianificazione ad alto livello.

### Input Adapter

Converte le decisioni astratte in input reali: tastiera, mouse, click, movimenti, attese, combinazioni di tasti.

### Experiment Manager

Organizza test controllati per scoprire i controlli del gioco, le relazioni input-effetto e la validità del comportamento.

### Learning Manager

Valuta le esperienze, decide cosa consolidare, cosa aggiornare e cosa rifiutare. Non deve trasformare ogni risultato in una regola generale senza verifica.

## Contratti dati

Le strutture principali definite per il progetto sono:

### Observation

Rappresenta una percezione singola:
- imagem / frame;
- timestamp;
- metadata;
- sorgente del dato.

### GameState

Rappresenta lo stato del gioco a un dato momento:
- oggetti osservati;
- HUD;
- obiettivo attuale;
- livello di confidenza;
- eventuali dati di contesto.

### Action

Rappresenta un'azione astratta da eseguire:
- nome dell'azione;
- parametri;
- eventuale contesto di esecuzione.

### ActionResult

Rappresenta il risultato osservato di un'azione:
- successo o fallimento;
- dettagli dell'esecuzione;
- eventuale evidenza osservabile.

### ExperienceRecord

Rappresenta un'esperienza registrata:
- stato iniziale;
- azione scelta;
- esito effettivo;
- contesto;
- timestamp;
- livello di affidabilità.

### KnowledgeItem

Rappresenta una conoscenza consolidata o in fase di verifica:
- contenuto;
- confidenza;
- contesto;
- evidenze;
- validità;
- data di creazione e aggiornamento.

## Loop operativo

```text
Frame / osservazione
    ↓
Perception Manager
    ↓
World Model
    ↓
Memory retrieval
    ↓
Decision & Planning
    ↓
Action proposal
    ↓
Input Adapter
    ↓
Action execution
    ↓
Result verification
    ↓
ExperienceRecord
    ↓
Learning Manager
    ↓
Knowledge update
```

## Requisiti architetturali

- Python come base di sviluppo iniziale.
- modularità e separazione di responsabilità;
- logging e tracciabilità di ogni decisione;
- possibilità di testare ciascun modulo indipendentemente;
- gestione dell'incertezza come prima classe;
- supporto al benchmark e alla valutazione delle performance;
- supporto all'evoluzione da mocka a dati reali senza riscrivere tutto.

## Compromessi e decisioni

### Python first

Python è la scelta corretta per:
- velocità di prototyping;
- flessibilità del lavoro di ricerca;
- compatibilità con librerie di elaborazione visiva e AI;
- costruzione della base logica del progetto.

Rust non va considerato come scelta iniziale obbligatoria. È una possibile ottimizzazione in una seconda fase, se si dimostra un collo di bottiglia reale.

### Percezione incrementale

La percezione non deve essere pensata come un modello monolitico. È meglio costruire un sistema di ROI, differenza tra frame, tracking e OCR selettivo, poi estendere con modelli più complessi.

### Apprendimento controllato

L'agente deve sperimentare e verificare, non assumere che ogni effetto osservato sia causato dall'ultimo input.

## Vincoli tecnici

- il sistema deve essere capace di trattare incertezza e mancati dati;
- i moduli devono poter essere testati indipendentemente;
- il loop operativo deve essere semplice da eseguire su scenario di prova;
- le prestazioni devono essere misurate, non presunte;
- la generalizzazione va dimostrata e non data per scontata.

## Principio finale

Il progetto cresce per incrementi verificabili. La complessità va aggiunta solo quando c'è un problema concreto e un caso d'uso chiaro.
