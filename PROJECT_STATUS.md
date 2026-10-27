# PROJECT_STATUS.md

## Stato generale

Il progetto è in una fase iniziale ma coerente: la struttura progetto è stata creata, il loop base operativa è definito e i benchmark fittizi per il nucleo logico sono disponibili.

## Stato dei componenti

### Documentazione
- README.md: presente
- AGENTS.md: presente
- ARCHITECTURE.md: presente
- ROADMAP.md: presente
- DECISIONS.md: presente
- ACCEPTANCE_TESTS.md: presente
- EXPERIMENT_LOG.md: presente

### Struttura del progetto
- app/: presente
- core/: presente
- perception/: presente
- decision/: presente
- control/: presente
- utils/: presente
- tests/: presente

### Funzionalità implementate
- engine base: presente
- tipi di base: presenti
- planner minimale: presente
- input adapter minimale: presente
- memoria semplice: presente
- test di inizializzazione: presenti
- benchmark fittizi: presenti

### Funzionalità mancanti o placeholder
- capture reale dello schermo
- percezione reale del gioco
- OCR e riconoscimento visivo
- input reale su tastiera/mouse
- gioco target operativo
- memoria avanzata e persistenza
- apprendimento dei controlli e delle dinamiche
- generalizzazione tra giochi

## Priorità attuali

1. Validare il loop base con i benchmark fittizi.
2. Collegare una percezione reale e un frame di prova.
3. Creare un ambiente di test controllato.
4. Definire i primi casi di apprendimento dei controlli.
5. Fare una verifica delle prestazioni del ciclo.

## Limiti noti

- il sistema non è ancora in grado di vedere un gioco reale;
- la percezione è ancora mockata;
- la logica di decisione è base e molto semplicistica;
- non esiste ancora un benchmark reale di gioco;
- la generalizzazione non è ancora verificata.

## Prossima azione

Sviluppare il primo livello di test realistico con frame simulato e validare il flusso completo osservazione → decisione → azione → deviazione → memoria.
