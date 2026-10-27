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
- scenari e loop simulati: verificati
- gioco simulato a griglia 30×30 con visuale raggio 4, ostacolo mobile, obiettivo mobile ogni due mosse, inseguitore e combattimento a PV: presente in `simulated_game/`
- memoria traccia entità anonime e inferisce categorie descrittive provvisorie da osservazioni ripetute; non inferisce ancora semantica o pericolosità
- agente online che seleziona controlli opachi in base all'incertezza e aggiorna profili degli effetti osservati; nessuna semantica o ricompensa di combattimento è fornita
- memoria spaziale e policy di esplorazione collegate al ciclo dell'engine; obiettivo nascosto: GM esplora le frontiere e aggiorna la mappa temporanea
- test automatici: 33 superati con Python 3.14.2 e pytest 9.0.2 (2026-10-27)
- prova con danno a 3 PV, seed 0–9 e limite 200 azioni: 6 vittorie, 1 sconfitta e 3 partite non concluse
- memoria di apprendimento solo in RAM e limitata all'istanza corrente: a ogni nuova partita il runner crea un nuovo engine e il learner riparte vuoto
- latenza di `process_frame` su frame mock: media 7,35 µs, p95 9,20 µs su 1.000 chiamate, logging disabilitato; non rappresenta prestazioni su un gioco reale

### Funzionalità mancanti o placeholder
- capture reale dello schermo
- percezione reale del gioco
- OCR e riconoscimento visivo
- input reale su tastiera/mouse
- gioco target operativo
- memoria avanzata e persistenza
- apprendimento generale dei controlli e delle dinamiche
- generalizzazione tra giochi

## Priorità attuali

1. Valutare e implementare persistenza tra partite, mantenendo separata la memoria del singolo episodio.
2. Migliorare la gestione di entità dinamiche e della mappa parziale; misurare i risultati su più mappe e seed.
3. Collegare una fonte di frame reale o un dataset riproducibile.
4. Implementare input reale solo in un ambiente controllato.
5. Misurare separatamente acquisizione, percezione, decisione e input su un benchmark reale.

## Limiti noti

- il sistema non è ancora in grado di vedere un gioco reale;
- la percezione è ancora mockata;
- il planner di navigazione resta basato su regole/BFS; l'apprendimento dei controlli è contestuale e non produce una strategia semantica;
- non esiste ancora un benchmark reale di gioco;
- la generalizzazione non è ancora verificata.

## Prossima azione

Estendere il gioco simulato a mappe di test variabili e mantenere la verifica separata dai benchmark mock e da eventuali prove su un gioco reale.
