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
- learner di controlli opachi: il planner indica la direzione verso obiettivo/frontiera, il learner inferisce gli spostamenti prodotti da ogni controllo e seleziona quello compatibile; persiste gli effetti per gioco. Non riceve la mappatura semantica; associazioni causali non garantite.
- memoria spaziale e policy di esplorazione collegate al ciclo dell'engine; obiettivo nascosto: GM esplora le frontiere e aggiorna la mappa temporanea
- test automatici: 40 superati con Python 3.14.2 e pytest 9.0.2 (2026-10-27)
- interfaccia del gioco simulato: sei input esposti come `control_1`–`control_6`; la policy usa gli effetti osservati per tradurre il passo del planner in un controllo anonimo.
- benchmark direzionale su tre ordini appaiati di 40 seed: modalità volatile 25 vittorie/0 sconfitte/15 timeout per ordine; memoria persistente 22/0/18, 21/0/19 e 22/0/18. La persistenza riduce turni e input medi, ma non aumenta le vittorie in questa prova.
- memoria iniziale: buffer STM da 60 frame/12 secondi, memoria MTM per stato e piano corrente, esperienze LTM limitate all'istanza, prove di controllo GSM persistite per gioco in SQLite; GM generale resta vuota e non riceve promozioni automatiche
- latenza di `process_frame` su frame mock: media 7,35 µs, p95 9,20 µs su 1.000 chiamate, logging disabilitato; non rappresenta prestazioni su un gioco reale

### Funzionalità mancanti o placeholder
- capture reale dello schermo
- percezione reale del gioco
- OCR e riconoscimento visivo
- input reale su tastiera/mouse
- gioco target operativo
- consolidamento verificato e trasferimento alla memoria generale
- apprendimento generale dei controlli e delle dinamiche
- generalizzazione tra giochi

## Priorità attuali

1. Distinguere input inefficaci da input temporaneamente bloccati e migliorare esplorazione/progresso verso l'obiettivo dinamico; il benchmark attuale non mostra un aumento delle vittorie con la persistenza.
2. Migliorare la gestione di entità dinamiche e della mappa parziale; misurare i risultati su più mappe e seed.
3. Collegare una fonte di frame reale o un dataset riproducibile.
4. Implementare input reale solo in un ambiente controllato.
5. Misurare separatamente acquisizione, percezione, decisione e input su un benchmark reale.

## Limiti noti

- il sistema non è ancora in grado di vedere un gioco reale;
- la percezione è ancora mockata;
- il planner di navigazione resta basato su regole/BFS; il learner calibra gli effetti dei controlli ma non apprende ancora una strategia generale;
- non esiste ancora un benchmark reale di gioco;
- la generalizzazione non è ancora verificata.

## Prossima azione

Estendere il gioco simulato a mappe di test variabili e mantenere la verifica separata dai benchmark mock e da eventuali prove su un gioco reale.
