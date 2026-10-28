# PROJECT_STATUS.md

## Stato generale

Il progetto ha un core di orchestrazione indipendente dall'ambiente e un adapter di test a griglia isolato. Non è ancora collegato a un gioco reale.

## Confine architetturale

- `core.GameEngine` riceve policy ed esecutore di azioni; può ricevere un adapter di osservazione. Il core non sceglie azioni, controlli o obiettivi predefiniti.
- La memoria nel core conserva osservazioni recenti, memoria di lavoro ed esperienze generiche.
- World model a griglia, BFS, esplorazione, memoria spaziale, learner dei controlli, regole del gioco e adapter input sono confinati in `simulated_game/`.
- `simulated_game.runtime` compone l'ambiente di prova con `GameEngine`; nessun modulo del core importa l'implementazione della simulazione.
- `app.main` non avvia un gioco o un fallback simulato: segnala che manca una configurazione di ambiente.

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
- engine generico con dipendenze iniettate: presente
- tipi di base: presenti
- memoria generica di osservazioni ed esperienze: presente
- test del contratto policy/esecutore: presenti
- planner, input adapter e memoria spaziale: presenti soltanto nell'adapter simulato
- gioco simulato a griglia 30×30 con visuale raggio 4, ostacolo mobile, obiettivo mobile ogni due mosse, inseguitore e combattimento a PV: presente in `simulated_game/`
- memoria traccia entità anonime e inferisce categorie descrittive provvisorie da osservazioni ripetute; non inferisce ancora semantica o pericolosità
- learner di controlli opachi: il planner indica la direzione verso obiettivo/frontiera, il learner inferisce gli spostamenti prodotti da ogni controllo e seleziona quello compatibile; persiste gli effetti per gioco. Non riceve la mappatura semantica; associazioni causali non garantite.
- memoria spaziale e policy di esplorazione collegate all'engine soltanto attraverso la composizione dell'adapter di test
- memoria dell'ultima posizione del target; la posizione è invalidata se osservata vuota. Posizioni diverse generano evidenza di mobilità; la soglia per smettere di inseguire il target ricordato è configurabile dall'adapter e varia nella ricerca strategica.
- recovery: dopo una transizione nota come bloccata, il planner evita quel passo per due decisioni. Non attribuisce un blocco a un passo pianificato se il controllo anonimo non ha un effetto di spostamento già identificato.
- suite completa aggiornata: 49 test superati il 2026-10-28, inclusi generatore di varianti, adapter, validazione held-out e persistenza delle strategie
- ricerca strategica evolutiva: genera varianti entro parametri bounded, confronta su seed appaiati e valida su holdout; il primo candidato dinamico ha fatto 10/20 vittorie contro 8/20 baseline ma ha avuto 6 regressioni appaiate, quindi non è stato promosso
- interfaccia del gioco simulato: sei input esposti come `control_1`–`control_6`; la policy usa gli effetti osservati per tradurre il passo del planner in un controllo anonimo.
- la mappa ricorda l'ultima posizione osservata dell'obiettivo e la usa quando è fuori visuale; la posizione memorizzata viene scartata quando è osservata vuota. I collegamenti temporaneamente bloccati sono evitati per due decisioni per consentire il recovery.
- benchmark riproducibile: `python -m simulated_game.benchmark`; distingue obiettivo statico senza inseguitore da scenario dinamico completo e misura esiti, stalli, blocchi, copertura, progresso e accuratezza delle previsioni direzionali.
- benchmark statico appaiato, 3 ordini di 40 seed: 40/40 vittorie con memoria volatile e persistente in ciascun ordine. Turni medi: 78,0 volatile, 62,67–62,70 persistente. Predizioni direzionali corrette 100% (2.741 campioni volatile, 2.497 persistenti); blocchi ai confini 81 contro 2.
- benchmark dinamico, un ordine appaiato di 40 seed: volatile 25 vittorie/0 sconfitte/15 timeout; persistente 22/0/17 timeout/1 stallo esplicito. Il bersaglio si è spostato in media 63,75 e 61,48 volte per episodio; era visibile circa 6 e 5 turni. Il target dinamico è stato inferito in 35/40 episodi volatili e 29/40 persistenti, ma non ha migliorato gli esiti in quest'ordine.
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

1. Migliorare l'esplorazione dopo aver inferito un target dinamico e valutare su ulteriori ordini di seed; il recupero dei controlli è verificato nel caso statico, ma l'ipotesi di mobilità non ha ancora migliorato le vittorie.
2. Separare stima/gestione di entità dinamiche da ricerca spaziale e testare trasferimento su un secondo ambiente.
3. Collegare una fonte di frame reale o un dataset riproducibile, dopo aver scelto un ambiente consentito.
4. Implementare input reale solo in un ambiente controllato, con stop e limiti operativi.
5. Misurare separatamente acquisizione, percezione, decisione e input su un benchmark reale.

## Limiti noti

- il sistema non è ancora in grado di vedere un gioco reale;
- la percezione è ancora mockata;
- il planner di navigazione resta basato su BFS e frontiere; gli effetti dei controlli sono appresi, ma strategia generale e adattamento a obiettivi mobili non sono dimostrati;
- non esiste ancora un benchmark reale di gioco;
- la generalizzazione non è ancora verificata.

## Prossima azione

Ampliare le varianti strategiche misurabili e ripetere l'holdout dinamico con più seed: il primo candidato ha aumentato le vittorie aggregate, ma ha anche invertito sei esiti appaiati e per questo non è stato promosso.

## Esperimento neurale in corso

- Policy DQN opzionale, separata dalla baseline e confinata all'adapter della griglia, con reward terminale e curiosità intrinseca.
- Il trainer riprende pesi, optimizer, replay, RNG e contatori; valuta periodicamente contro la baseline su seed validation fissi e riserva seed distinti al confronto finale.
- Conserva checkpoint latest riprendibile e miglior checkpoint validation; la baseline non viene sostituita automaticamente.
- Il training completo da zero (300 episodi) è stato nettamente peggiore nell'holdout: il checkpoint best ha fatto 6/40 vittorie e 1 sconfitta, contro 24/40 vittorie e zero sconfitte baseline; 20 regressioni e 2 miglioramenti appaiati. Nessuna promozione.
- Warm-start con 100 episodi baseline, 13.979 esempi, 5 epoche di imitazione e 300 episodi DQN: checkpoint best 11/40 vittorie e 3 sconfitte, contro 24/40 e 0 baseline; 17 regressioni e 4 miglioramenti appaiati. Migliora lo scratch, ma non raggiunge la baseline e non è promosso.
- Test specifici coprono maschere, decadimento epsilon, training state round-trip, ripresa tra invocazioni, imitazione supervisionata e valutazione appaiata.
- La policy non dimostra ragionamento generale, obiettivi semantici autonomi o trasferimento tra giochi.
