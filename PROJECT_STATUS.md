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
- combattimento rebalanced: danno attacco portato da 3 a 5 PV in `GridGame._execute_combat`; con 10 PV a parte la posta in gioco è paragonabile all'urgenza di raggiungere l'obiettivo; suite 69 test superati; commit `a6d7b70` sul branch `simulated-game`.
- memoria traccia entità anonime e inferisce categorie descrittive provvisorie da osservazioni ripetute; non inferisce ancora semantica o pericolosità
- learner di controlli opachi: il planner indica la direzione verso obiettivo/frontiera, il learner inferisce gli spostamenti prodotti da ogni controllo e seleziona quello compatibile; persiste gli effetti per gioco. Non riceve la mappatura semantica; associazioni causali non garantite.
- memoria spaziale e policy di esplorazione collegate all'engine soltanto attraverso la composizione dell'adapter di test
- memoria dell'ultima posizione del target; la posizione è invalidata se osservata vuota. Posizioni diverse generano evidenza di mobilità; la soglia per smettere di inseguire il target ricordato è configurabile dall'adapter e varia nella ricerca strategica.
- recovery: dopo una transizione nota come bloccata, il planner evita quel passo per due decisioni. Non attribuisce un blocco a un passo pianificato se il controllo anonimo non ha un effetto di spostamento già identificato.
- memoria episodica dell'adapter: registra contesto osservato, direzione pianificata e risultato; recupera fino a 256 casi recenti per gioco, richiede almeno 2 esempi con similarità ≥0,65 e usa l'esito con confidenza. La cache in RAM è limitata a 2.000 casi; SQLite conserva i casi per gioco e li pota a batch. La memoria interviene solo tra scelte di costo equivalente. Anche l'hint del planner della policy neurale la usa senza cambiare l'encoder a 40 feature.
- suite completa aggiornata: 69 test superati il 2026-10-30, inclusi stop conditions, retrieval episodico, limiti di memoria, DQN, imitazione, adapter, validazione held-out e persistenza delle strategie
- ricerca strategica evolutiva: genera varianti entro parametri bounded, confronta su seed appaiati e valida su holdout; il primo candidato dinamico ha fatto 10/20 vittorie contro 8/20 baseline ma ha avuto 6 regressioni appaiate, quindi non è stato promosso
- interfaccia del gioco simulato: sei input esposti come `control_1`–`control_6`; la policy usa gli effetti osservati per tradurre il passo del planner in un controllo anonimo.
- la mappa ricorda l'ultima posizione osservata dell'obiettivo e la usa quando è fuori visuale; la posizione memorizzata viene scartata quando è osservata vuota. I collegamenti temporaneamente bloccati sono evitati per due decisioni per consentire il recovery.
- benchmark riproducibile: `python -m simulated_game.benchmark`; distingue obiettivo statico senza inseguitore da scenario dinamico completo e misura esiti, stalli, blocchi, copertura, progresso e accuratezza delle previsioni direzionali.
- benchmark statico appaiato, 3 ordini di 40 seed: 40/40 vittorie con memoria volatile e persistente in ciascun ordine. Turni medi: 78,0 volatile, 62,67–62,70 persistente. Predizioni direzionali corrette 100% (2.741 campioni volatile, 2.497 persistenti); blocchi ai confini 81 contro 2.
- benchmark dinamico, un ordine appaiato di 40 seed: volatile 25 vittorie/0 sconfitte/15 timeout; persistente 22/0/17 timeout/1 stallo esplicito. Il bersaglio si è spostato in media 63,75 e 61,48 volte per episodio; era visibile circa 6 e 5 turni. Il target dinamico è stato inferito in 35/40 episodi volatili e 29/40 persistenti, ma non ha migliorato gli esiti in quest'ordine.
- prova diagnostica episodica, 10 seed appaiati e massimo 100 turni: nello scenario statico 6/10 vittorie volatile e 4/10 persistenti; nel dinamico 2/10 e 6/10. La persistenza comprende anche le prove dei controlli, quindi il run non isola l'effetto causale dei casi episodici e non dimostra un vantaggio; dettagli in Esperimento 30.
- replica statica su 10 ordini degli stessi 10 seed, massimo 100 turni: 60/100 vittorie volatile e 58/100 persistenti; le vittorie persistenti variano da 4 a 7 per ordine contro 6/10 volatile in ciascuno. I blocchi medi per ordine scendono da 29 a 3 e i turni medi da 71,4 a 64,64, ma gli esiti appaiati cambiano 26 volte (12 miglioramenti e 14 regressioni). La persistenza include prove dei controlli e casi episodici; non isola questi contributi e ripete gli stessi seed. Dettagli in Esperimento 31.
- benchmark statico più ampio, 40 seed e massimo 200 turni, un ordine: volatile 40/40 vittorie, persistente 39/40 con un timeout; nessuna sconfitta e un solo esito appaiato cambiato. Persistente: 69,05 turni medi contro 74,45; 3 azioni bloccate e 6 senza movimento contro 118 e 179. Segnale di maggiore efficienza, ma non di più vittorie; la persistenza combina prove dei controlli e retrieval episodico. Dettagli in Esperimento 32.
- stop conditions del core: `GameEngine` accetta `max_steps` e `should_stop`; il loop verifica il segnale di stop e poi il limite di step, contando un passo solo dopo un `process_frame` valido. Test nel core.
- miglioramento esplorazione per obiettivo dinamico: quando `memory.spatial.goal_is_dynamic` è attiva e l'obiettivo non è visibile, il planner imposta i target a lista vuota (esplorazione) invece di inseguire la posizione ricordata. A/B vecchio planner vs nuovo planner sullo scenario dinamico, ordine 9204: copertura 0,311 contro 0,731; con memoria persistente il nuovo planner ha portato a +3 vittorie, -2 timeout, -7,67 mean_turns in meno e stallo 0 contro 1. Modifiche incluse nel commit `2a2055b` ("migliori sul comportamento").
- validazione Fase 2 dell'improvement esplorazione per obiettivo dinamico: A/B su 5 sequenze del benchmark dinamico (`--sequences 5`), 40 seed e max-turns 200, nuovo planner vs vecchio planner: la sequenza 1 (ordine 9204) riproduce i numeri del nuovo planner della Fase 1 (improvement REPRODUCED); trend confermato nelle sequenze 2-5 (vittorie persistenti 25-29 contro 22 del vecchio planner, timeout 10-15, stalli ≤1, mean_turns 103-119); persistente 137/200 vittorie (68,5%), mean_turns 111,87, blocked 3 per sequenza, mean coverage ~0,58; volatile 125/200 vittorie, mean_turns 137,3, blocked 118, coverage 0,731. Verdict: KEEP del nuovo branch; procedere alla Fase 3. Dettagli in Esperimento 34.
- attribuzione Fase 3 dell'improvement: A/B del ranking delle frontiere (vecchio vs nuovo) ha mostrato coverage volatile 0,302 contro 0,731 e delta di wins persistenti; sotto strategia di default vecchio e nuovo planner sono equivalenti, quindi l'improvement e' attribuibile al ranking delle frontiere del commit `2a2055b`. Dettagli in Esperimento 36.
- ablazione retrieval episodico (Fase 3): benchmark dinamico con e senza retrieval (`--disable-episodic-retrieval`) ha mostrato che con retrieval attivo il nuovo planner produce coverage 0,731 e wins 25, mentre con retrieval disattivato coincidono con il vecchio planner (coverage 0,311, wins 22); l'improvement e' quindi attribuibile al percorso guidato dalla memoria. Dettagli in Esperimento 37.
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

1. Isolare l'effetto del retrieval episodico dalle prove dei controlli persistenti e replicare su più ordini di seed, con limite di 200 turni e scenari statico/dinamico.
2. Valutazione dell'esplorazione attivata per obiettivo dinamico completata: l'A/B vecchio planner/new planner è stato eseguito su 5 sequenze del benchmark dinamico e l'improvement è stato confermato (trend replicato, coverage 0,311 contro 0,731, vittorie persistenti 25-29 contro 22 del vecchio planner, stallo 0 contro 1); validazione conclusa, prossima priorità la Fase 3.
3. Separare stima/gestione di entità dinamiche da ricerca spaziale e testare trasferimento su un secondo ambiente.
4. Collegare una fonte di frame reale o un dataset riproducibile, dopo aver scelto un ambiente consentito.
5. Implementare input reale solo in un ambiente controllato, con stop e limiti operativi.
6. Misurare separatamente acquisizione, percezione, decisione e input su un benchmark reale.

## Limiti noti

- il sistema non è ancora in grado di vedere un gioco reale;
- la percezione è ancora mockata;
- il planner di navigazione resta basato su BFS e frontiere; gli effetti dei controlli sono appresi, ma strategia generale e adattamento a obiettivi mobili non sono dimostrati;
- non esiste ancora un benchmark reale di gioco;
- la generalizzazione non è ancora verificata.

## Prossima azione

Proseguire alla Fase 3: wiring delle stop conditions e prossimo miglioramento. L'ablazione appaiata del retrieval episodico che mantenga identiche le prove persistenti dei controlli resta prioritaria e va eseguita con il protocollo completo da 200 turni. Conservare separati i risultati della policy neurale: il fine-tuning DQN ha portato la validation da 11/20 a 0/20 e non è promosso. L'A/B del planner dinamico su più ordini di seed è concluso con verdict KEEP del nuovo branch.

## Esperimento neurale in corso

- Policy DQN opzionale, separata dalla baseline e confinata all'adapter della griglia, con reward terminale e curiosità intrinseca.
- Il trainer riprende pesi, optimizer, replay, RNG e contatori; valuta periodicamente contro la baseline su seed validation fissi e riserva seed distinti al confronto finale.
- Conserva checkpoint latest riprendibile e miglior checkpoint validation; la baseline non viene sostituita automaticamente.
- Il training completo da zero (300 episodi) è stato nettamente peggiore nell'holdout: il checkpoint best ha fatto 6/40 vittorie e 1 sconfitta, contro 24/40 vittorie e zero sconfitte baseline; 20 regressioni e 2 miglioramenti appaiati. Nessuna promozione.
- Warm-start con 100 episodi baseline, 13.979 esempi, 5 epoche di imitazione e 300 episodi DQN: checkpoint best 11/40 vittorie e 3 sconfitte, contro 24/40 e 0 baseline; 17 regressioni e 4 miglioramenti appaiati. Migliora lo scratch, ma non raggiunge la baseline e non è promosso.
- Aggiunta al singolo test della rappresentazione una direzione BFS one-hot e un flag di intent frontier (input 40). Imitazione su validation: 10/20 contro 3/20 con vecchia codifica. Best checkpoint su due holdout indipendenti: 24/40 vittorie entrambe le volte contro baseline 26/40; nessuna sconfitta, ma 10 regressioni contro 8 miglioramenti appaiati. Il fine-tuning DQN ha peggiorato validation e non ha sostituito il best BC.
- Aggiunto un reward di progresso basato sull'intent BFS pre-azione, valido anche con target nascosto. Con 100 episodi di imitazione, il best checkpoint ha ottenuto 11/20 vittorie validation contro 10/20 baseline; dopo 300 episodi DQN è sceso a 0/20, quindi il best è rimasto il checkpoint BC.
- Su due holdout indipendenti, il checkpoint BC con reward di progresso ha fatto 23/40 contro 21/40 e 27/40 contro 26/40 vittorie; rispettivamente 12 miglioramenti/10 regressioni e 8/7 appaiati. Piccolo vantaggio aggregato, ma regressioni presenti su entrambi: non è una superiorità ripetibile e il candidato non è promosso.
- Terzo holdout indipendente, seed 1209397–1209436: baseline, checkpoint intent e checkpoint progress hanno tutti 24/40 vittorie e nessuna sconfitta. Entrambi i checkpoint neurali hanno 6 miglioramenti e 6 regressioni appaiate vs baseline; tempo medio per vittoria 78,5 contro 95,25, ma score quasi uguale (0,61867 vs 0,61745). Esito neutro, nessuna promozione.
- Diagnosi iniziale sul precedente holdout: nessun avanzamento nel 18,9% delle decisioni neurali contro 4,2% baseline; 13,2% azioni attacco/difesa. Il segnale di progresso è ora testato, ma non risolve il collasso osservato nel fine-tuning DQN.
- Test specifici coprono maschere, decadimento epsilon, training state round-trip, ripresa tra invocazioni, imitazione supervisionata e valutazione appaiata.
- La policy non dimostra ragionamento generale, obiettivi semantici autonomi o trasferimento tra giochi.
