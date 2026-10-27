# EXPERIMENT_LOG.md

## Scopo

Documentare gli esperimenti, i risultati e le conclusioni in modo replicabile.

## Template di esperimento

### Esperimento 1 - Benchmark menu start

- Obiettivo: verificare che il planner riconosca un bottone di start e scelga `click`.
- Input simulato: `{"objects": ["start_button"], "goal": "start"}`
- Modifica effettuata: None (test di base)
- Risultato: successo
- Conclusione: il loop base e il planner sono coerenti con un caso semplice.

### Esperimento 2 - Stato vuoto

- Obiettivo: verificare la gestione di dati mancanti.
- Input simulato: `None`
- Risultato: wait e stato di sicurezza.
- Conclusione: il sistema non va in errore quando non trova dati.

### Esperimento 3 - Move left objective

- Obiettivo: verificare l'interpretazione di un obiettivo di movimento.
- Input simulato: `{"objects": ["player"], "goal": "move_left"}`
- Risultato: `move_left` scelto e risultato registrato.
- Conclusione: un obiettivo semplice è operativo e il loop risponde coerentemente.

### Esperimento 4 - Verifica riproducibile del prototipo

- Data: 2026-10-27
- Obiettivo: verificare test, demo e latenza del percorso simulato.
- Ambiente: Windows, Python 3.14.2, pytest 9.0.2.
- Comandi: `python -m pytest -q`; `python -m app.main --demo`.
- Risultato test: 10 test superati; demo completata su 6 scenari.
- Algoritmo: planner deterministico basato su regole; frame fornito come dizionario Python.
- Misura: 1.000 chiamate a `GameEngine.process_frame`, logging disabilitato; media 7,35 µs, mediana 6,90 µs, p95 9,20 µs. Sono state registrate 1.000 esperienze in memoria.
- Limiti: misura locale del solo percorso simulato; non include cattura reale, computer vision, OCR, input di sistema o latenza del gioco. Non è un benchmark di FPS né una misura comparabile alla prestazione su un gioco.
- Conclusione: il flusso mock osservazione → stato → decisione → azione simulata → memoria è riproducibile. La misura non dimostra capacità di percepire o giocare.

### Esperimento 5 - Gioco simulato a griglia

- Data: 2026-10-27
- Obiettivo: verificare un ciclo chiuso in cui le azioni modificano lo stato osservato.
- Input: griglia 5×3, giocatore in `(0, 1)`, obiettivo in `(4, 1)`, ostacolo in `(2, 1)`.
- Algoritmo: ricerca BFS del percorso più breve nel planner per stati che dichiarano l'obiettivo `reach_goal`.
- Risultato: obiettivo raggiunto in 6 mosse; ogni nuova osservazione ha riportato la posizione aggiornata; 6 esperienze registrate.
- Test: completamento del percorso con ostacolo e collisione che non modifica la posizione; verificati nella suite.
- Comando: `python -m simulated_game`.
- Limiti: mondo e osservazione sono strutture dati sintetiche; non prova percezione visuale, apprendimento o controllo di un gioco reale.
- Conclusione: il prototipo dimostra osservazione → pianificazione → azione applicata → nuova osservazione e verifica della vittoria nell'ambiente simulato.

### Esperimento 6 - Obiettivo casuale nel gioco a griglia

- Data: 2026-10-27
- Obiettivo: rendere variabile la destinazione mantenendo scenari raggiungibili e riproducibili nei test.
- Regola: scelta casuale tra celle libere a distanza Manhattan di almeno 4 dalla posizione iniziale; il costruttore accetta un seed opzionale.
- Verifiche: destinazione valida e distinta da partenza/ostacoli; seed identico riproduce la stessa destinazione; il planner completa il percorso entro il numero di celle della griglia.
- Limiti: ostacolo, dimensione e algoritmo restano fissi; la difficoltà varia solo tramite la posizione del traguardo.

### Esperimento 7 - Ostacolo mobile

- Data: 2026-10-27
- Obiettivo: fare cambiare posizione all'ostacolo a ogni azione direzionale e verificare la reazione del planner.
- Regola: dopo ogni azione direzionale, anche se bloccata, l'ostacolo viene ricollocato casualmente in una cella diversa da quella attuale, dal giocatore e dall'obiettivo.
- Risultato: le nuove osservazioni includono la posizione aggiornata dell'ostacolo; il planner ricalcola il percorso a ogni iterazione.
- Verifiche: test su due mosse consecutive, nessuna sovrapposizione e aggiornamento anche dopo un movimento bloccato.
- Limiti: ambiente sintetico; non prova percezione visuale o apprendimento.

### Esperimento 8 - Obiettivo mobile ogni due mosse

- Data: 2026-10-27
- Obiettivo: spostare la destinazione durante la partita senza sovrapporla al giocatore o all'ostacolo.
- Regola: contare ogni azione direzionale di Game-Master; mantenere l'obiettivo per la prima azione e sceglierne una nuova casella libera dopo la seconda. Le azioni bloccate contano come turno.
- Risultato: l'osservazione successiva riporta la nuova destinazione e il planner ricalcola il percorso.
- Verifiche: test che controlla posizione invariata dopo un turno e spostamento al secondo, con vincoli anti-sovrapposizione.
- Limiti: dinamica casuale sintetica; il planner non anticipa i futuri spostamenti.

### Esperimento 9 - Visuale limitata a raggio 4

- Data: 2026-10-27
- Obiettivo: verificare come reagisce il planner quando non riceve le coordinate globali dell'obiettivo.
- Regola: osservazione limitata a un disco euclideo di raggio 4 dal giocatore; obiettivo e ostacoli fuori visuale non sono inclusi nello stato osservato.
- Risultato: se l'obiettivo è visibile, il planner continua a calcolare il percorso; se è nascosto, il planner sceglie `wait` e la partita si ferma senza progresso.
- Verifiche: test che controllano omissione delle coordinate nascoste e assenza di movimento con obiettivo fuori visuale.
- Conclusione: la limitazione rivela che GM dipende da coordinate note e non possiede ancora esplorazione o memoria spaziale utile a cercare un obiettivo sconosciuto.

### Esperimento 10 - Integrazione della memoria spaziale e dell'esplorazione

- Data: 2026-10-27
- Obiettivo: collegare `MemoryStore`, `WorldModel`, `ExplorationPolicy` e planner nel ciclo reale dell'engine.
- Modifiche: normalizzazione dello stato osservato; aggiornamento delle celle visibili/visitate e degli ostacoli osservati; selezione delle frontiere raggiungibili; pattugliamento delle celle meno visitate dopo aver esplorato tutta la mappa.
- Risultato: con obiettivo inizialmente nascosto, GM esplora senza riceverne le coordinate; su 10 seed fissi ha raggiunto l'obiettivo in tutti i casi, tra 6 e 44 turni.
- Test: regressioni legacy, memoria spaziale, esplorazione con target nascosto, aggiornamento dinamico e completamento su 10 seed.
- Limiti: esplorazione euristica deterministica, non apprendimento; griglia e osservazione restano strutturate/sintetiche; non prova generalizzazione né percezione visuale.

### Esperimento 11 - Nemico inseguitore con informazione invariata

- Data: 2026-10-27
- Obiettivo: osservare la risposta di GM a un'entità attiva senza introdurre un campo semantico specifico per il nemico.
- Regole ambiente: il nemico si attiva quando la distanza euclidea dal giocatore è entro 4 celle; dopo ogni mossa di GM, se attivo, insegue di una cella con un percorso BFS. La partita termina in caso di cattura.
- Osservazione per GM: il nemico visibile è incluso nel campo già esistente `obstacles`, senza una label `enemy`; le posizioni fuori dal campo visivo di GM sono omesse.
- Risultato su seed 0–9: 8 vittorie e 2 catture; il nemico si è attivato nella maggior parte delle partite. GM evita la posizione corrente come un ostacolo, ma non riconosce né anticipa l'inseguimento.
- Verifiche: attivazione e avanzamento dell'inseguitore; visibilità senza nuovo campo; esiti sia positivi sia negativi su seed fissi.
- Limiti: simulazione 2D deterministica a griglia; GM non ha un modello della dinamica del nemico.

### Esperimento 12 - Categorizzazione comportamentale di entità anonime

- Data: 2026-10-27
- Obiettivo: assegnare categorie descrittive provvisorie usando osservazioni ripetute, senza etichette semantiche del gioco.
- Dati: identificativo anonimo stabile e posizione corrente, limitati alla visuale già esistente.
- Categorie inferibili: `unclassified`, `stationary`, `mobile`, `approaches_agent`, `moves_away_from_agent`.
- Criterio: la categoria `stationary`/`mobile` deriva dai cambiamenti di posizione; le ipotesi direzionali richiedono almeno tre variazioni coerenti consecutive, restano distinte dalla categoria e hanno confidenza prudente.
- Test: entità anonime ferme, in movimento e che riducono ripetutamente la distanza dall'agente; verifica che non venga assegnata subito un'etichetta semantica.
- Limiti: è una classificazione euristica basata sul movimento, non apprendimento semantico; non deduce intenzione o pericolosità. Un'entità osservata per pochi turni resta `unclassified` o `mobile`; una dinamica casuale può produrre ipotesi temporanee.

### Esperimento 13 - PV e scoperta per tentativi di input anonimi

- Data: 2026-10-27
- Regole dell'ambiente: giocatore ed entità hanno 10 PV; attacco infligge 3 PV, difesa annulla il danno; l'entità sceglie casualmente e arretra di una casella dopo aver attaccato. A 0 PV il giocatore perde; l'entità a 0 PV lascia un ostacolo statico.
- Osservazione: GM riceve solo i PV propri e quelli dell'entità anonima, senza nomi o regole di combattimento.
- Decisione sperimentale: l'osservazione elenca due controlli anonimi; un agente online seleziona i controlli meno provati per contesto spaziale e aggiorna il profilo medio dei cambiamenti numerici osservati. Non usa i dettagli dell'azione per conoscere l'esito e non riceve una ricompensa di combattimento.
- Risultato dopo l'aumento del danno a 3 PV: seed 0–9, massimo 200 azioni, 6 vittorie (seed 0, 1, 2, 3, 7, 8), 1 sconfitta (seed 5) e 3 partite non concluse (seed 4, 6, 9). Nella sconfitta il giocatore e l'entità arrivano entrambi a 0 PV nello stesso turno; le partite non concluse terminano al limite con 1–7 PV per il giocatore e l'entità ancora presente. Il danno maggiore rende il rischio concreto, ma non aumenta le vittorie rispetto alla prova precedente.
- Persistenza: ogni episodio del runner crea un nuovo `GameEngine`; il learner dei controlli e la memoria partono vuoti. Gli episodi su seed diversi sono quindi prove indipendenti, non apprendimento cumulativo tra partite.
- Test: feedback che modifica la selezione; mitigazione danno; arretramento; morte del giocatore; rimozione dell'entità e conversione in ostacolo.
- Limiti: l'agente sceglie in base all'incertezza, ma non deduce concetti semantici come “attacco”, “difesa” o “minaccia”; conserva associazioni fra controllo, contesto e variazioni osservabili. Non ha un modello dell'avversario né pianifica per sconfiggerlo. Quattro seed non sono conclusi entro 200 azioni e uno ha perso, con eliminazione simultanea dell'entità. L'esperimento dimostra apprendimento contestuale elementare, non scoperta autonoma dell'algoritmo o generalizzazione.

### Esperimento 14 - Effetto della memoria persistente tra partite

- Data: 2026-10-27
- Obiettivo: misurare se la memoria persistente cambia la prestazione, non soltanto il numero di prove ripetute.
- Metrica primaria: tasso di vittoria entro 200 turni. Metriche secondarie: sconfitte, timeout, turni medi, azioni dedicate ai controlli e risultati appaiati per seed.
- Metodo: confronto sugli stessi 40 seed (0–39), massimo 200 turni per partita, danno 3 PV. La baseline riparte senza prove a ogni episodio; la condizione persistente usa un database SQLite temporaneo condiviso, inizialmente vuoto. Ogni condizione parte dallo stesso insieme di seed; il database di benchmark è eliminato a fine test.
- Baseline volatile: 25/40 vittorie (62,5%; IC Wilson 95% 47,0–75,8%), 5 sconfitte e 10 timeout; media 123,2 turni e 39,95 azioni di controllo per partita.
- Memoria persistente: 22/40 vittorie (55,0%; IC Wilson 95% 39,8–69,3%), 1 sconfitta e 17 timeout; media 127,2 turni e 2,58 azioni di controllo per partita.
- Confronto appaiato: la memoria ha trasformato 4 esiti non-vittoria in vittorie, ma ha anche trasformato 7 vittorie della baseline in non-vittorie; 18 seed sono vittorie in entrambe le condizioni. La differenza del tasso di vittoria è −7,5 punti percentuali con memoria persistente.
- Conclusione: la memoria riduce drasticamente le prove di controllo e le sconfitte, ma aumenta i timeout e non migliora il tasso di vittoria in questa prova. Gli intervalli di confidenza sono ampi e sovrapposti; 40 seed non bastano a dimostrare una differenza statisticamente affidabile. La metrica primaria da mantenere per i prossimi confronti è il tasso di vittoria entro 200 turni, affiancato dai timeout e dal numero di prove per verificare il costo dell'apprendimento.
- Interpretazione tecnica: il learner riutilizza le associazioni e smette di esplorare i controlli quando raggiunge la soglia, ma non sceglie in base agli effetti favorevoli né sa se il controllo appreso è utile per il successo. Meno esperimenti non equivale a migliore adattamento; la maggiore quota di timeout è compatibile con un apprendimento prematuro che riduce l'esplorazione senza migliorare la decisione.
- Limiti: i seed sono eseguiti in ordine crescente e la memoria è cumulativa, perciò l'esito dipende dall'ordine; occorrono repliche con ordini diversi e una metrica di avanzamento verso l'obiettivo per distinguere lo stallo dal timeout vicino al traguardo.

### Esperimento 15 - Selezione dei controlli guidata dagli esiti

- Data: 2026-10-27
- Obiettivo: fare in modo che le informazioni memorizzate influenzino la selezione delle azioni e limitare prove ripetute senza effetto.
- Modifica: sostituita la soglia di arresto dopo un numero fisso di prove con un selettore UCB contestuale che confronta l'utilità osservata; l'utilità usa variazione del progresso verso l'obiettivo, variazione dello stato del giocatore e stato terminale. Le prove senza variazioni osservabili portano all'astensione dopo tre tentativi per controllo. Gli esiti vengono salvati in SQLite per gioco.
- Metodo: stessi seed 0–39, massimo 200 turni, danno 3 PV; confronto tra memoria volatile e database persistente pulito condiviso tra seed.
- Risultati: entrambe le condizioni hanno prodotto 27/40 vittorie (67,5%), 0 sconfitte e 13 timeout; media turni 117,2 con memoria volatile e 116,7 con memoria persistente; media azioni di controllo 17,5 e 17,2 rispettivamente. Nessuno dei 40 esiti per seed è cambiato con la memoria persistente.
- Confronto storico: rispetto alla policy precedente testata sugli stessi 40 seed (25 vittorie, 5 sconfitte, 10 timeout in modalità volatile), la nuova policy ha avuto 2 vittorie in più e nessuna sconfitta, ma 3 timeout aggiuntivi. È un confronto deterministico sullo stesso insieme di seed, non una dimostrazione di miglioramento generalizzabile.
- Conclusione: la nuova policy usa un segnale di utilità per valutare i controlli e smette di ripetere test inerti; la persistenza non ha però modificato gli esiti in questo test. Il learner non impara ancora piani composti né modifica il planner BFS, che continua a decidere i movimenti.
- Limiti e prossimi criteri: l'utilità non assegna ancora credito a conseguenze ritardate come l'eliminazione di un'entità; l'esplorazione/UCB e l'astensione per prove inerti sono euristiche implementate nel codice, non algoritmi inventati dall'agente. Misurare il progresso verso il traguardo e ripetere gli esperimenti con più ordini di seed prima di attribuire causalità.

### Esperimento 16 - Apprendimento più generale dagli effetti osservati

- Data: 2026-10-27
- Obiettivo: togliere la dipendenza dal controllo codificato e dal contesto a distanza fissa, e usare le informazioni strutturate disponibili per valutare le azioni.
- Modifiche: il learner accetta i controlli effettivamente esposti dall'osservazione e l'ActionManager li inoltra senza richiedere registrazione individuale; confronta valori numerici, booleani, stringhe, nullità e variazioni strutturali presenti in osservazioni arbitrarie. Stima il progresso quando il World Model fornisce posizione e obiettivo, altrimenti si basa solo su esiti terminali espliciti. Le azioni e gli esiti restano contestuali e persistibili per gioco. Il movimento continua a essere deciso dal planner BFS.
- Metrica primaria: vittorie entro 200 turni. Metriche secondarie: sconfitte, timeout e media delle azioni di controllo.
- Risultati su seed 0–39 in ordine crescente, DB pulito: memoria volatile 17 vittorie, 19 sconfitte e 4 timeout; memoria persistente 25 vittorie, 4 sconfitte e 11 timeout. Le azioni di controllo medie sono 33,9 e 17,1 rispettivamente. La memoria ha cambiato l'esito su 15 seed.
- Replicazione della condizione persistente con seed in ordine decrescente e DB indipendente pulito: 27 vittorie, 4 sconfitte, 9 timeout; media 17,1 azioni di controllo. La differenza fra gli ordini mostra che l'apprendimento cumulativo è dipendente dalla sequenza delle esperienze.
- Conclusione: in queste prove la memoria persistente ha migliorato il tasso di vittoria rispetto alla memoria volatile e ridotto le sconfitte, pagando con più timeout. È un segnale promettente, non conclusivo: la baseline volatile ha un'elevata variabilità e la singola sequenza non separa l'effetto dell'ordine da quello della conoscenza.
- Limiti: l'attribuzione di cambiamenti all'ultima azione è correlazionale; non gestisce ancora credito ritardato in modo esplicito, non sceglie strategie di movimento e non misura ogni possibile aspetto del gioco. “Tutte le variabili” indica qui tutti i valori primitivi/strutturali ricevuti nell'osservazione, non informazioni non osservate o non estraibili. Il criterio di progresso disponibile è ancora spaziale; in ambienti senza target strutturato, servono segnali terminali espliciti.

### Esperimento 17 - Sei controlli anonimi e sequenze appaiate

- Data: 2026-10-27
- Obiettivo: verificare il gioco in cui GM riceve controlli anonimi invece di nomi espliciti di movimento e combattimento, e misurare l'effetto della memoria tra partite.
- Interfaccia: l'osservazione espone `control_1`–`control_6`; la mappatura interna del simulatore non viene trasmessa a GM. Il learner tratta i sei input come controlli opachi.
- Metodo: tre ordini casuali e riproducibili dei seed 0–39; per ogni ordine, stessi seed nella condizione volatile e nella condizione persistente. Massimo 200 turni per partita; database persistente temporaneo e indipendente per ciascuna sequenza.
- Risultati: memoria volatile, in tutte e tre le sequenze, 5 vittorie, 17 sconfitte e 18 timeout su 40; media 146,7 turni. Memoria persistente: sequenza 1, 4 vittorie/4 sconfitte/32 timeout, 182,2 turni medi; sequenza 2, 3/5/32, 182,1 turni medi; sequenza 3, 4/4/32, 175,4 turni medi. Il risultato è cambiato rispetto alla condizione volatile su 25, 25 e 22 seed rispettivamente.
- Verifiche: `python -m pytest -q` (38 test superati); `git diff --check` senza errori.
- Conclusione: i controlli anonimi sono integrati, ma la persistenza non migliora i risultati in queste sequenze: porta a molte meno sconfitte, ma anche a molte meno vittorie e più timeout. L'esito dipende dall'ordine di apprendimento; non si può concludere che la memoria migliori la prestazione. La baseline volatile è identica nelle tre sequenze perché riparte senza conoscenze pregresse e il comportamento è deterministico.
- Limiti e prossimi criteri: il test misura la policy corrente, che può scegliere direttamente tra input opachi, non la capacità di pianificare movimenti composti. I dati non dimostrano che il learner abbia dedotto il significato dei controlli; associazioni tra input e cambiamenti osservati possono essere confondenti. Prima di ampliare il benchmark, occorre migliorare l'uso degli effetti appresi per orientare l'azione verso obiettivi e distinguere progresso, stallo e semplice sopravvivenza.

### Esperimento 18 - Pianificazione direzionale tramite effetti appresi

- Data: 2026-10-27
- Obiettivo: fare usare al learner gli effetti osservati per realizzare la direzione scelta dal planner, senza fornire la corrispondenza tra nome anonimo e semantica dell'input.
- Modifica: il planner continua a scegliere il passo BFS verso l'obiettivo visibile o una frontiera; il learner confronta la posizione dell'agente prima/dopo ogni controllo e associa al controllo lo spostamento osservato. Se conosce un controllo che produce spostamento nella direzione desiderata lo seleziona; altrimenti prova controlli non mappati o controlli senza effetto provati da posizioni diverse. Quattro origini distinte senza spostamento sono considerate evidenza di un controllo non locomotivo. Le stime sono persistite nelle prove del gioco e non sono etichette semantiche.
- Test: test mirati sulla scoperta della direzione, sul riuso del controllo corrispondente e sul non ripetere all'infinito un input senza spostamento; suite completa: 40 test superati.
- Metodo: tre ordini casuali riproducibili degli stessi seed 0–39; confronto appaiato tra memoria volatile e database persistente pulito e indipendente per sequenza; limite 200 turni.
- Risultati volatile: in tutte le sequenze 25 vittorie, 0 sconfitte e 15 timeout su 40; media 132,2 turni/azioni di controllo.
- Risultati persistenti: sequenza 1, 22 vittorie/0 sconfitte/18 timeout, media 127,8 turni e 124,0 azioni di controllo; sequenza 2, 21/0/19, media 126,6 turni e 122,7 azioni; sequenza 3, 22/0/18, media 126,2 turni e 122,4 azioni. Gli esiti sono cambiati su 19, 18 e 19 seed rispetto alla modalità volatile.
- Conclusione: il learner ora collega una direzione di piano a controlli anonimi osservandone l'effetto sullo stato, e la persistenza riduce di circa 8–10 il numero medio di azioni di controllo. Tuttavia, in questo benchmark non aumenta le vittorie: ne produce 21–22 contro 25 in modalità volatile, con più timeout. È un progresso architetturale, non una dimostrazione di miglioramento del tasso di successo.
- Limiti e prossimi criteri: la politica di esplorazione delle frontiere e il bersaglio che cambia frequentemente limitano il completamento. L'euristica delle quattro posizioni per classificare un controllo senza movimento può confondere un input bloccato da ostacoli con un input non locomotivo; le stime restano specifiche del gioco. Il prossimo miglioramento deve distinguere blocco locale da effetto stabile e verificare il progresso sul bersaglio dinamico, mantenendo il confronto appaiato.

## Registro da aggiornare

Per ogni nuovo esperimento, aggiungere:
- data;
- descrizione;
- input simulato o reale;
- modello/algoritmo usato;
- risultato e metriche;
- regressioni o problemi;
- decisione successiva.

## Nota

Il log deve essere usato come base per prendere decisioni tecniche e per evitare di confondere un comportamento casuale con un reale miglioramento.
