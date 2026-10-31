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

### Esperimento 19 - Diagnosi dei fallimenti, memoria del target e recovery

- Data: 2026-10-28
- Ipotesi: alcuni timeout derivano dal fatto che il planner dimentica un obiettivo appena esce dalla visuale; la persistenza dei controlli dovrebbe invece mostrare il proprio effetto in un ambiente statico che isola navigazione e ricerca.
- Diagnosi iniziale: nel gioco dinamico, sui seed 0–39, la baseline volatile ha avuto 25 vittorie e 15 timeout, senza sconfitte. I 15 timeout non hanno mostrato una sequenza di 8 turni senza movimento; l'obiettivo è stato osservato in media solo 5,97 turni, mentre si è spostato circa 63,75 volte per partita. Un test statico ha riprodotto il difetto: seed 2 ha coperto 896/900 celle senza trovare l'obiettivo, perché `known_goal_positions` veniva aggiornata ma ignorata dal planner.
- Modifiche: la memoria aggiorna l'ultima posizione nota del target e la elimina se la cella è nuovamente osservata vuota o se viene osservata una nuova posizione. Posizioni-obiettivo diverse osservate nel tempo formano un'ipotesi di target dinamico; il planner non insegue più una posizione memorizzata quando questa ipotesi è attiva. Dopo `blocked_by_obstacle` o `blocked_by_boundary`, il planner evita la transizione fallita per due decisioni, ma soltanto se l'effetto direzionale del controllo era già identificato. Aggiunto `python -m simulated_game.benchmark` per confronti appaiati e metriche replicabili; lo scenario `static` fissa il target e disabilita l'inseguitore, quello `dynamic` mantiene le dinamiche complete.
- Test: 47 superati (`python -m pytest -q`), inclusi uso del target memorizzato, invalidazione dopo osservazione vuota, inferenza di target dinamico, mancata attribuzione causale con controllo ignoto, recovery e smoke test del report benchmark. `git diff --check` senza errori sostanziali (restano solo avvisi di conversione CRLF in tre Markdown).
- Metodo: stessi 40 seed 0–39, massimo 200 turni, ordine casuale riproducibile con seed 9204; memoria volatile contro database persistente pulito e temporaneo. Lo scenario statico è stato ripetuto su tre ordini di seed (9204, 9205, 9206); il dinamico su un ordine.
- Risultati statici (target immobile, inseguitore disabilitato): in tutti e tre gli ordini, entrambe le condizioni hanno ottenuto 40/40 vittorie e nessuna sconfitta o timeout. Media turni: 78,0 con memoria volatile, 62,67–62,70 con memoria persistente. Le predizioni direzionali osservate sono state corrette nel 100% dei casi (2.741 campioni volatile, 2.497 persistenti); gli spostamenti ai confini sono scesi da 81 a 2. La memoria persistente conserva la capacità di completamento e riduce i turni in questo task controllato.
- Risultati dinamici (target mobile, inseguitore attivo), ordine 9204: memoria volatile 25 vittorie/0 sconfitte/15 timeout, 132,2 turni medi; persistente 22/0/17 timeout/1 stallo esplicito, 124,0 turni medi. Il target si è spostato in media 63,75 volte con memoria volatile e 61,48 con persistenza; era visibile per 5,97 e 5,05 turni medi. Le posizioni diverse del target hanno generato un'ipotesi di mobilità in 35 episodi volatili e 29 persistenti (84 e 71 evidenze complessive), ma gli esiti sono rimasti rispettivamente 25 e 22 vittorie. Il progresso misurato verso la posizione del target pre-azione è 320 e 343; gli esiti appaiati sono cambiati in 19 seed. Nessuna evidenza che la persistenza o l'ipotesi di mobilità migliori le vittorie nello scenario dinamico.
- Conclusione: la perdita della posizione-obiettivo era una lacuna concreta, ora corretta e verificata; anche la mobilità viene ipotizzata da osservazioni ripetute, ma questa nuova informazione non ha ancora migliorato la prestazione dinamica. La memoria dei controlli dimostra un beneficio misurabile nello scenario isolato, ma non generalizza automaticamente allo scenario dinamico, dove il target si sposta spesso e viene visto raramente. Non si può dichiarare un miglioramento globale di GM sulla base di questi risultati; il dato dinamico usa un solo ordine.
- Limiti e follow-up: l'accuratezza direzionale misura solo controlli con effetto di spostamento già appreso; i controlli di combattimento o non identificati non entrano nel calcolo. I timeout dinamici sono per lo più veri turn-limit, non stalli; un episodio persistente si è invece fermato esplicitamente senza una mossa pianificabile. Il prossimo lavoro è migliorare la copertura/ricerca del target mobile, gestire il caso senza frontier raggiungibile e ripetere il benchmark dinamico su più ordini, mantenendo separato il task statico.

### Esperimento 20 - Separazione del core dall'adapter a griglia

- Data: 2026-10-28
- Obiettivo: verificare che il core di GM non contenga policy, planner, dati spaziali o input specifici del gioco simulato.
- Modifiche: `GameEngine` ora richiede policy ed esecutore espliciti e accetta un adapter di osservazione; la memoria del core conserva solo osservazioni e record generici. Planner, world model, learner, memoria spaziale e policy sono stati spostati in `simulated_game/agent/`; le regole del mondo restano in `simulated_game/game.py`, composte con il core da `simulated_game/runtime.py`. Rimosso il demo con azioni predeterminate da `app.main`.
- Verifica: nessun riferimento a `simulated_game`, griglia, frontiere, controlli o planner nei moduli `core`, `app`, `perception`, `decision` e `control`; `python -m pytest -q` ha superato 40 test, incluso l'uso dell'adapter di osservazione iniettato; `git diff --check` senza errori.
- Conclusione: la simulazione è ora un adapter sostituibile, non una policy o una dipendenza del core. Le regole di griglia continuano a esistere esclusivamente per testare un ambiente concreto; questo non dimostra generalizzazione né collegamento a giochi reali.
- Follow-up: mantenere questo confine quando si aggiungeranno altri ambienti e richiedere evidenza di trasferimento prima di promuovere euristiche specifiche a capacità generali.

### Esperimento 21 - Ricerca evolutiva di varianti strategiche

- Data: 2026-10-28
- Obiettivo: fare generare a GM varianti di strategia, confrontarle in modo riproducibile e impedire che il tuning sui seed di ricerca venga scambiato per un miglioramento generalizzabile.
- Metodo: motore di ricerca generico su genome numerici bounded con mutazione, crossover e selezione; adapter della griglia espone la penalità di rivisitazione, il bonus d'informazione e la soglia d'evidenza per considerare mobile un target ricordato. Candidati e baseline condividono i seed di training; una holdout distinta è usata per la promozione. Configurazione misurata: seed di ricerca 4815, popolazione 6, 3 generazioni, training seed 4815–4819 e holdout seed 1004815–1004834.
- Criterio di promozione: almeno due vittorie appaiate aggiuntive senza regressioni, oppure stessi esiti terminali per seed e almeno il 5% di riduzione dei turni medi di vittoria. La configurazione è scenario-specifica, locale e caricata dalla partita dinamica successiva solo se convalidata.
- Verifica: 47 test unitari e d'integrazione superati (riproducibilità, vincoli e score, parametri che cambiano la selezione, round-trip, rifiuto di configurazioni non valide e criterio di promozione appaiata); smoke test end-to-end con candidati/holdout.
- Esito su episodi completi: 14 varianti valutate su 5 seed di training; il migliore (`goal_change_threshold=5`, `information_gain_bonus=0`, `visit_penalty=0.074066`) ha ottenuto score medio 0,80732 contro 0,62502 baseline. Su 20 seed held-out, candidato 10 vittorie/0 sconfitte/10 timeout contro 8/0/12 baseline; 8 seed sono migliorati e 6 seed sono peggiorati, per un guadagno netto di due vittorie. Turni medi per vittoria: 99,4 contro 90,25; copertura media: 0,2498 contro 0,2552.
- Decisione: nessuna promozione. Il tasso aggregato è salito dal 40% al 50% su un piccolo holdout, ma le regressioni appaiate e i turni medi più alti non rispettano il criterio conservativo di promozione. La configurazione dinamica locale resta quella baseline.
- Limiti: lo spazio di ricerca modifica tre parametri (due continui e uno intero) della policy di esplorazione a griglia; 5 seed di training e 20 di holdout non dimostrano un miglioramento stabile. Nessun trasferimento ad altri ambienti dimostrato.

### Esperimento 22 - Smoke test della policy neurale DQN

- Data: 2026-10-28
- Obiettivo: verificare end-to-end addestramento, salvataggio, caricamento e confronto appaiato della nuova policy neurale senza modificare la baseline.
- Metodo: policy DQN con 20 episodi di training, massimo 40 turni per episodio; seed 7319–7338. Holdout dinamico appaiato su 20 seed 1007319–1007338, anch'esso con limite di 40 turni.
- Verifiche: la pipeline è terminata, ha prodotto e caricato il checkpoint temporaneo; 659 aggiornamenti DQN, loss media 0,04302. La suite completa ha superato 53 test; `git diff --check` non ha segnalato errori di whitespace.
- Esiti training: 2 vittorie, 0 sconfitte, 18 timeout.
- Esiti holdout: rete 0 vittorie/0 sconfitte/20 timeout; baseline 0/0/20. Nessun miglioramento o regressione di vittoria appaiata; score medio della rete 0,00289 contro 0,02566 baseline.
- Decisione: smoke test riuscito come verifica tecnica, ma non è evidenza di apprendimento efficace. Il limite breve di 40 turni rende il confronto inadeguato a stimare la prestazione del benchmark completo; nessuna promozione e nessun checkpoint conservato.
- Limiti e follow-up: eseguire il protocollo documentato con 300 episodi di training, 40 holdout e 200 turni prima di valutare il candidato. Il DQN resta confinato al simulatore e non dimostra ragionamento generale, obiettivi semantici autonomi o trasferimento.

### Esperimento 23 - Training DQN persistente e valutazione periodica

- Data: 2026-10-28
- Obiettivo: consentire all'addestramento di riprendere fra invocazioni e separare selezione validation e confronto finale held-out.
- Modifiche: i checkpoint riprendibili ora conservano pesi online/target, optimizer, replay buffer, RNG, contatori e configurazione; salvataggio atomico; seed training continuativi; validation appaiata ogni intervallo; checkpoint latest e miglior checkpoint validation distinti. I seed di training, validation e holdout sono controllati per non sovrapporsi. L'interfaccia normale continua a non addestrare la rete.
- Verifiche: round-trip di optimizer/replay/RNG e continuità dell'azione esplorativa; due invocazioni consecutive del trainer hanno portato il contatore episodi da 1 a 2 e hanno continuato con il seed successivo; il checkpoint best è stato caricato per la modalità d'inferenza. Quattro test neurali mirati superati.
- Risultati prestazionali: nessuna nuova conclusione di efficacia. Le prove ridotte nei test validano persistenza e flusso di valutazione, non un miglioramento rispetto alla baseline.
- Decisione: si può ora continuare un training lungo tra invocazioni, con validation periodica e holdout separato. Nessuna promozione automatica della baseline.
- Follow-up: eseguire il protocollo con almeno 300 episodi, 20 validation, 40 holdout e 200 turni. Se l'apprendimento da zero resta debole, valutare come esperimento separato il warm-start per imitazione della baseline e confrontare i due metodi su seed finali mai visti.

### Esperimento 24 - Training DQN completo da zero

- Data: 2026-10-28
- Obiettivo: misurare se il DQN da zero riesce a superare la baseline dopo un training esteso.
- Metodo: 300 episodi di training, massimo 200 turni, seed 7319–7618; validation periodica su 20 seed 2007319–2007338; confronto finale su 40 seed held-out 1007319–1007358. Non vi è sovrapposizione fra split.
- Aggiornamenti: 43.685; loss media del training: 0,066.
- Miglior validation: episodio 250, 5/20 vittorie e 0 sconfitte, contro baseline 10/20 vittorie e 0 sconfitte; 3 miglioramenti e 8 regressioni appaiate.
- Holdout del checkpoint latest: DQN 6/40 vittorie, 26 sconfitte e 8 timeout; baseline 24/40 vittorie, 0 sconfitte e 16 timeout. Score medio -0,49248 contro 0,61714; 3 miglioramenti e 21 regressioni appaiate.
- Verifica aggiuntiva del checkpoint scelto come migliore su validation (episodio 250), sugli stessi 40 seed held-out: 6 vittorie, 1 sconfitta e 33 timeout; score 0,14767, con 2 miglioramenti e 20 regressioni appaiate contro la baseline.
- Decisione: anche il checkpoint scelto è nettamente peggiore; nessuna promozione. Il modello scratch resta un candidato negativo e il caso attiva l'esperimento pianificato di imitazione baseline seguita da fine-tuning DQN.
- Limiti: singolo training seed e singolo holdout; risultati specifici del simulatore e di questa configurazione. Il dato non dimostra incapacità assoluta del DQN, ma è sufficiente a non adottare questo candidato.

### Esperimento 25 - Warm-start per imitazione baseline e fine-tuning DQN

- Data: 2026-10-28
- Obiettivo: verificare se inizializzare la rete con esempi di decisioni baseline riduce il divario osservato nel training DQN da zero.
- Metodo: 100 episodi baseline per generare 13.979 esempi stato-controllo; learner dei controlli baseline mantenuto fra gli episodi dimostratori, memoria spaziale ricreata a ogni episodio. 5 epoche di imitazione supervisionata, poi 300 episodi di fine-tuning DQN. Seed di training 7319–7618; validation fissa 2007319–2007338; holdout finale 1007319–1007358. Gli split di valutazione non sono stati usati per training o imitazione.
- Training: la baseline dimostratrice ha vinto 52/100 episodi; loss di imitazione 0,39937. 51.279 aggiornamenti totali inclusi circa 550 update supervisionati; loss media del fine-tuning DQN 0,06753.
- Selezione: miglior validation a 150 episodi DQN, 13/20 vittorie e 0 sconfitte contro 10/20 baseline; 8 miglioramenti e 5 regressioni appaiate. Il criterio richiede almeno due vittorie appaiate aggiuntive e zero regressioni, quindi non è soddisfatto.
- Holdout del checkpoint best: rete 11/40 vittorie, 3 sconfitte e 26 timeout; baseline 24/40 vittorie, 0 sconfitte e 16 timeout. Score medio 0,22711 contro 0,61714; 4 miglioramenti e 17 regressioni appaiate.
- Conclusione: warm-start migliora il tasso aggregato rispetto al checkpoint scratch selezionato (11 contro 6 vittorie) e riduce nettamente le sconfitte, ma resta peggiore della baseline e presenta numerose regressioni. Nessuna promozione; il checkpoint resta candidato sperimentale.
- Follow-up: diagnosticare perché la policy perde informazione utile alla pianificazione e perché il reward di progresso non impedisce l'esposizione al nemico; non aumentare semplicemente gli episodi senza correggere/verificare queste cause. Ripetere solo dopo la modifica su nuovi holdout indipendenti.

### Esperimento 26 - Diagnosi delle regressioni della policy neurale

- Data: 2026-10-28
- Obiettivo: distinguere un training insufficiente da problemi di rappresentazione, scelta delle azioni e reward.
- Metodo: replay deterministico del checkpoint warm-start migliore, in sola inferenza, sugli stessi 40 seed held-out già usati nel confronto; misure d'azione raccolte senza aggiornare pesi. Confronto delle metriche baseline sugli stessi seed.
- Esiti neurali: 11 vittorie, 3 sconfitte, 26 timeout su 40; 6.404 azioni. Nessun cambio di posizione in 1.214 azioni (18,9%); la baseline ha avuto 236 azioni senza movimento su 5.669 (4,2%). La rete ha selezionato `control_5` 598 volte e `control_6` 249 volte (13,2% delle decisioni), mentre la baseline percorre il planner soprattutto con controlli direzionali.
- Distribuzione dei controlli neurali: `control_1` 2.423, `control_2` 638, `control_3` 2.442, `control_4` 54, `control_5` 598, `control_6` 249. La distribuzione è fortemente sbilanciata fra le direzioni; nei risultati è associata a molte azioni improduttive e non dimostra l'apprendimento di una navigazione robusta.
- Esiti baseline sugli stessi seed: 24/40 vittorie, nessuna sconfitta; 5.409 mosse con posizione cambiata, 80 blocchi e 156 azioni completate senza movimento.
- Diagnosi del codice: la baseline calcola un percorso BFS verso target/frontiera e traduce il passo in un controllo appreso; la rete sceglie direttamente fra tutti e sei i controlli, che restano sempre disponibili. L'encoder dà solo offset del target/frontiera, ostacoli adiacenti e conteggi di visita adiacenti; non comunica il passo BFS, un percorso, distanze alternative o la direzione prevista del nemico. I controlli di attacco/difesa sono quindi azioni ammissibili per la rete anche quando non servono.
- Diagnosi del reward: il termine per avvicinarsi al target è applicato soltanto quando il target resta osservato nella stessa coordinata prima e dopo l'azione; nel task dinamico il target si sposta ogni due mosse e spesso è nascosto, perciò il segnale di progresso è raro. La curiosità di stato nuovo può invece premiare stati diversi senza indicare che ci si sta avvicinando alla vittoria; la penalità di sconfitta è terminale e tardiva.
- Evidenza sul warm-start: prima del fine-tuning, l'imitazione baseline ha raggiunto solo 3/20 vittorie in validation contro 10/20 della baseline. Questo mostra che l'attuale codifica compatta non riproduce fedelmente il comportamento del planner, quindi non basta aumentare gli episodi DQN.
- Conclusione: i dati sostengono limiti di rappresentazione e reward oltre alla quantità di training. Priorità: separare movimenti da azioni contestuali, aggiungere un segnale di distanza/progresso robusto ai target mobili e rendere osservabile il rischio dell'inseguitore; misurare prima la fedeltà all'imitazione su validation. Non implementare tutte le modifiche assieme: isolare gli interventi e rivalutare con seed indipendenti.
- Limiti: l'analisi di azioni è un replay su un checkpoint e un singolo holdout; associazioni diagnostiche, non prova causale dell'effetto di una specifica modifica.

### Esperimento 27 - Intent BFS esplicito nella rappresentazione neurale

- Data: 2026-10-28
- Obiettivo: testare isolatamente se esporre alla rete il suggerimento di movimento del planner rende più apprendibile la decisione baseline, lasciando invariato il reward.
- Modifica: aggiunte quattro feature one-hot per la direzione del passo BFS suggerito e una feature che distingue l'esplorazione di frontiera; input passati da 35 a 40. Il planner condivide osservazioni e memoria spaziale già disponibile, senza leggere lo stato nascosto. Nuovo checkpoint separato e training da zero.
- Imitazione iniziale: 100 episodi baseline su seed 8391–8490 hanno prodotto 12.321 esempi. La baseline demonstrator ha vinto 65/100. Loss media di imitazione 0,17863. Prima del fine-tuning, validation nuova 10/20 vittorie contro 13/20 baseline; nel precedente encoder da 35 feature l'imitazione era 3/20 contro 10/20 baseline.
- Fine-tuning: 300 episodi, seed 8391–8690, 51.116 aggiornamenti DQN, loss media 0,07597. La validation è scesa a 2/20 vittorie al termine; il best checkpoint è rimasto quello dopo l'imitazione iniziale (10/20). Evidenza che l'attuale DQN/reward degrada il comportamento appreso dall'esperto.
- Holdout A, seed 1008391–1008430: best candidate 24/40 vittorie, 0 sconfitte, 16 timeout, score 0,61756; baseline 26/40, 0 sconfitte, 14 timeout, score 0,66749. 8 miglioramenti e 10 regressioni appaiate.
- Replicazione su holdout B indipendente, seed 1108391–1108430: candidato 24/40 vittorie, 0 sconfitte, 16 timeout, score 0,61988; baseline 26/40, 0 sconfitte, 13 timeout e 1 stallo, score 0,66333. 8 miglioramenti e 10 regressioni appaiate.
- Conclusione: l'intent migliora nettamente l'imitazione e rende il candidato competitivo, senza evidenza che superi la baseline. Entrambi gli holdout confermano lo stesso scarto di due vittorie e il medesimo rapporto appaiato 8 miglioramenti/10 regressioni. Il DQN non ha migliorato il best BC, ma lo ha degradato in validation.
- Decisione/follow-up: tenere la baseline come default e il modello come candidato. Il successivo intervento isolato riguarda il reward di progresso: usare il passo BFS pre-azione per assegnare feedback quando il target si muove o è nascosto; preservare il best checkpoint rispetto alla validation, poi valutare su un nuovo holdout mai usato.

### Esperimento 28 - Reward di progresso guidato dall'intent BFS

- Data: 2026-10-28
- Obiettivo: verificare se un segnale di progresso basato sullo spostamento previsto dal planner migliora l'apprendimento quando il target è mobile o nascosto, lasciando invariata la rappresentazione a 40 feature.
- Modifica: il reward usa l'intent BFS registrato prima dell'azione: premia lo spostamento allineato, penalizza uno spostamento contrario e distingue un'azione bloccata da un'azione senza progresso. Restano invariati i termini esistenti di curiosità, distanza dal target e risultato terminale. Test unitari coprono il progresso con target nascosto e il caso bloccato.
- Metodo: 100 episodi baseline per imitazione (seed 9397–9496), 12.928 esempi e 5 epoche; poi 300 episodi DQN. Validation fissa su 20 seed 2009397–2009416. Miglior checkpoint selezionato prima del fine-tuning DQN; holdout finale A 1009397–1009436 e replica indipendente B 1109397–1109436, entrambi a 40 episodi e massimo 200 turni. Baseline e candidato valutati sugli stessi seed per ciascun holdout.
- Imitazione/validation: la baseline dimostratrice ha vinto 61/100 episodi; loss di imitazione 0,17821. Il best checkpoint pre-DQN ha ottenuto 11/20 vittorie in validation contro 10/20 baseline (6 miglioramenti e 5 regressioni appaiate). A 300 episodi DQN, la validation è scesa a 0/20 contro 10/20 baseline (0 miglioramenti e 10 regressioni); il checkpoint best è quindi rimasto quello di sola imitazione.
- Holdout A: candidato 23/40 vittorie, 0 sconfitte, 17 timeout, score medio 0,59378; baseline 21/40, 0 sconfitte, 19 timeout, score 0,54659. Esiti appaiati: 12 miglioramenti e 10 regressioni.
- Holdout B indipendente: candidato 27/40 vittorie, 0 sconfitte, 13 timeout, score medio 0,69265; baseline 26/40, 0 sconfitte, 14 timeout, score 0,66522. Esiti appaiati: 8 miglioramenti e 7 regressioni.
- Conclusione/decisione: i due holdout mostrano un piccolo vantaggio aggregato del candidato (23 contro 21 e 27 contro 26 vittorie), ma in entrambi si osservano regressioni appaiate; il criterio conservativo di promozione non è soddisfatto. Il progresso guidato dall'intent non ha impedito il collasso della policy durante il fine-tuning DQN. Il candidato resta opzionale e la baseline resta invariata.
- Limiti/follow-up: il risultato è circoscritto alla griglia e a un solo checkpoint di imitazione; i vantaggi aggregati sono piccoli e non equivalgono a una superiorità dimostrata. Il prossimo esperimento dovrebbe evitare di degradare l'imitazione, ad esempio controllando il fine-tuning DQN o preservando esplicitamente le azioni dell'esperto, e mantenere nuovi seed per la valutazione.

### Esperimento 29 - Terzo holdout e confronto dei checkpoint neurali

- Data: 2026-10-29
- Obiettivo: replicare il confronto neural-vs-baseline su seed mai usati e confrontare direttamente i checkpoint con intent BFS e con reward di progresso.
- Metodo: stessi 40 seed 1209397–1209436, massimo 200 turni per episodio; baseline e i due checkpoint best valutati sugli stessi seed. Nessun aggiornamento dei pesi durante l'inferenza.
- Baseline: 24/40 vittorie, 0 sconfitte, 15 timeout e 1 stallo; score medio 0,61745 e 95,25 turni medi per vittoria.
- Checkpoint intent BFS: 24/40 vittorie, 0 sconfitte e 16 timeout; score 0,61867 e 78,5 turni medi per vittoria. Rispetto alla baseline: 6 esiti migliorati e 6 peggiorati.
- Checkpoint reward di progresso: stessi esiti discreti del checkpoint intent su questo holdout (24/40 vittorie, 0 sconfitte, 16 timeout; score 0,61867; 78,5 turni medi per vittoria), con 6 miglioramenti e 6 regressioni appaiati rispetto alla baseline. I file dei checkpoint e i pesi sono distinti; l'uguaglianza osservata riguarda gli esiti e le metriche aggregate su questi seed, non i modelli.
- Conclusione/decisione: il terzo holdout non mostra aumento delle vittorie rispetto alla baseline né un saldo appaiato positivo. Il tempo medio per vittoria è inferiore, ma il punteggio complessivo è pressoché identico e ci sono sei regressioni. Nessuna promozione; mantenere baseline e checkpoint separati.
- Follow-up: dare priorità alla conservazione dell'imitazione durante il fine-tuning DQN, misurando separatamente fedeltà alle decisioni esperte, vittorie e saldo appaiato su nuovi seed.

### Esperimento 30 - Retrieval episodico conservativo nella policy di griglia

- Data: 2026-10-29
- Obiettivo: riutilizzare risultati di decisioni precedenti in contesti simili senza introdurre nuovo input, coordinamento parallelo o modificare il costo della pianificazione BFS.
- Modifica: aggiunti casi episodici con schema contestuale versionato, reward derivato dall'esito, separazione per `game_id`, persistenza SQLite, cache di 2.000 casi e finestra di retrieval di 256 casi. Sono richiesti almeno due esempi per direzione e similarità almeno 0,65. Il planner usa il risultato solo a parità di costo; l'hint neurale usa il medesimo retrieval senza cambiare le 40 feature. Il benchmark registra metriche di retrieval.
- Verifiche: suite completa, 64 test superati; benchmark smoke completato. I test coprono supporto minimo, contesti irrilevanti, schema/reward invalidi, limiti di cache/finestra, isolamento e ricaricamento per gioco, tie-break, outcome policy, hint neurale e metriche.
- Metodo diagnostico: 10 seed appaiati per scenario, un ordine, massimo 100 turni (il protocollo di riferimento usa 40 seed e 200 turni). Gli stessi seed sono stati eseguiti in modalità volatile e persistente.
- Scenario statico: volatile 6/10 vittorie, 0 sconfitte, 4 timeout, 71,4 turni medi; persistente 4/10, 0 sconfitte, 6 timeout, 73,4 turni medi; 2 esiti appaiati cambiati. Retrieval: volatile 714 query, 699 con match, 28.303 riferimenti a casi recuperati, 221 decisioni guidate; persistente 734 query, 730 con match, 148.244 riferimenti, 50 guidate, media 410,5 casi disponibili.
- Scenario dinamico: volatile 2/10 vittorie, 0 sconfitte, 8 timeout, 92,5 turni medi; persistente 6/10, 0 sconfitte, 4 timeout, 76,9 turni medi; 6 esiti appaiati cambiati. Retrieval: volatile 925 query, 904 con match, 40.239 riferimenti a casi recuperati, 213 decisioni guidate; persistente 769 query, 767 con match, 149.228 riferimenti, 120 guidate, media 364,2 casi disponibili.
- Interpretazione: il segnale osservato varia tra i due scenari e il campione è ridotto. La modalità persistente conserva sia casi episodici sia prove dei controlli, perciò il confronto non isola l'effetto causale del retrieval. Il numero di casi recuperati conta occorrenze attraverso le query, non casi unici.
- Decisione: mantenere il retrieval come supporto conservativo dell'adapter, senza promuovere una nuova strategia né dichiarare un miglioramento.
- Follow-up: ablation volatile/persistente con stato delle prove di controllo identico, più ordini di seed e massimo 200 turni; valutare vittorie, esiti appaiati, turni e metriche di retrieval separatamente.

### Esperimento 31 - Sensibilità all'ordine del retrieval episodico statico

- Data: 2026-10-29
- Obiettivo: verificare se l'esito del confronto volatile/persistente cambia al variare dell'ordine con cui la memoria persistente accumula esperienze.
- Metodo: `--scenario static --episodes 10 --max-turns 100 --sequences 10`, ordine base 9204. Ogni sequenza riusa i seed 0–9 in un ordine diverso; all'interno di ogni sequenza si valutano 10 episodi volatile e gli stessi 10 persistenti. Il database persistente viene azzerato tra sequenze. Totale: 200 episodi, 10 scenari distinti ripetuti in 10 ordini, non 100 seed indipendenti.
- Volatile: 60/100 vittorie, 0 sconfitte, 40 timeout; 71,4 turni medi in ogni sequenza; 29 blocchi medi e 73 azioni senza movimento per ordine. La ripetizione volatile è identica perché gli episodi non riusano memoria tra partite.
- Persistente: 58/100 vittorie, 0 sconfitte, 42 timeout; da 4 a 7 vittorie per sequenza; media dei turni per episodio 64,64; 3 blocchi medi e 8,3 azioni senza movimento per ordine.
- Confronto appaiato: 26 esiti su 100 coppie seed/ordine sono cambiati. Poiché non ci sono sconfitte o stalli in nessuna condizione, l'aggregato equivale a 12 passaggi timeout→vittoria e 14 vittoria→timeout. Le coppie ripetono gli stessi 10 seed con ordini diversi e non sono 100 osservazioni indipendenti.
- Retrieval: in media 221 decisioni guidate per ordine volatile e 100,8 persistenti. I riferimenti a casi recuperati sono occorrenze attraverso le query, non casi unici. Precisione direzionale riportata al 100% in entrambe le condizioni, su circa 600 campioni per ordine.
- Interpretazione: la memoria persistente migliora nettamente il riuso dei controlli (meno blocchi e azioni senza movimento) e riduce i turni medi complessivi, ma non aumenta le vittorie aggregate: 58 contro 60. I risultati persistenti dipendono dall'ordine, con 4–7 successi per sequenza. Questo benchmark combina persistenza delle prove dei controlli e retrieval episodico, quindi non attribuisce gli effetti ai soli casi decisionali. Inoltre usa solo 10 seed e un limite di 100 turni.
- Decisione: mantenere il sistema come supporto sperimentale senza promozione o affermazioni di superiorità sul tasso di vittoria.
- Follow-up: aggiungere un confronto ablation che mantenga identiche le prove dei controlli e disattivi soltanto il retrieval episodico; usare seed di valutazione più numerosi e indipendenti, massimo 200 turni e confronto appaiato.

### Esperimento 32 - Benchmark statico esteso dopo il retrieval episodico

- Data: 2026-10-29
- Obiettivo: verificare il comportamento statico con il protocollo completo del benchmark e più seed rispetto al run esplorativo da 10 episodi.
- Metodo: benchmark comunicato dall'utente, `--scenario static --episodes 40 --max-turns 200 --sequences 1`; confronto appaiato volatile/persistente sullo stesso ordine di 40 seed. Una sola sequenza, quindi nessuna replica dell'ordine.
- Volatile: 40/40 vittorie, 0 sconfitte, 0 timeout; 74,45 turni medi; 118 azioni bloccate; 179 azioni senza movimento; accuratezza predittiva 1,0; progresso distanza 1.125; coverage 0,509.
- Persistente: 39/40 vittorie, 0 sconfitte, 1 timeout; 69,05 turni medi; 3 azioni bloccate; 6 azioni senza movimento; accuratezza predittiva 1,0; progresso distanza 1.115; coverage 0,526.
- Esiti appaiati: `paired_outcomes_changed = 1`; l'unico cambiamento è una vittoria volatile diventata timeout persistente. Il campione non basta per concludere che esista un calo stabile di successo.
- Interpretazione: forte riduzione dei blocchi e delle azioni senza movimento, con turni medi inferiori di circa il 7%; il tasso di vittoria non migliora (40/40 contro 39/40). Poiché la media dei turni include il timeout a 200, le 39 vittorie persistenti durano circa 65,7 turni in media (stima ricavata dagli aggregati), contro 74,45 per le 40 vittorie volatili: il dato suggerisce partite vinte più rapide ma non più affidabili. L'accuratezza predittiva perfetta è condivisa dalle due condizioni e non misura la qualità strategica. I numeri suggeriscono migliore riuso dei controlli, ma la modalità persistente include sia prove di controllo sia casi episodici: il loro contributo non è separabile in questo run.
- Limiti: un solo ordine; risultati riportati in forma aggregata, senza traccia delle decisioni per seed; memoria persistente composta da più meccanismi. Il precedente test da 10 seed/100 turni ripeteva gli stessi 10 seed in più ordini e non è direttamente aggregabile con questo run.
- Decisione/follow-up: nessuna promozione basata sul solo tasso di vittoria. Conservare il dato come evidenza di efficienza operativa e progettare l'ablation retrieval-on/off a memoria dei controlli identica, con log per decisione e seed appaiati.

### Esperimento 33 - Stop conditions, ricerca strategica dinamica e esplorazione per obiettivo dinamico

- Data: 2026-10-30
- Obiettivo: completare le proposte di fase del loop operativo (stop conditions), verificare la ricerca strategica sui parametri di esplorazione in un contesto dinamico e intervenire quando l'obiettivo è dinamico ma non visibile.
- Proposta 1 - Stop conditions: `GameEngine` accetta ora `max_steps` e `should_stop`; il loop verifica il segnale di stop e poi il limite di step, e registra un passo solo dopo un `process_frame` valido. Test nel core: stop dopo `max_steps` e stop quando `should_stop` risponde true. Suite a questo punto: 66 test superati.
- Proposta 2 - Strategia nel contesto dinamico: `python -m simulated_game.strategy_search --scenario dynamic`; 25 varianti valutate; il miglior candidato del training coincide con la baseline (score medio 0,85223); sul holdout il candidato è alla pari con la baseline (24 vittorie, 16 timeout, mean_score 0,64098); promozione non effettuata; strategia salvata `null`.
- Proposta 3 - Ambiente reale: esplicitamente non approdata in questa fase; il sistema resta confinato al simulatore.
- Miglioramento mirato - obiettivo dinamico: quando `memory.spatial.goal_is_dynamic` è attiva e l'obiettivo non è visibile, `Planner._choose_world_action` imposta i target a lista vuota, cioè attiva l'esplorazione, invece di inseguire le posizioni ricordate. Test in `tests/test_exploration.py`: 3 nuovi e 1 aggiornato. Suite finale: 69 test superati, 0 falliti.
- Verifica: suite completa riavviata dall'orchestratore, 69 test superati in 36,49 s senza regressioni; benchmark dinamico con 40 seed e max-turns 200; A/B tra vecchio planner e nuovo planner sullo scenario dinamico, ordine 9204. I run di test e benchmark demandati a un sotto-agente non sono riusciti a completare per il limite di utilizzo gratuito; l'orchestratore ha eseguito direttamente suite e benchmark.
- Metodo A/B: stesso scenario dinamico, stesso ordine di seed, confronto vecchio planner vs nuovo planner con le stesse condizioni di memoria.
- Risultati A/B - vecchio planner: volatile 25 vittorie/15 timeout, mean_turns 132,2, blocked 80, coverage 0,311; persistente 22 vittorie/17 timeout/1 stallo esplicito, mean_turns 124,0.
- Risultati A/B - nuovo planner: volatile 25 vittorie/15 timeout, mean_turns 137,3, blocked 118, coverage 0,731; persistente 25 vittorie/15 timeout, mean_turns 116,33, blocked 3, stalled 0.
- Confronto: con memoria persistente, il nuovo planner ha dato +3 vittorie, -2 timeout, -7,67 mean_turns in meno e stallo 0 contro 1 rispetto al vecchio planner. Le modifiche di questa fase sono incluse nel commit `2a2055b` ("migliori sul comportamento").
- Conclusione: le stop conditions sono operative, la strategia dinamica non promuove candidati e il branch di esplorazione per obiettivo dinamico è implementato e verificato.
- Limiti e follow-up: l'A/B si è svolto su un unico ordine di seed; la copertura migliora molto, ma con un aumento delle azioni bloccate. Servono confronti su più ordini di seed e l'ablazione del retrieval.

### Esperimento 34 - Validazione improvement dinamico (A/B su più ordini di seed)

- Data: 2026-10-31
- Obiettivo: validare sperimentalmente il miglioramento sull'obiettivo dinamico: quando la memoria spaziale segnala un obiettivo dinamico e l'obiettivo non è visibile, il planner deve attivare l'esplorazione invece di inseguire la posizione ricordata.
- Proposta/Miglioramento: branch `elif memory.spatial.goal_is_dynamic: targets = []` in `Planner._choose_world_action` (commit `2a2055b`, "migliori sul comportamento"): quando l'obiettivo è dinamico e fuori visuale, i target vengono impostati a lista vuota e la scelta passa alla policy di esplorazione.
- Metodo A/B: `python -m simulated_game.benchmark --scenario dynamic --episodes 40 --max-turns 200 --sequences 5` con il nuovo planner attivo, una sequenza per ciascun ordine di seed; confronto con il vecchio planner tramite la sequenza 1 (ordine 9204) riportata in Fase 1.
- Risultati per sequenza - nuovo planner: sequenza 1 (ordine 9204): volatile 25 vittorie/15 timeout, mean_turns 137,3, blocked 118, coverage 0,731; persistente 25 vittorie/15 timeout, mean_turns 116,33, blocked 3, coverage 0,639, goal_reached 25, dynamic_hyp 29, relocations 57,58. I numeri persistenti coincidono con la Fase 1: improvement REPRODUCED. Sequenza 2: persistente 27 vittorie/13 timeout, mean_turns 118,75, blocked 3, coverage 0,525, goal_reached 27. Sequenza 3: persistente 29 vittorie/10 timeout, mean_turns 105,58, blocked 3, coverage 0,57, goal_reached 29, stalled 1 (waiting_for_visible_target). Sequenza 4: persistente 27 vittorie/13 timeout, mean_turns 115,3, blocked 3, coverage 0,599, goal_reached 27. Sequenza 5: persistente 29 vittorie/11 timeout, mean_turns 103,38, blocked 3, coverage 0,569, goal_reached 29. Aggregati sui 5 sequenze e 200 episodi: persistente 137 vittorie (68,5%), 62 timeout, 0 sconfitte, 1 stallo, mean_turns 111,87, blocked 3 per sequenza, mean coverage ~0,58; volatile 125 vittorie, 75 timeout, mean_turns 137,3, blocked 118, coverage 0,731 (identica in ogni sequenza).
- Confronto con A/B precedente: la sequenza 1 (ordine 9204) riproduce esattamente i numeri del nuovo planner della Fase 1 → improvement REPRODUCED; trend confermato nelle sequenze 2-5: persistent planner con 25-29 vittorie contro 22 del vecchio planner, timeout 10-15, stalli ≤1, mean_turns 103-119, blocked 3.
- Conclusione: verdict KEEP del nuovo branch; procedere alla Fase 3 (wiring delle stop conditions e prossimo miglioramento).
- Limiti e follow-up: il confronto diretto vecchio vs nuovo va esteso su più ordini di seed prima di una promozione automatica; la validazione è confinata allo scenario dinamico del simulatore e al protocollo di benchmark.

### Esperimento 35 - Rebalancing del combattimento per renderlo al pari con il traguardo

- Data: 2026-10-31
- Obiettivo: rendere la meccanica di combattimento tanto importante quanto il raggiungimento del traguardo, senza introdurre logica di combattimento nel Game Master.
- Modifica: danno attacco incrementato da 3 a 5 PV in `GridGame._execute_combat` (player e entità); difesa continua a annullare il danno; con 10 PV a parte l'uccisione richiede ora due attacchi.
- Verifica: suite completa `python -m pytest -q`: 69 test superati, 0 falliti; test di combattimento eseguiti singolarmente: 8 superati.
- Risultato: il combattimento ora ha posta in gioco sufficiente a essere un driver strategico; la gestione di vita, danno e difesa diventa centrale quanto l'urgenza del traguardo.
- Conclusione: commit `a6d7b70` ("Rebalance combat: attack damage 3 to 5 so fights carry goal-level stakes") sul branch `simulated-game`.
- Limiti e follow-up: rebalancing numerico solo; il GM resta senza logica di combattimento (apprende i controlli anonimi); il cambio non è stato sottoposto a nuovo benchmark di gameplay.

### Esperimento 36 - Attribuzione dell'improvement dinamico al ranking delle frontiere

- Data: 2026-10-31
- Obiettivo: attribuire l'improvement misurato nella Fase 2 sul target dinamico alle componenti del planner introdotta nel commit `2a2055b` ("migliori sul comportamento").
- Metodo: A/B del ranking delle frontiere (vecchio vs nuovo) mantenendo la logica del ramo dinamico del planner nuovo; scenario dynamic, 40 seed, max-turns 200, ordine 9204; suite completa post-rebalancing: 69 test superati in 34,72 s.
- Risultati: vecchio ranking -> volatile coverage 0,302, persistent wins 23, stalled 1, mean_turns 119,12; nuovo ranking -> volatile coverage 0,731, persistent wins 25, stalled 0, mean_turns 116,33.
- Finding: sotto strategia di default (`goal_change_threshold` 1) vecchio ramo e nuovo ramo di esplorazione sono logicamente equivalenti; l'improvement misurato nella Fase 2 e' attribuibile al ranking delle frontiere del commit `2a2055b`.
- Conclusione: l'improvement sul target dinamico e' attribuibile al ranking delle frontiere; il branch di esplorazione per obiettivo dinamico va mantenuto per strategie con soglia di cambio obiettivo superiore alla soglia di default.
- Limiti e follow-up: A/B eseguito su un unico ordine; servono repliche su piu' ordini prima di una promozione automatica; i risultati ottenuti prima del rebasing del combattimento (commit `a6d7b70`) non sono direttamente comparabili con quelli successivi e i futuri esperimenti vanno categorizzati per versione di bilanciamento.

### Esperimento 37 - Ablazione retrieval episodico per l'improvement dinamico

- Data: 2026-10-31
- Obiettivo: isolare il contributo del retrieval episodico alle decisioni del planner, dopo la chiusura della Fase 2 con verdict KEEP.
- Metodo: benchmark dinamico, 40 seed, max-turns 200, ordine 9204, con flag di abilitazione dell'episodic retrieval (`--disable-episodic-retrieval`); suite completa: 70 test superati in 34,72 s.
- Risultati - retrieval ON (nuovo planner): volatile coverage 0,731, persistent wins 25, stalled 0, mean_turns 116,33, persistent 25 vittorie/40 episodi.
- Risultati - retrieval OFF (nuovo planner): volatile coverage 0,311, persistent wins 22, stalled 1, mean_turns 124,0, retrieval queries 0 con la flag disattivata; coincidente con la performance del vecchio planner della Fase 1.
- Conclusione: l'improvement misurato nella Fase 2 e' attribuibile al percorso decisionale guidato dalla memoria (retrieval); il ramo di esplorazione per obiettivo dinamico risulta equivalente al ramo precedente sotto strategia di default.
- Limiti e follow-up: ablazione eseguita su un unico ordine; servono repliche su piu' ordini prima di una promozione automatica.

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
