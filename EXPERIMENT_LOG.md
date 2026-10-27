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
