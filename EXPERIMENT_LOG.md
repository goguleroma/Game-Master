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
