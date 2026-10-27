# ACCEPTANCE_TESTS.md

## Obiettivo

Definire i test minimamente necessari per verificare che il sistema stia effettivamente migliorando e non soltanto “girando”.

## Test di base

### 1. Test inizializzazione del motore

**Contesto:** il sistema viene creato senza un gioco reale.

**Risultato atteso:**
- GameEngine inizializzato;
- planner inizializzato;
- input adapter inizializzato;
- memory inizializzata.

### 2. Stato senza dati

**Input simulato:** `None` o stato vuoto.

**Risultato atteso:**
- stato non crasha;
- planner ritorna `wait`;
- input adapter esegue un risultato positivo per wait.

### 3. Menu con bottone start

**Input simulato:** `{"objects": ["start_button"], "goal": "start"}`

**Risultato atteso:**
- planner sceglie `click`;
- input adapter esegue click;
- memory registra azione e risultato.

### 4. Obiettivo move_left

**Input simulato:** `{"objects": ["player"], "goal": "move_left"}`

**Risultato atteso:**
- planner sceglie `move_left`;
- input adapter esegue movimento;
- risultato registrato come successo.

### 5. Oggetti sconosciuti

**Input simulato:** `{"objects": ["unknown"]}`

**Risultato atteso:**
- planner sceglie `wait`;
- sistema non crasha;
- la conoscenza resta incerta e non diventa azione falsa.

## Test di integrazione

### 6. Loop completo simulato

**Sequenza:**
- osservazione;
- estrazione stato;
- scelta azione;
- esecuzione;
- registrazione memoria.

**Risultato atteso:**
- il ciclo completo esegue senza eccezioni;
- la memoria registra `state`, `action` e `result`.

### 7. Gioco simulato a griglia

**Contesto:** il giocatore deve raggiungere l'obiettivo evitando un ostacolo.

**Risultato atteso:**
- il planner trova un percorso valido;
- l'obiettivo casuale è raggiungibile e distante almeno quattro mosse;
- un seed riproduce la stessa posizione dell'obiettivo;
- l'osservazione omette obiettivo e ostacoli oltre il raggio visivo;
- se l'obiettivo è nascosto, il planner non riceve le sue coordinate e sceglie una frontiera da esplorare;
- la memoria spaziale registra celle osservate e posizioni visitate;
- il gioco viene completato su più seed senza accedere allo stato nascosto;
- il simulatore usa una griglia 30×30 e può restare irrisolto al limite di turni fissato, registrando il limite senza fallire i test di regressione;
- l'obiettivo mantiene la posizione dopo la prima azione direzionale e cambia dopo la seconda;
- l'obiettivo non si sovrappone al giocatore o all'ostacolo;
- l'ostacolo cambia casella dopo ogni azione direzionale senza sovrapporsi al giocatore o all'obiettivo;
- il planner rivaluta il percorso con la nuova posizione dell'ostacolo;
- l'adattatore applica ogni mossa allo stato del gioco simulato;
- il ciclo osserva nuovamente lo stato aggiornato;
- il gioco segnala la vittoria solo quando il giocatore raggiunge l'obiettivo;
- il nemico si attiva quando il giocatore entra nel raggio di 4 caselle e lo insegue;
- il nemico è esposto al planner usando il campo ostacoli esistente, senza un campo dedicato;
- giocatore ed entità partono da 10 PV; ogni attacco infligge 3 PV e la difesa annulla il danno;
- l'osservazione espone i PV senza etichette o istruzioni di combattimento;
- l'osservazione espone due controlli opachi senza la loro corrispondenza semantica;
- l'agente seleziona controlli opachi in base all'incertezza e aggiorna profili medi confrontando osservazioni prima/dopo, senza leggere esiti semanticamente etichettati;
- un'entità a 0 PV diventa un ostacolo statico; il giocatore a 0 PV perde la partita;
- le esperienze del ciclo sono registrate in memoria.

## Test di performance

### 8. Misura del ciclo

**Obiettivo:** individuare tempo di esecuzione del loop.

**Metriche da registrare:**
- tempo di acquisizione;
- tempo di stato;
- tempo di decisione;
- tempo di input adapter;
- tempo totale.

## Criteri di accettazione per fase

### Fase base

Il motore deve:
- avviarsi senza eccezioni;
- usare dati fittizi come input;
- selezionare azioni coerenti;
- registrare esperienze.

### Fase percezione

Il sistema deve:
- catturare frame reali;
- interpretare i dati in modo coerente;
- gestire assenza di frame o dati incompleti.

### Fase apprendimento

Il sistema deve:
- distinguere tra osservazione e conoscenza verificate;
- memorizzare eventi con contesto;
- usare l'esperienza per cambiare la scelta dell'azione in modo misurabile.

### Fase generalizzazione

Il sistema deve:
- applicare conoscenze in contesti diversi;
- riconoscere quando una regola non è più valida;
- stabilire che il miglioramento non sia solo “casuale”.
