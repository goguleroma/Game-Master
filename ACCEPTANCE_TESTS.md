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

## Test di perforrformance

### 7. Misura del ciclo

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
