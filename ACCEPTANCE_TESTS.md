# ACCEPTANCE_TESTS.md

## Obiettivo

Definire i test minimamente necessari per verificare che il sistema stia effettivamente migliorando e non soltanto “girando”.

## Test del core indipendente dall'ambiente

### 1. Dipendenze esplicite

**Contesto:** creare `GameEngine` senza policy o esecutore.

**Risultato atteso:** la costruzione richiede componenti espliciti; il core non sceglie una policy o un adapter di test come fallback.

### 2. Decisione delegata

**Contesto:** un adapter fornisce uno stato normalizzato e una policy produce un'azione con nome e parametri arbitrari.

**Risultato atteso:** il core passa lo stato alla policy ed esegue l'azione attraverso l'esecutore iniettato, senza interpretare nomi di azioni o regole del gioco.

### 3. Risultato ed esperienza

**Contesto:** l'esecutore restituisce un risultato.

**Risultato atteso:** il core passa il risultato alla policy e registra stato, azione e risultato in memoria.

### 4. Policy senza azione

**Contesto:** la policy dichiara che non è opportuno agire.

**Risultato atteso:** il core non chiama l'esecutore e non registra un'azione inventata.

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

Le prove delle regole della griglia sono confinate ai test dell'adapter in `tests/test_simulated_game.py`; i test del core non devono codificare obiettivi o controlli di alcun gioco.

### 7.1 Memoria episodica e retrieval

**Contesto:** registrare casi decisionali dall'osservazione e dall'esito di azioni dell'adapter simulato.

**Risultato atteso:**
- vengono salvati contesti conformi allo schema e reward finiti nell'intervallo `[-1, 1]`;
- retrieval richiede almeno due esempi sufficientemente simili e non riusa casi di un altro `game_id`;
- i casi persistenti vengono ricaricati, mentre cache e finestra di retrieval restano limitate;
- la memoria influenza il planner soltanto tra alternative di costo uguale e il caso scelto è contabilizzato come decisione guidata;
- il benchmark riporta query, query con match, riferimenti a casi recuperati, decisioni guidate e media dei casi disponibili;
- l'hint della policy neurale usa il retrieval senza cambiare la dimensione dell'encoder.

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
- l'osservazione espone i sei controlli opachi senza la loro corrispondenza semantica;
- il planner indica un passo verso obiettivo/frontiera e il learner, dopo aver osservato gli effetti, seleziona il controllo anonimo che realizza lo spostamento desiderato;
- un target visto e poi uscito dalla visuale resta recuperabile dalla memoria; una cella target osservata vuota invalida l'ipotesi;
- dopo una transizione di movimento segnalata come bloccata, il planner evita temporaneamente quel passo e trova un percorso alternativo se esiste;
- il benchmark può isolare target statico e target dinamico e riporta esiti appaiati, turni, stalli, blocchi, copertura, progresso e accuratezza delle predizioni;
- l'agente aggiorna il valore atteso dei controlli usando progresso verso l'obiettivo, variazioni del proprio stato e risultato terminale, mantenendo i controlli anonimi;
- la selezione esplora ancora alternative incerte, ma interrompe le prove in un contesto quando tutti i controlli hanno prodotto solo osservazioni invariate;
- gli esiti appresi sono persistiti per gioco e possono influire sulle scelte delle partite successive;
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

### 9. Ricerca e promozione di varianti

**Contesto:** generare varianti con parametri limitati e valutare baseline e candidati.

**Risultato atteso:**
- la ricerca è riproducibile con seed fissati e non produce parametri fuori dai limiti;
- tutti i candidati di training sono confrontati sugli stessi seed;
- la selezione usa reward e evaluator forniti dall'ambiente, non regole di griglia nel core;
- la promozione usa seed di holdout separati e gli stessi seed per baseline e candidato;
- una variante con esiti regressivi resta non promossa e non cambia la strategia caricata dai run successivi.

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
