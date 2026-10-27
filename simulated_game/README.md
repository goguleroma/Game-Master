# Gioco simulato a griglia

Ambiente a griglia **30×30** per provare il ciclo osservazione → decisione → azione → nuova osservazione. Il giocatore (`P`) deve raggiungere l'obiettivo (`G`) evitando l'ostacolo mobile (`#`). L'obiettivo cambia casualmente all'avvio ed è sempre raggiungibile, a una distanza minima di quattro mosse; poi si sposta ogni due azioni direzionali di Game-Master. Dopo ogni azione direzionale l'ostacolo cambia casella. Né ostacolo né obiettivo possono sovrapporsi al giocatore; non possono neanche occupare la stessa casella.

GM osserva solo il disco di celle entro raggio 4 dal giocatore. Le celle fuori visuale sono mostrate come `?`; coordinate dell'obiettivo e ostacoli lontani sono omessi dall'osservazione. La policy aggiorna una mappa parziale in memoria e, quando il traguardo non è visibile, sceglie un percorso verso una frontiera di celle non ancora osservate. Se la mappa è stata esplorata, usa le visite memorizzate per pattugliare le zone meno visitate.

È presente un'entità attiva (`E`) con raggio visivo 4: quando rileva il giocatore, lo insegue fino a una casella di distanza. Entrambi partono da 10 PV. In contatto, un attacco toglie 3 PV e la difesa annulla il danno; l'entità sceglie casualmente tra le due, e arretra di una casella dopo aver attaccato. A 0 PV il giocatore perde; se l'entità arriva a 0 PV viene rimossa e lascia un ostacolo statico.

GM vede un ID anonimo, posizioni e valori numerici osservabili, oltre alla disponibilità di due controlli opachi (`control_1`, `control_2`). L'osservazione non rivela la corrispondenza dei controlli e non contiene le parole “attacco” o “difesa” né istruzioni di combattimento. Un agente di apprendimento confronta le variazioni numeriche prima/dopo ogni prova e mantiene un profilo medio degli effetti osservati per controllo e contesto spaziale. Sceglie autonomamente i controlli con meno evidenze, riducendo le prove quando la loro incertezza scende sotto la soglia stabilita. Non riceve una ricompensa di combattimento e non ha una regola che lo indirizzi all'attacco o alla difesa. Il metodo di esplorazione e inferenza è implementato nel codice: GM non inventa un algoritmo di apprendimento da zero e non formula spiegazioni linguistiche.

La memoria distingue inoltre entità `stationary` e `mobile`; può formulare ipotesi comportamentali come `approaches_agent` solo dopo almeno tre variazioni coerenti consecutive. Queste ipotesi sono descrittive, non identificano intenzione o pericolosità.

**La memoria non persiste tra partite.** Il runner crea un nuovo `GameEngine` per ogni esecuzione e il learner dei controlli riparte senza prove precedenti. Le mappe, le esperienze e le associazioni apprese sono conservate solo in memoria durante l'episodio corrente.

Dalla cartella principale del repository:

```powershell
python -m simulated_game
```

Per ripetere una partita specifica (ad esempio lo scenario di stress con seed 4):

```powershell
python -m simulated_game --seed 4
```

Quando l'obiettivo è visibile, il planner usa una ricerca del percorso più breve sulla mappa nota, ricalcolandolo dopo ogni turno. Anche un'azione direzionale bloccata conta come turno. L'adattatore applica le mosse allo stato simulato; il ciclo successivo aggiorna la memoria e osserva le nuove posizioni. La partita termina se GM raggiunge il traguardo o finisce i PV. Per ripetere uno scenario nei test si può passare un seed a `GridGame(seed=42)`. Non vengono usati screenshot, tastiera o mouse reali.
