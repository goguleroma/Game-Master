# Gioco simulato a griglia

Ambiente a griglia **30×30** per provare il ciclo osservazione → decisione → azione → nuova osservazione. Il giocatore (`P`) deve raggiungere l'obiettivo (`G`) evitando l'ostacolo mobile (`#`). L'obiettivo cambia casualmente all'avvio ed è sempre raggiungibile, a una distanza minima di quattro mosse; poi si sposta ogni due azioni direzionali di Game-Master. Dopo ogni azione direzionale l'ostacolo cambia casella. Né ostacolo né obiettivo possono sovrapporsi al giocatore; non possono neanche occupare la stessa casella.

GM osserva solo il disco di celle entro raggio 4 dal giocatore. Le celle fuori visuale sono mostrate come `?`; coordinate dell'obiettivo e ostacoli lontani sono omessi dall'osservazione. La policy aggiorna una mappa parziale in memoria e, quando il traguardo non è visibile, sceglie un percorso verso una frontiera di celle non ancora osservate. Se la mappa è stata esplorata, usa le visite memorizzate per pattugliare le zone meno visitate.

È presente un'entità attiva (`E`) con raggio visivo 4: quando rileva il giocatore, lo insegue fino a una casella di distanza. Entrambi partono da 10 PV. In contatto, un attacco toglie 3 PV e la difesa annulla il danno; l'entità sceglie casualmente tra le due, e arretra di una casella dopo aver attaccato. A 0 PV il giocatore perde; se l'entità arriva a 0 PV viene rimossa e lascia un ostacolo statico.

GM riceve sei controlli elencati dall'ambiente e non etichettati semanticamente; l'adattatore li esegue senza esporre la corrispondenza dei nomi. Il planner sceglie un passo BFS verso l'obiettivo visibile o una frontiera; il learner confronta la posizione prima/dopo i controlli provati e usa gli spostamenti appresi per selezionare l'input che corrisponde al passo desiderato. I controlli senza spostamento vengono ritestati da posizioni diverse prima di essere trattati come non locomotivi. Le evidenze sono persistite per il gioco simulato. Queste associazioni non dimostrano causalità: un controllo di movimento può essere bloccato localmente e cambiamenti ambientali simultanei possono confondere gli altri effetti. Il planner resta basato su BFS/regole; l'apprendimento calibra il collegamento ai controlli, non la strategia di esplorazione.

La memoria distingue inoltre entità `stationary` e `mobile`; può formulare ipotesi comportamentali come `approaches_agent` solo dopo almeno tre variazioni coerenti consecutive. Queste ipotesi sono descrittive, non identificano intenzione o pericolosità.

**Memoria iniziale:** STM mantiene al massimo 60 stati recenti e li elimina dopo 12 secondi; MTM contiene osservazione e piano correnti; esperienze e mappa restano nell'istanza. Il runner salva in `.gm-memory.sqlite3` le prove dei controlli, separate per `grid_30x30_v1`, e le ricarica nelle successive partite simulate. Il file è escluso da Git. La memoria generale resta vuota: non vengono promosse automaticamente regole specifiche del gioco a conoscenze trasferibili.

Dalla cartella principale del repository:

```powershell
python -m simulated_game
```

Per ripetere una partita specifica (ad esempio lo scenario di stress con seed 4):

```powershell
python -m simulated_game --seed 4
```

Quando l'obiettivo è visibile, il planner usa una ricerca del percorso più breve sulla mappa nota, ricalcolandolo dopo ogni turno. Anche un'azione direzionale bloccata conta come turno. L'adattatore applica le mosse allo stato simulato; il ciclo successivo aggiorna la memoria e osserva le nuove posizioni. La partita termina se GM raggiunge il traguardo o finisce i PV. Per ripetere uno scenario nei test si può passare un seed a `GridGame(seed=42)`. Non vengono usati screenshot, tastiera o mouse reali.
