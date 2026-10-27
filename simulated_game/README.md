# Gioco simulato a griglia

Ambiente a griglia per provare il ciclo osservazione → decisione → azione → nuova osservazione. Il giocatore (`P`) deve raggiungere l'obiettivo (`G`) evitando l'ostacolo mobile (`#`). L'obiettivo cambia casualmente all'avvio ed è sempre raggiungibile, a una distanza minima di quattro mosse; poi si sposta ogni due azioni direzionali di Game-Master. Dopo ogni azione direzionale l'ostacolo cambia casella. Né ostacolo né obiettivo possono sovrapporsi al giocatore; non possono neanche occupare la stessa casella.

GM osserva solo il disco di celle entro raggio 4 dal giocatore. Le celle fuori visuale sono mostrate come `?`; coordinate dell'obiettivo e ostacoli lontani sono omessi dall'osservazione. Al momento il planner non esplora attivamente: quando l'obiettivo non è visibile sceglie `wait`, e la demo segnala questo limite.

Dalla cartella principale del repository:

```powershell
python -m simulated_game
```

Quando l'obiettivo è visibile, il planner usa una ricerca del percorso più breve sulla posizione corrente di giocatore, obiettivo e ostacolo, ricalcolandolo dopo ogni turno. Anche un'azione direzionale bloccata conta come turno. L'adattatore applica le mosse allo stato simulato; il ciclo successivo osserva le nuove posizioni. Per ripetere uno scenario nei test si può passare un seed a `GridGame(seed=42)`. Non vengono usati screenshot, tastiera o mouse reali.
