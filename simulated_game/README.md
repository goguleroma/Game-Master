# Gioco simulato a griglia

Questo pacchetto è un adapter di test isolato, non una parte delle strategie del core. Le regole della griglia, il world model, la policy, il learner e la memoria spaziale qui descritti non devono diventare assunzioni di `core.GameEngine`; un altro ambiente dovrà fornire propri adapter e policy.

Ambiente a griglia **30×30** per provare il ciclo osservazione → decisione → azione → nuova osservazione. Il giocatore (`P`) deve raggiungere l'obiettivo (`G`) evitando l'ostacolo mobile (`#`). L'obiettivo cambia casualmente all'avvio ed è sempre raggiungibile, a una distanza minima di quattro mosse; poi si sposta ogni due azioni direzionali di Game-Master. Dopo ogni azione direzionale l'ostacolo cambia casella. Né ostacolo né obiettivo possono sovrapporsi al giocatore; non possono neanche occupare la stessa casella.

GM osserva solo il disco di celle entro raggio 4 dal giocatore. Le celle fuori visuale sono mostrate come `?`; coordinate dell'obiettivo e ostacoli lontani sono omessi dall'osservazione. La policy aggiorna una mappa parziale in memoria e, quando il traguardo non è visibile, sceglie un percorso verso una frontiera di celle non ancora osservate. Se la mappa è stata esplorata, usa le visite memorizzate per pattugliare le zone meno visitate.

È presente un'entità attiva (`E`) con raggio visivo 4: quando rileva il giocatore, lo insegue fino a una casella di distanza. Entrambi partono da 10 PV. In contatto, un attacco toglie 5 PV e la difesa annulla il danno; l'entità sceglie casualmente tra le due, e arretra di una casella dopo aver attaccato. A 0 PV il giocatore perde; se l'entità arriva a 0 PV viene rimossa e lascia un ostacolo statico. Con 10 PV a parte e un danno di 5, il combattimento è ora tanto importante quanto il traguardo: la gestione di vita, danno e difesa diventa centrale nel loop.

GM riceve sei controlli elencati dall'ambiente e non etichettati semanticamente; l'adattatore li esegue senza esporre la corrispondenza dei nomi. Il planner sceglie un passo BFS verso l'obiettivo visibile o una frontiera; il learner confronta la posizione prima/dopo i controlli provati e usa gli spostamenti appresi per selezionare l'input che corrisponde al passo desiderato. La memoria conserva l'ultima posizione nota del target e il planner la usa mentre è fuori visuale; la posizione viene invalidata se la cella viene osservata vuota o se viene osservato il target in un altro punto. Dopo un'azione bloccata il planner evita temporaneamente quella transizione e cerca un percorso alternativo. Le evidenze dei controlli sono persistite per il gioco simulato. Le associazioni non dimostrano causalità: un controllo di movimento può essere bloccato localmente e cambiamenti ambientali simultanei possono confondere gli altri effetti. Il planner resta basato su BFS/regole; l'apprendimento calibra il collegamento ai controlli, non la strategia di esplorazione.

La memoria episodica registra, per ogni azione direzionale osservabile, il contesto a grana grossa (target, ostacoli adiacenti, distanza da entità, salute e frontiere), la direzione pianificata e il reward derivato dall'esito osservato. I casi sono separati per `game_id`, il contesto ha schema versionato, il recupero esamina fino a 256 casi recenti, richiede similarità minima 0,65 e almeno due esempi per direzione. La cache in memoria contiene al massimo 2.000 casi; SQLite conserva una finestra recente, potata a batch. Il planner può usarli solo per scegliere tra percorsi BFS di uguale lunghezza o frontiere con punteggio uguale; il comportamento senza supporto resta invariato. La policy neurale usa lo stesso suggerimento del planner e mantiene l'encoder a 40 feature. Queste decisioni non sono conoscenze generali e non sono trasferite ad altri giochi.

Per eseguire il benchmark appaiato tra memoria volatile e persistente, con metriche di successo, timeout, blocchi, stalli, copertura, progresso, accuratezza delle previsioni e retrieval episodico:

```powershell
python -m simulated_game.benchmark --scenario all --episodes 40 --sequences 1
```

`static` fissa il target e disabilita l'inseguitore per isolare navigazione e memoria; `dynamic` mantiene target mobile e inseguitore. `--sequences` ripete il confronto sugli stessi seed (`0`–`episodes-1`) in ordini diversi: serve a misurare la sensibilità all'ordine, non aggiunge seed indipendenti. Il benchmark usa un database temporaneo per ogni sequenza e non modifica `.gm-memory.sqlite3`.

Un primo run diagnostico da 10 seed e massimo 100 turni ha prodotto, in un ordine, 6/10 vittorie volatile contro 4/10 persistenti nello scenario statico e 2/10 contro 6/10 nel dinamico. Ripetendo il solo scenario statico in 10 ordini degli stessi seed si sono osservate 60/100 vittorie volatile e 58/100 persistenti, con forte riduzione dei blocchi e dei turni medi ma senza vantaggio nel numero di vittorie. La persistenza include anche le prove dei controlli: i risultati non isolano l'effetto dei casi episodici. Vedere gli Esperimenti 30–31 nel registro principale.

Una verifica statica successiva su 40 seed e massimo 200 turni (un ordine) ha dato 40/40 vittorie volatile e 39/40 persistenti, con un solo esito appaiato diverso. La persistenza ha ridotto le azioni bloccate da 118 a 3, quelle senza movimento da 179 a 6 e i turni medi da 74,45 a 69,05. È un segnale di efficienza nel task, non prova che il retrieval episodico causi il miglioramento: la persistenza include anche l'apprendimento dei controlli. Vedere l'Esperimento 32 nel registro principale.

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

Per far generare e confrontare varianti dei pesi della strategia di esplorazione:

```powershell
python -m simulated_game.strategy_search --scenario dynamic
```

Il generatore muta e combina tre parametri: penalità per le celle già visitate, bonus per l'informazione nuova e soglia di osservazioni di mobilità prima di smettere di inseguire un target ricordato. Confronta i candidati sugli stessi seed, poi verifica il migliore su un holdout separato (40 episodi per default). Salva `.gm-strategy-dynamic.json` e la partita successiva lo carica **solo** se il candidato ottiene almeno due vittorie appaiate in più senza regressioni, oppure mantiene gli stessi esiti e riduce di almeno il 5% i turni medi di vittoria. Il file è locale ed escluso da Git; in caso contrario la strategia resta invariata. Per provare separatamente il task statico usare `--scenario static`; la configurazione statica non sostituisce quella usata dalla partita dinamica.

Questa è ricerca evolutiva di parametri, non una rete neurale e non generazione libera di piani arbitrari: la grammatica delle varianti è limitata ai parametri configurati dall'adapter. Un risultato positivo nel simulatore non dimostra trasferimento ad altri giochi.

## Esperimento neurale isolato

La policy DQN è opzionale e non sostituisce la baseline. La codifica include un suggerimento del planner (una direzione BFS verso il target o una frontiera), e il reward misura anche se lo spostamento osservato segue tale intent quando il target è nascosto o mobile. Queste feature sono locali all'adapter della griglia e cambiano la dimensione di input: i precedenti checkpoint non sono compatibili. Su tre holdout, il checkpoint per imitazione ha avuto piccoli vantaggi aggregati nei primi due e un pareggio nelle vittorie sul terzo, ma regressioni appaiate in tutti; il fine-tuning DQN ha degradato la validation, quindi nessuna promozione è stata fatta. Installare le dipendenze aggiuntive, addestrare un candidato e confrontarlo con la baseline:

```powershell
python -m pip install -r requirements-neural.txt
python -m simulated_game.neural_experiment --train-episodes 300 --holdout-episodes 40
```

Il training riprende automaticamente `.gm-neural-training.pt` quando esiste; `--train-episodes` indica quanti episodi aggiungere in questa esecuzione. Salva rete, optimizer, replay buffer, stato RNG e contatori. Ogni 50 episodi valuta la rete contro la baseline su un insieme fisso di validation seed. Ogni invocazione finale valuta su seed disgiunti da training e validation, e non riutilizza un holdout già consumato in una ripresa successiva. Il miglior checkpoint secondo la validation è `.gm-neural-training.best.pt`; la baseline non viene sostituita automaticamente. Usare un percorso diverso con `--model` e `--fresh` per iniziare un training indipendente.

Il report include vittorie, sconfitte, timeout, stalli e regressioni appaiate. Per provare il candidato in una partita simulata:

```powershell
python -m simulated_game --policy neural --model .gm-neural-training.best.pt
```

Se il training da zero è debole, si può confrontare un warm-start che imita prima la policy baseline e poi affina con DQN, usando un checkpoint distinto:

```powershell
python -m simulated_game.neural_experiment --model .gm-neural-imitation.pt --fresh --initialization baseline --train-episodes 300 --imitation-episodes 100
```

Questo esperimento apprende una policy per l'ambiente simulato; curiosità intrinseca e vittoria/sconfitta non equivalgono a obiettivi semantici autonomi o a generalizzazione ad altri giochi.

Quando l'obiettivo è visibile, il planner usa una ricerca del percorso più breve sulla mappa nota, ricalcolandolo dopo ogni turno. Anche un'azione direzionale bloccata conta come turno. L'adattatore applica le mosse allo stato simulato; il ciclo successivo aggiorna la memoria e osserva le nuove posizioni. La partita termina se GM raggiunge il traguardo o finisce i PV. Per ripetere uno scenario nei test si può passare un seed a `GridGame(seed=42)`. Non vengono usati screenshot, tastiera o mouse reali.
