# Stato progetto

## Fase corrente

Core di orchestrazione indipendente dall'ambiente, verificato insieme a un adapter di test a griglia. Non è ancora collegato a un gioco reale.

## Confine architetturale verificato

- `core.GameEngine` richiede policy ed esecutore di azioni; l'adapter di osservazione è iniettabile.
- Il core conserva solo memoria generica di osservazioni ed esperienze; non conosce coordinate, controlli, planner o regole della griglia.
- World model, planner, policy, learner e memoria spaziale della griglia sono in `simulated_game/agent/`; le regole del mondo sono in `simulated_game/game.py`.
- L'adapter di test compone questi componenti con il core tramite `simulated_game/runtime.py`; non è una strategia integrata nel core.
- `core.strategy_search` può generare, combinare e mutare varianti di parametri numerici entro limiti dichiarati, con valutazione fornita dall'ambiente.
- `simulated_game.strategy_search` varia tre parametri della policy di esplorazione e confronta le varianti con una baseline su seed condivisi; usa seed di holdout separati e promuove solo miglioramenti conservativi.
- Il generatore è verificato; il primo candidato ha migliorato il tasso aggregato nel piccolo holdout, ma con esiti regressivi appaiati. Non è dimostrato un miglioramento stabile; la strategia attiva non cambia senza conferma.
- Primo confronto dinamico completo: 14 varianti esplorate su 5 seed; candidato migliore 10/20 vittorie su holdout contro 8/20 baseline, ma con 6 regressioni appaiate. Nessuna promozione automatica; occorrono repliche più ampie.

## Evidenze verificate

- Ambiente simulato a griglia 30×30 con osservazione parziale, sei controlli anonimi, ostacolo mobile, target mobile e inseguitore.
- Il planner usa BFS/frontiere; il learner inferisce gli spostamenti dei controlli dalle osservazioni prima/dopo, senza ricevere la mappatura semantica.
- La memoria riutilizza la posizione più recente del target fuori visuale e la invalida se la cella viene osservata vuota o se il target è ri-osservato altrove.
- Posizioni del target diverse osservate nel tempo formano un'ipotesi di target dinamico; il planner evita di inseguire la posizione ricordata quando l'ipotesi è attiva. Nel benchmark è stata inferita in 35/40 episodi volatili e 29/40 persistenti, senza migliorare i tassi di vittoria.
- Dopo un esito di movimento bloccato, il planner evita quella transizione per due decisioni per tentare un percorso alternativo.
- Suite aggiornata: 49 test superati il 2026-10-28, inclusi generatore di varianti, adapter, validazione held-out e persistenza delle strategie.
- Benchmark replicabile: `python -m simulated_game.benchmark --scenario all --episodes 40 --sequences 1`.
- Target statico senza inseguitore, tre ordini appaiati di 40 seed: 40/40 vittorie con memoria volatile e persistente in ogni ordine. Turni medi 78,0 contro 62,67–62,70; accuratezza direzionale osservata 100% (2.741 campioni volatile, 2.497 persistenti).
- Target dinamico con inseguitore, un ordine appaiato di 40 seed: volatile 25 vittorie/0 sconfitte/15 timeout; persistente 22/0/17 timeout/1 stallo esplicito. Il target si è mosso circa 62–64 volte per partita ed è stato visibile circa 5–6 turni. Risultato non conclusivo sul beneficio della memoria persistente.

## Lacune e limiti

- La ricerca di un target intermittente che si sposta frequentemente resta debole; l'agente rileva la mobilità ma non la usa ancora per ottenere più vittorie.
- La policy di esplorazione e BFS sono euristiche, non apprendimento generale di strategie.
- Il risultato dinamico sopra usa un solo ordine di seed; occorrono repliche per valutare la sensibilità all'ordine.
- Acquisizione/cattura reale, percezione visiva/OCR e input reali non sono implementati.
- Il trasferimento tra giochi e la memoria generale non sono dimostrati.

## Prossimo obiettivo

Migliorare e misurare la ricerca del target dinamico, replicando i confronti su più ordini senza regressioni nel caso statico. Valutare reti neurali soltanto se un collo di bottiglia misurato lo giustifica.
