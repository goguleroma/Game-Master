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
- La memoria episodica dell'adapter registra contesto osservabile, intent e outcome; il retrieval è per gioco, con soglia di similarità/supporto, finestra limitata e tie-break tra scelte di costo uguale. Il hint neurale riusa il planner senza cambiare le 40 feature.
- Suite aggiornata: 64 test superati il 2026-10-29, inclusi retrieval episodico, limiti di memoria, adapter, validazione held-out e persistenza delle strategie.
- Benchmark replicabile: `python -m simulated_game.benchmark --scenario all --episodes 40 --sequences 1`.
- Target statico senza inseguitore, tre ordini appaiati di 40 seed: 40/40 vittorie con memoria volatile e persistente in ogni ordine. Turni medi 78,0 contro 62,67–62,70; accuratezza direzionale osservata 100% (2.741 campioni volatile, 2.497 persistenti).
- Target dinamico con inseguitore, un ordine appaiato di 40 seed: volatile 25 vittorie/0 sconfitte/15 timeout; persistente 22/0/17 timeout/1 stallo esplicito. Il target si è mosso circa 62–64 volte per partita ed è stato visibile circa 5–6 turni. Risultato non conclusivo sul beneficio della memoria persistente.
- Retrieval episodico, scenario statico: dieci ordini degli stessi 10 seed hanno prodotto 60/100 vittorie volatile e 58/100 persistenti; blocchi medi 29 contro 3 e turni medi 71,4 contro 64,64. La persistenza combina casi episodici e prove dei controlli; il dato non dimostra un aumento delle vittorie né isola il contributo dei due tipi di memoria.
- Verifica statica su 40 seed/200 turni, un ordine: 40/40 vittorie volatile e 39/40 persistenti; 118 contro 3 blocchi e 74,45 contro 69,05 turni medi. La modalità persistente combina memoria dei controlli e casi episodici, quindi l'effetto specifico del retrieval resta da isolare.

## Lacune e limiti

- La ricerca di un target intermittente che si sposta frequentemente resta debole; l'agente rileva la mobilità ma non la usa ancora per ottenere più vittorie.
- La policy di esplorazione e BFS sono euristiche, non apprendimento generale di strategie.
- Il risultato dinamico sopra usa un solo ordine di seed; occorrono repliche per valutare la sensibilità all'ordine.
- Acquisizione/cattura reale, percezione visiva/OCR e input reali non sono implementati.
- Il trasferimento tra giochi e la memoria generale non sono dimostrati.
- La prima prova diagnostica del retrieval usa 10 seed e massimo 100 turni; inoltre la memoria persistente include prove dei controlli. Il contributo causale dei casi episodici resta da isolare.

## Prossimo obiettivo

Isolare il contributo del retrieval episodico dalle prove dei controlli persistenti e replicare su più ordini di seed e 200 turni. Valutare reti neurali soltanto se un collo di bottiglia misurato lo giustifica.
