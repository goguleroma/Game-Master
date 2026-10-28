# Specifica del prodotto

## Visione
Creare un agente Python capace di adattarsi a videogiochi supportati. La generalizzazione tra giochi è un obiettivo da dimostrare sperimentalmente, non una capacità garantita.

## Ciclo
1. Acquisire osservazione con timestamp e provenienza.
2. Interpretare e registrare incertezza.
3. Aggiornare stima dello stato.
4. Recuperare esperienze pertinenti.
5. Scegliere obiettivo o esperimento.
6. Proporre un'azione limitata.
7. Validare tramite safety gate.
8. Eseguire tramite adapter.
9. Osservare nuovamente.
10. Confrontare effetto previsto e osservato.
11. Registrare esperienza.
12. Aggiornare ipotesi e metriche.

## Requisiti
- Lifecycle avvio/pausa/stop/chiusura.
- Osservazioni valide, parziali, invalide o stale.
- Stato con provenienza e incertezza.
- Proposte con parametri, budget, durata e scadenza.
- Esecuzione cancellabile e verifica separata dall'esito tecnico dell'input.
- Scoperta controlli tramite esperimenti.
- Memoria di esperienze, ipotesi, fatti verificati e skill.
- Valutazione contro baseline.
- Stop sicuro, timeout, log e test.

## Fuori ambito iniziale
Compatibilità universale, training di grandi modelli da zero, trasferimento garantito, cloud di default, bypass anti-cheat e GUI sofisticata prima della validazione del core.

## Livelli di successo
1. Ciclo end-to-end in ambiente controllato.
2. Scoperta controlli migliore della baseline secondo metriche.
3. Planning multi-step.
4. Memoria persistente utile.
5. Trasferimento testato su secondo ambiente.
6. Adattamento a un nuovo gioco supportato con setup manuale limitato.

Un requisito è completo solo con implementazione, test/evaluation, risultati e limiti documentati.
