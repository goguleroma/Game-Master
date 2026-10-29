# Architettura

## Schema logico
Game/Environment ↔ Game Adapter → Observation → Perception → World Model ↔ Memory → Central Agent → Planner/Experiment Manager → Safety Gate → Input Adapter → Action Result → Outcome Verification → Experience/Learning.

Implementare soltanto i componenti necessari al primo percorso end-to-end.

## Responsabilità
- **Central Agent:** lifecycle, coordinamento, timeout, cancellazione, correlazione, error handling.
- **Game Adapter:** osservazioni e metadati.
- **Perception:** feature/entità con provenienza e confidenza.
- **World Model:** stato stimato, cambiamenti, incertezza e dati stale.
- **Memory:** esperienze, ipotesi, conoscenze e skill.
- **Planner:** obiettivi e azioni candidate.
- **Experiment Manager:** esperimenti limitati e valutazione delle evidenze.
- **Strategy Search:** propone varianti di parametri tramite selezione, crossover e mutazione; riceve evaluator e limiti dall'adapter dell'ambiente.
- **Input Adapter:** invio input, limiti, cancellazione e rilascio garantito.
- **Safety Gate:** rifiuto di azioni non valide o eccessive.
- **Learning/Evaluation:** aggiornamento, baseline e metriche.

## Confine implementato

`core.GameEngine` riceve policy ed esecutore di azioni tramite dependency injection; può inoltre ricevere un adapter di osservazione. Il core registra esperienze con memoria generica e non contiene planner, coordinate, controlli o regole del simulatore. L'implementazione a griglia, inclusi world model e memoria spaziale, è confinata in `simulated_game/agent/` e `simulated_game/`, composta attraverso `simulated_game.runtime`. Il test simula un ambiente concreto ma non definisce l'architettura o la policy di GM.

L'adapter a griglia aggiunge una memoria episodica per gioco: contesto osservabile, direzione pianificata e risultato con reward limitato. Il retrieval usa una finestra recente e supporto ripetuto; il planner lo applica solo tra scelte di costo equivalente. Questo non amplia il contratto del core né costituisce memoria trasferibile tra ambienti.

## Struttura suggerita
```text
src/game_master/
  app.py
  core/
  contracts/
  adapters/
  perception/
  world/
  decision/
  experiments/
  input/
  memory/
  learning/
  evaluation/
tests/
docs/
```
È una proposta, non un obbligo di creare moduli vuoti.

## Ciclo
Controlla stop e budget; acquisisci e valida osservazione; aggiorna stato; recupera memoria; scegli exploit/explore/pausa; genera e valida azione; esegui entro limiti; rilascia input; osserva il risultato; valuta; registra esperienza; emetti metriche.

Il dominio non deve importare adapter concreti o librerie di modelli. Usare timeout, cancellazione, retry limitati e cleanup anche dopo errori.
