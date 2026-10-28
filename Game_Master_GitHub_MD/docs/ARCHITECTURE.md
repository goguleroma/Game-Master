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
- **Input Adapter:** invio input, limiti, cancellazione e rilascio garantito.
- **Safety Gate:** rifiuto di azioni non valide o eccessive.
- **Learning/Evaluation:** aggiornamento, baseline e metriche.

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
