# Roadmap

## Fase 0 — Scope
- [ ] Ispezionare repository e preservare codice utile.
- [ ] Decidere OS, Python, toolchain e test.
- [ ] Scegliere ambiente, task, reset e metrica.
- [ ] Configurare CI e smoke test.
**Uscita:** checkout installabile e ambiente/task documentati.

## Fase 1 — Vertical slice
- [ ] Contratti minimi observation/action/result/experience.
- [ ] Adapter toy e safety gate.
- [ ] Ciclo Central Agent.
- [ ] Stop, timeout e cleanup garantito.
- [ ] Log strutturati e test di errori/cancellazione.
**Uscita:** demo end-to-end ripetibile e stop verificato.

## Fase 2 — Baseline
- [ ] Policy di base e task misurabile.
- [ ] Run ripetibili, seed e configurazione.
- [ ] Metriche di successo, azioni, tempo e fallimenti.
**Uscita:** risultati confrontabili.

## Fase 3 — Scoperta controlli
- [ ] Esperimenti limitati.
- [ ] Confronto prima/dopo.
- [ ] Ipotesi, confidenza e prove contrarie.
- [ ] Confronto con baseline.
**Uscita:** scoperta misurabile di controlli definiti.

## Fase 4 — World model e planning
- [ ] Stato con provenienza.
- [ ] Obiettivi e progresso.
- [ ] Planning breve e recovery.
**Uscita:** goal raggiunto meglio della baseline.

## Fase 5 — Memoria persistente
- [ ] Storage scelto sui dati reali.
- [ ] Persistenza versionata e retrieval.
- [ ] Confronto cold-start/memoria.
**Uscita:** beneficio ripetibile.

## Fase 6 — Modelli neurali
- [ ] Identificare collo di bottiglia misurato.
- [ ] Definire dataset e split.
- [ ] Interfaccia sostituibile e confronto con baseline.
**Uscita:** miglioramento che giustifica complessità e costo.

## Fase 7 — Trasferimento
- [ ] Secondo ambiente.
- [ ] Valutare zero-shot/few-shot/adattamento e trasferimento negativo.
**Uscita:** risultati replicabili e limiti documentati.

## Fase 8 — Robustezza e rilascio
- [ ] Stress test, misure risorse, stop/recovery.
- [ ] Documentare ambienti supportati e limiti.
**Uscita:** matrice di test dichiarata superata.

Non saltare una fase senza motivazione ed evidenze.
