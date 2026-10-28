# Strategia di apprendimento

## Fasi
A. **Baseline strumentata:** registrare osservazioni, azioni, risultati, latenza e fallimenti.
B. **Esperimenti isolati:** definire stato iniziale, input, durata, stop condition e osservazioni prima/dopo; confrontare con baseline/no-op.
C. **Stima effetti:** stimare azione → cambiamento con contesto e incertezza.
D. **Goal-directed behavior:** usare il modello dei controlli per raggiungere obiettivi misurabili.
E. **Memoria:** verificare se il recupero di esperienze migliora la baseline.
F. **Trasferimento:** testare un secondo ambiente, inclusi trasferimento negativo e mancato trasferimento.

## Esplorazione vs sfruttamento
Scegliere esplicitamente se usare conoscenze supportate, esplorare per ridurre incertezza, osservare ancora, recuperare da un errore o fermarsi. Evitare button-mashing casuale come default.

## Algoritmi
Non scegliere un algoritmo prima di avere ambiente e baseline. Valutare bandit, system identification, model-based planning, imitation o reinforcement learning in base a reward, costo campioni, dimensione stato/azioni, riproducibilità e hardware. Non addestrare grandi modelli prima di validare la pipeline.

## Metriche
Successo, progresso, azioni per obiettivo, esperimenti per identificare controlli, accuratezza predittiva, recovery, latenza, costo compute e confronto con baseline.
