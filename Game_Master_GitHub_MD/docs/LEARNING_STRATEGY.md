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

## Ricerca di varianti strategiche

- La ricerca è un componente generico che opera su parametri numerici espliciti, limitati e serializzabili; non importa un gioco né inventa controlli che l'ambiente non espone.
- Un adapter definisce quali parametri cambiano la propria policy e un evaluator restituisce un punteggio per seed. Il generatore crea varianti casuali, seleziona le migliori, combina i parametri e li muta.
- Ogni candidato e la baseline sono valutati sugli stessi seed di training. Le varianti sono ordinate per punteggio medio, mentre la promozione usa seed di holdout mai impiegati nella ricerca.
- Non sovrascrivere una strategia soltanto perché vince sui seed di ricerca: richiedere vantaggio su almeno due seed di holdout appaiati senza regressioni di vittoria, oppure prestazioni terminali identiche e almeno il 5% di riduzione dei turni medi per vittoria. Se il criterio non è soddisfatto, lasciare invariata la strategia salvata.
- La prima integrazione esplora tre parametri della policy a griglia: penalità di rivisitazione, bonus d'informazione e soglia intera di evidenze di mobilità prima di abbandonare un target ricordato. È un esperimento di ottimizzazione locale e interpretabile, non una prova di strategie generaliste. Non esegue ricerca online durante input reali.
- La policy DQN ibrida-curiosità è un candidato opzionale, separato dalla baseline e circoscritto all'adapter della griglia. Il checkpoint di training conserva pesi, optimizer, replay buffer e stato RNG così gli episodi possono proseguire fra invocazioni. Validation fissa e holdout finale sono disgiunti dai seed di training; si conserva il miglior checkpoint di validation, senza promuovere automaticamente la baseline. Il training da zero ha fatto 6/40 vittorie sul holdout; il warm-start con imitazione baseline e DQN ne ha fatte 11/40, contro 24/40 della baseline e con 17 regressioni appaiate. Il warm-start migliora lo scratch ma resta insufficiente: prima di scalare il numero di episodi occorre diagnosticare rappresentazione, pianificazione e reward.
- Primo intervento isolato sulla rappresentazione: aggiunte feature one-hot per la direzione suggerita dal planner BFS e un flag per distinguere l'esplorazione di frontiera. Valutare dapprima la fedeltà all'imitazione, poi il fine-tuning, mantenendo invariati il reward e i seed split indipendenti; i checkpoint di input dimension precedente sono incompatibili.
- Secondo intervento isolato: reward di progresso che confronta lo spostamento osservato con l'intent BFS registrato prima dell'azione, anche quando il target è nascosto o si muove. Il best checkpoint di sola imitazione ha fatto 23/40 e 27/40 vittorie sui primi due holdout, contro 21/40 e 26/40 baseline; entrambi hanno avuto regressioni appaiate (12/10 e 8/7). Sul terzo holdout, sia il checkpoint intent sia quello progress hanno pareggiato la baseline a 24/40, con 6 miglioramenti e 6 regressioni appaiate; il tempo medio per vittoria del candidato è stato inferiore, ma lo score aggregato è pressoché invariato. Il vantaggio non è ripetibile abbastanza per promuovere il modello. 300 episodi DQN hanno ridotto la validation da 11/20 a 0/20, perciò il checkpoint BC è rimasto selezionato. Prossimo: verificare un fine-tuning che preservi il comportamento imitato, senza riutilizzare i seed held-out.

## Metriche
Successo, progresso, azioni per obiettivo, esperimenti per identificare controlli, accuratezza predittiva, recovery, latenza, costo compute e confronto con baseline.
