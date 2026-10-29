# Roadmap

## Fase 0 — Scope e repository
- [x] Repository Python esistente ispezionato e preservato.
- [x] Ambiente toy scelto per provare il ciclo logico.
- [ ] Scegliere e documentare il primo ambiente reale consentito.

## Fase 1 — Vertical slice simulato
- [x] Contratti e ciclo osservazione → decisione → azione → verifica.
- [x] Adapter del gioco simulato e memoria delle esperienze.
- [x] Separare il core generico da planner, world model e memoria spaziale della griglia.
- [ ] Safety gate, lifecycle, stop/cancellazione e cleanup per l'input reale.

## Fase 2 — Baseline e misurazione
- [x] Seed ripetibili, confronto volatile/persistente e metriche.
- [x] Benchmark separato per task statico e dinamico.
- [ ] Rendere standard la replica di più ordini per i test dinamici.

## Fase 3 — Scoperta dei controlli
- [x] Input anonimi, confronto prima/dopo e persistenza delle evidenze.
- [x] Misura dell'accuratezza delle previsioni direzionali.
- [ ] Aggiungere gestione esplicita di confidenza, prove contrarie e cambi di dinamica.

## Fase 4 — World model, memoria e planning
- [x] Mappa parziale, esplorazione per frontiere e obiettivo recente fuori visuale.
- [x] Invalidazione del target ricordato se osservato vuoto e recovery da passo bloccato.
- [x] Ipotesi di mobilità del target ricavata da posizioni diverse osservate.
- [x] Retrieval episodico contestuale nell'adapter, con supporto minimo e uso conservativo sui pareggi del planner.
- [ ] Isolare e replicare il beneficio del retrieval con prove dei controlli identiche e benchmark a 200 turni.
- [ ] Migliorare la ricerca del target mobile e la gestione di frontiere non raggiungibili, senza assumere le regole del simulatore.
- [ ] Dimostrare vantaggio misurabile sul task dinamico contro la baseline.

## Fase 4b — Ricerca di varianti strategiche
- [x] Generatore evolutivo generico su parametri numerici con range espliciti, seed riproducibili e confronto sistematico.
- [x] Prima integrazione nell'adapter simulato: genera varianti dei pesi di esplorazione senza contaminare il core.
- [x] Verifica held-out appaiata e promozione conservativa; nessuna modifica automatica in assenza di miglioramento.
- [ ] Trovare una variante che migliori il successo su un holdout ampio e replicato; il meccanismo da solo non garantisce un miglioramento.
- [ ] Valutare strategie a sequenze/competenze soltanto dopo aver misurato i limiti dell'ottimizzazione parametrica.

## Fase 5 — Memoria persistente
- [x] Persistenza per gioco e benchmark cold-start/volatile contro persistente.
- [x] Beneficio ripetuto nel task statico: stessi 40 seed completati con meno turni.
- [ ] Verificare che non ci sia trasferimento negativo nel task dinamico e ripetere su più ordini.

## Fase 6 — Percezione e ambiente reale
- [ ] Scegliere un ambiente consentito e riproducibile.
- [ ] Integrare acquisizione reale, percezione e input con limiti e stop sicuro.
- [ ] Misurare separatamente latenza e affidabilità dei componenti.

## Fase 7 — Modelli neurali
- [ ] Identificare un collo di bottiglia che i meccanismi attuali non risolvono.
- [ ] Definire dati, split e baseline, poi confrontare un modello sostituibile.
- [ ] Adottare il modello soltanto se il miglioramento misurato giustifica costi e complessità.

## Fase 8 — Trasferimento e robustezza
- [ ] Valutare un secondo ambiente, zero-shot/few-shot e trasferimento negativo.
- [ ] Eseguire stress test e documentare ambienti e limiti supportati.

Non dichiarare una fase completata senza implementazione, test e risultati riproducibili.
