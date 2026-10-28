# Contratti dati

I campi seguenti sono concettuali; implementare tipi e validazione dopo aver deciso lo stack. Usare timestamp timezone-aware e schemi versionati.

## Observation
`schema_version`, `observation_id`, `session_id`, `environment_id`, `timestamp`, `source`, `payload_type`, payload/riferimento, validità (`valid`, `partial`, `invalid`, `stale`) e metadata.

## WorldState
ID/versione, sessione, timestamp, entità/feature, variabili, ipotesi di obiettivo, provenienza (`observed`, `inferred`, `unknown`), confidenza e indicatori stale.

## ActionProposal
ID, sessione, timestamp, tipo e parametri, obiettivo/effetto atteso, durata/budget, confidenza, origine, scadenza e trace ID. Una proposta non prova l'esecuzione.

## ActionResult
ID azione/trace, tempi, stato (`executed`, `rejected`, `cancelled`, `timed_out`, `failed`, `partial`), input inviati, errore e conferma rilascio input. Esecuzione tecnica non equivale a successo nel gioco.

## ExperienceRecord
Osservazione/stato prima, proposta, risultato, osservazione dopo, delta, effetto previsto/osservato, confondenti, reward/progresso, livello di attribuzione causale e versioni.

## KnowledgeItem
ID/versione, tipo (`hypothesis`, `verified_fact`, `control_model`, `skill`), ambito, contenuto, prove a supporto e contrarie, confidenza, regola di verifica e condizioni di rivalidazione.

## Evidenza
Distinguere osservato, correlato, ipotesi causale, supportato da ripetizioni e verificato nei contesti dichiarati. Testare dati invalidi, azioni non supportate, timestamp, serializzazione e migrazioni.
