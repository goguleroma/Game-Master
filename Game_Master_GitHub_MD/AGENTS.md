# Istruzioni per agenti di coding

## Missione
Costruire Game Master da zero: agente Python che impara a controllare giochi attraverso osservazioni ed esperimenti, pianifica, agisce, verifica gli effetti e migliora nel tempo.

## Prima di lavorare
1. Leggi questo file e i documenti pertinenti in `docs/`.
2. Ispeziona il repository reale e preserva codice utile già esistente.
3. Non dichiarare implementata una funzione solo perché è documentata.
4. Definisci un risultato osservabile e criteri di completamento.
5. Non inventare requisiti mancanti: documenta le assunzioni e preferisci default reversibili.

## Architettura
- Architettura ibrida: orchestratore esplicito più moduli deterministici e modelli neurali quando giustificati.
- Central Agent coordina il ciclo, non contiene ogni algoritmo.
- Controlli appresi tramite osservazioni ed esperimenti controllati.
- Interfacce chiare tra percezione, world model, planning, input, memoria e learning.
- Stop sicuro, timeout, log strutturati e test.
- Nessun invio di dati a servizi esterni senza approvazione.

## Regole
- Prima un piccolo percorso end-to-end funzionante; evitare moduli vuoti.
- Core indipendente da adapter, provider di modelli e storage concreti.
- Aggiungere dipendenze solo con motivazione.
- Distinguere osservazioni, inferenze, ipotesi e conoscenze verificate.
- Ogni azione deve essere limitata nel tempo, cancellabile e sicura.
- Rispettare termini e regole dei giochi; niente bypass anti-cheat.
- Non dichiarare test superati se non eseguiti.
- Aggiornare documentazione, roadmap e stato sulla base di prove.

Per attività importanti riportare obiettivo, file modificati, assunzioni, test reali, limiti e prossimo passo.
