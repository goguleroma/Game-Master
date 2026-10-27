# Roadmap

## Obiettivo

Definire una roadmap concreta e verificabile per il progetto Game Master.

## Fase 0 - Infrastruttura iniziale

- definire struttura del repo
- impostare standard di sviluppo
- definire documenti base
- impostare dipendenze minime
- creare primo scaffold tecnico

Criterio di completamento:
- il repo contiene documentazione base e architettura minima pronto per lo sviluppo.

## Fase 1 - Loop base operativo

- definire il ciclo osservazione-decisione-azione-verifica
- implementare tipi base
- implementare engine principale
- creare planner minimale
- creare input adapter
- registrare esperienze in memoria

Criterio di completamento:
- il sistema esegue un ciclo completo e registrabile su un esempio controllato.

## Fase 2 - Percezione e stato del gioco

- implementare cattura dello schermo o dei frame
- definire estrazione di informazioni visive
- costruire rappresentazione dello stato interno
- distinguere oggetti, UI e obiettivi

Criterio di completamento:
- lo stato del gioco viene convertito in una struttura leggibile e utilizzabile dal planner.

## Fase 3 - Apprendimento dei controlli

- scoprire input disponibili e il loro effetto
- testare combinazioni e durate
- registrare relazioni input-effetto
- definire confidenza e incertezza

Criterio di completamento:
- il sistema è in grado di associare azioni a risultati osservabili in modo tracciabile.

## Fase 4 - Memoria e conoscenza

- distinguere osservazione, ipotesi, conoscenza verificata
- gestire memoria a breve e media durata
- evitare consolidare errori come regole assolute
- introdurre revisioni delle conoscenze

Criterio di completamento:
- la memoria migliora il comportamento del sistema su più sessioni.

## Fase 5 - Pianificazione e strategia

- definire obiettivi e criteri di successo
- generare azioni candidate e valutarle
- pianificare brevi sequenze di azioni
- valutare trade-off tra esplorazione e approccio diretto

Criterio di completamento:
- l'agente esegue piani a breve termine da cui emerge un comportamento coerente.

## Fase 6 - Generalizzazione

- verificare riuso di conoscenze tra giochi diversi
- separare tecniche generali da dettagli specifici
- testare adattamento a nuovi ambienti

Criterio di completamento:
- una parte delle competenze apprese è riutilizzabile in ambienti non identici.

## Fase 7 - Ottimizzazione e maturazione

- migliorare latenza del ciclo
- ridurre colli di bottiglia
- migliorare logging e diagnosi
- stabilire suite di test e benchmark

Criterio di completamento:
- il sistema è stabile, osservabile e migliorabile nel tempo.
