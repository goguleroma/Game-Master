# Game Master

Agente videoludico Python con architettura ibrida: orchestratore centrale, moduli deterministici e modelli neurali specializzati quando giustificati. L'agente osserva, interpreta, decide, agisce, verifica l'esito e apprende dall'esperienza.

## Decisioni confermate
- Central Agent come coordinatore del ciclo.
- Scoperta autonoma dei controlli con esperimenti limitati.
- Memoria distinta in esperienze, ipotesi, conoscenze verificate e skill.
- Sviluppo incrementale con test e metriche.
- Dati locali per default; servizi esterni solo con consenso esplicito.

Leggere `AGENTS.md`, poi `docs/PRODUCT_SPEC.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md` e `docs/OPEN_QUESTIONS.md`.

Il repository va ispezionato prima di modificare codice: questi documenti descrivono obiettivi, non funzionalità già implementate. Primo obiettivo: ciclo end-to-end in un ambiente controllato.
