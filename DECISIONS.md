# Decisions

## Decisione 1 - Struttura iniziale del progetto

**Decisione:** usare una struttura basata su moduli specializzati per percezione, decisione, controllo, memoria e motore centrale.

**Motivazione:** separare responsabilità permette di mantenere il codice leggibile, testabile e facilmente estendibile.

**Alternative considerate:** monolite unico, architettura completamente neurale, sistema esclusivamente basato su regole.

**Conseguenze:** occorre mantenere interfacce chiare tra i moduli e evitare dipendenze circolari.

## Decisione 2 - Focus su MVP operativo

**Decisione:** non iniziare con una architettura ultra complessa; partire da un loop minimale di osservazione, scelta, azione e verifica.

**Motivazione:** un prototipo funzionante consente di verificare il sistema e correggere le scelte prima di estendere la complessità.

**Conseguenze:** le funzioni avanzate saranno integrate solo dopo la stabilizzazione dell'operatività di base.

## Decisione 3 - Priorità alla tracciabilità

**Decisione:** loggare azioni, risultati e decisioni in modo semplice e ordinato.

**Motivazione:** senza registro dei risultati non è possibile misurare il progresso o capire dove si verificano errori.

**Conseguenze:** la memoria iniziale dovrà essere semplice ma sufficientemente esplicita per essere utile.

## Decisione 4 - Uso di Python come base

**Decisione:** mantenere Python come linguaggio principale per la base del progetto.

**Motivazione:** il progetto è stato concepito come prototipo e Python consente di iterare rapidamente su architettura e logica.

**Conseguenze:** eventuali ottimizzazioni o migrazioni future dovranno essere motivate da evidenze concrete di prestazione o necessità tecnica.

## Decisione 5 - Nessuna prematura complessità

**Decisione:** evitare di introdurre moduli di apprendimento avanzato prima che il ciclo base sia verificato.

**Motivazione:** la complessità architetturale senza evidenze funzionali produce codice difficile da mantenere.

**Conseguenze:** i moduli avanzati saranno aggiunti solo quando il sistema base avrà compiuto un ciclo di lavoro davvero verificato.
