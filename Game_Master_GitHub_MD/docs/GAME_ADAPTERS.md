# Adapter per giochi

L'adapter isola il core dai dettagli del gioco, cattura schermo e input.

## Progressione consigliata
1. Ambiente toy deterministico.
2. Ambiente con API ufficiale.
3. Gioco desktop visuale con cattura/input consentiti.
4. Ambiente specializzato.

Un ambiente toy valida il core ma non dimostra compatibilità con un gioco reale.

## Responsabilità
Inizializzazione/cleanup, capacità, osservazioni con timeout, reset quando supportato, lifecycle, identità/versione e seed opzionale. L'input adapter valida azioni, applica limiti, supporta cancellazione e rilascia input attivi.

## Scoperta controlli
Testare input candidati per durate limitate, confrontare osservazioni prima/dopo, ripetere prove ambigue e registrare evidenze. Un mapping manuale è un prior/fallback, non apprendimento autonomo.

Rispettare regole e termini del gioco. Niente bypass anti-cheat. Prevedere stop d'emergenza, durata massima e budget. Scegliere il primo ambiente per osservabilità, riproducibilità, reset, licenza e costo di setup.
