-- The two-phase tests need prepared transactions, which a server allows none of by default. Takes
-- effect at the restart the entrypoint makes once initdb is done.
ALTER SYSTEM SET max_prepared_transactions = 10;
