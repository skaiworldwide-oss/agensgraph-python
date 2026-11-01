-- Runs once at initdb as the image's superuser (postgres up to 2.17, agens on 2.18).
-- Every database created afterwards, the suite's included, inherits vector.
CREATE EXTENSION IF NOT EXISTS vector;
\c template1
CREATE EXTENSION IF NOT EXISTS vector;
