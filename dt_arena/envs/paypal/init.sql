-- The pinned PayPal API image reads and writes invoices but does not create
-- this table. Keep the schema at the environment boundary so a fresh task
-- project is usable before any victim or injection tool call.
CREATE TABLE IF NOT EXISTS invoices (
  id TEXT PRIMARY KEY,
  recipient_email TEXT NOT NULL,
  status TEXT NOT NULL,
  items JSONB NOT NULL DEFAULT '[]'::jsonb
);
