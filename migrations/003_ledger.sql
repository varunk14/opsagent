-- The ledger the tools act on, and the rule that it never pays back more than
-- was taken.
--
-- charges are what was actually taken for an order; Priya's order 4821 has two.
-- refunds are money paid back, one row per honoured tool call. Its
-- idempotency_key is UNIQUE and references tool_calls, so a refund cannot exist
-- without the keyed call that issued it, and one call cannot issue two.
--
-- The cap -- refunds on an order never exceed its charges -- spans rows, which a
-- CHECK cannot express. A trigger does, and it locks the order row first: two
-- refunds racing on one order then queue behind each other, instead of both
-- reading the old total and both fitting under it. Being in the database, the
-- cap also stops code that writes refunds without going through the executor.

CREATE TABLE IF NOT EXISTS charges (
    id           bigserial PRIMARY KEY,
    order_id     text NOT NULL REFERENCES orders (id),
    amount_paise bigint NOT NULL CHECK (amount_paise > 0),
    charged_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS charges_order_id_idx ON charges (order_id);

CREATE TABLE IF NOT EXISTS refunds (
    id              bigserial PRIMARY KEY,
    order_id        text NOT NULL REFERENCES orders (id),
    amount_paise    bigint NOT NULL CHECK (amount_paise > 0),
    reason          text NOT NULL,
    run_id          uuid NOT NULL REFERENCES runs (id),
    idempotency_key text NOT NULL UNIQUE REFERENCES tool_calls (idempotency_key),
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS refunds_order_id_idx ON refunds (order_id);

CREATE OR REPLACE FUNCTION refunds_within_charges() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    charged  bigint;
    refunded bigint;
BEGIN
    -- Serialise refunds per order. Under READ COMMITTED each statement below
    -- takes a fresh snapshot, so once the lock is ours the sums include any
    -- refund that was committed while we waited for it.
    PERFORM 1 FROM orders WHERE id = NEW.order_id FOR UPDATE;

    SELECT coalesce(sum(amount_paise), 0) INTO charged FROM charges WHERE order_id = NEW.order_id;
    SELECT coalesce(sum(amount_paise), 0) INTO refunded FROM refunds WHERE order_id = NEW.order_id;

    IF refunded + NEW.amount_paise > charged THEN
        RAISE EXCEPTION
            'refund of % paise on order % would exceed the % paise charged (% already refunded)',
            NEW.amount_paise, NEW.order_id, charged, refunded
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER refunds_within_charges
    BEFORE INSERT ON refunds
    FOR EACH ROW EXECUTE FUNCTION refunds_within_charges();
