-- The refund cap names itself in the error it raises.
--
-- 003's trigger raised a bare check_violation, which the executor could not tell
-- apart from any other CHECK on refunds -- amount_paise > 0 among them. It
-- reported all of them to the planner as an ordinary refusal, so a bug upstream
-- would have read as a policy answer. With CONSTRAINT set, only the cap is a
-- refusal and every other violation surfaces as the bug it is.
--
-- A new file rather than an edit to 003, because the runner applies each file
-- once: a database that already has 003 would keep the old function.

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
            USING ERRCODE = 'check_violation', CONSTRAINT = 'refunds_within_charges';
    END IF;
    RETURN NEW;
END;
$$;
