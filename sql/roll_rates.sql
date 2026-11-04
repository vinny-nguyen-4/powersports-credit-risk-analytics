-- Dollar-weighted roll-rate (transition) matrix over the most recent N months.
-- Flow from bucket i to bucket j = next-month balance (or charge-off amount) of loans
-- that moved i -> j, divided by the starting balance in bucket i. Rows sum to < 1
-- because scheduled principal payments and payoffs run the balance down.
-- Parameters: $as_of (DATE), $lookback (INTEGER months).
WITH transitions AS (
    SELECT
        p.month,
        p.status                    AS from_status,
        p.balance                   AS from_balance,
        LEAD(p.status)        OVER w AS to_status,
        LEAD(p.balance)       OVER w AS to_balance,
        LEAD(p.chargeoff_amt) OVER w AS to_chargeoff
    FROM performance p
    WHERE p.month <= CAST($as_of AS DATE)
    WINDOW w AS (PARTITION BY p.loan_id ORDER BY p.month)
),
window_rows AS (
    SELECT *
    FROM transitions
    WHERE from_status IN (0, 1, 2, 3)
      AND to_status IS NOT NULL
      AND month >  CAST($as_of AS DATE) - to_months(CAST($lookback AS INTEGER) + 1)
),
starting AS (
    SELECT from_status, SUM(from_balance) AS start_balance
    FROM window_rows
    GROUP BY from_status
)
SELECT
    w.from_status,
    w.to_status,
    COUNT(*)                                                               AS accounts,
    SUM(CASE WHEN w.to_status = 4 THEN w.to_chargeoff ELSE w.to_balance END) AS flow_amount,
    s.start_balance,
    SUM(CASE WHEN w.to_status = 4 THEN w.to_chargeoff ELSE w.to_balance END)
        / s.start_balance                                                  AS roll_rate
FROM window_rows w
JOIN starting s USING (from_status)
GROUP BY w.from_status, w.to_status, s.start_balance
ORDER BY w.from_status, w.to_status;
