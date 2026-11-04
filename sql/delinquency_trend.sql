-- Monthly portfolio trend: outstanding balance, delinquency, and annualized net charge-off rate.
-- Status codes: 0 current, 1 30dpd, 2 60dpd, 3 90+dpd, 4 charged off, 5 paid off.
WITH monthly AS (
    SELECT
        p.month,
        SUM(p.balance)                                                   AS ending_balance,
        SUM(CASE WHEN p.status IN (1, 2, 3) THEN p.balance ELSE 0 END)   AS dq30_plus_balance,
        SUM(CASE WHEN p.status IN (2, 3)    THEN p.balance ELSE 0 END)   AS dq60_plus_balance,
        SUM(p.chargeoff_amt)                                             AS gross_chargeoffs,
        SUM(p.recovery_amt)                                              AS recoveries,
        COUNT(*) FILTER (WHERE p.status IN (0, 1, 2, 3))                 AS active_accounts
    FROM performance p
    GROUP BY p.month
)
SELECT
    m.month,
    m.active_accounts,
    m.ending_balance,
    m.dq30_plus_balance / NULLIF(m.ending_balance, 0)                      AS dq30_plus_rate,
    m.dq60_plus_balance / NULLIF(m.ending_balance, 0)                      AS dq60_plus_rate,
    m.gross_chargeoffs,
    m.recoveries,
    m.gross_chargeoffs - m.recoveries                                      AS net_chargeoffs,
    12 * (m.gross_chargeoffs - m.recoveries)
        / NULLIF(AVG(m.ending_balance) OVER (ORDER BY m.month ROWS BETWEEN 1 PRECEDING AND CURRENT ROW), 0)
                                                                           AS annualized_nco_rate,
    mac.unemployment_rate
FROM monthly m
LEFT JOIN macro mac ON mac.month = m.month
ORDER BY m.month;
