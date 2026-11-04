-- Portfolio segmentation: current exposure and delinquency (as-of month) plus
-- lifetime net loss rate on vintages seasoned at least 18 months.
-- Parameters: $as_of (DATE), $segment (column name chosen by the caller from a whitelist).
WITH loan_seg AS (
    SELECT
        loan_id,
        orig_month,
        amount_financed,
        product,
        new_used,
        CASE
            WHEN fico < 620 THEN '1: <620'
            WHEN fico < 660 THEN '2: 620-659'
            WHEN fico < 700 THEN '3: 660-699'
            WHEN fico < 740 THEN '4: 700-739'
            ELSE                 '5: 740+'
        END AS fico_band,
        CASE
            WHEN ltv < 0.80 THEN '1: <80%'
            WHEN ltv < 0.95 THEN '2: 80-94%'
            WHEN ltv < 1.05 THEN '3: 95-104%'
            ELSE                 '4: 105%+'
        END AS ltv_band,
        CAST(term_months AS VARCHAR) || ' mo' AS term_band
    FROM loans
),
snapshot AS (
    SELECT loan_id, status, balance
    FROM performance
    WHERE month = CAST($as_of AS DATE) AND status IN (0, 1, 2, 3)
),
lifetime AS (
    SELECT loan_id, SUM(chargeoff_amt - recovery_amt) AS net_loss
    FROM performance
    GROUP BY loan_id
)
SELECT
    ls.{segment}                                                           AS segment,
    COUNT(*)                                                               AS loans_originated,
    SUM(ls.amount_financed)                                                AS amount_originated,
    SUM(s.balance)                                                         AS current_balance,
    SUM(CASE WHEN s.status IN (1, 2, 3) THEN s.balance ELSE 0 END)
        / NULLIF(SUM(s.balance), 0)                                        AS dq30_plus_rate,
    SUM(CASE WHEN ls.orig_month <= CAST($as_of AS DATE) - INTERVAL 18 MONTH THEN lt.net_loss END)
        / NULLIF(SUM(CASE WHEN ls.orig_month <= CAST($as_of AS DATE) - INTERVAL 18 MONTH
                          THEN ls.amount_financed END), 0)                 AS seasoned_net_loss_rate
FROM loan_seg ls
LEFT JOIN snapshot s USING (loan_id)
LEFT JOIN lifetime lt USING (loan_id)
GROUP BY 1
ORDER BY 1;
