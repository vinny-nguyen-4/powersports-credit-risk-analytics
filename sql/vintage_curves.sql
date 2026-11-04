-- Vintage analysis: cumulative net loss rate (% of original amount financed)
-- by quarterly origination vintage and months on book (MOB).
WITH vintages AS (
    SELECT
        loan_id,
        amount_financed,
        strftime(orig_month, '%Y') || '-Q' || CAST(quarter(orig_month) AS VARCHAR) AS vintage
    FROM loans
),
vintage_size AS (
    SELECT vintage, COUNT(*) AS loans, SUM(amount_financed) AS orig_amount
    FROM vintages
    GROUP BY vintage
),
losses_by_mob AS (
    SELECT
        v.vintage,
        p.mob,
        SUM(p.chargeoff_amt - p.recovery_amt) AS net_loss
    FROM performance p
    JOIN vintages v USING (loan_id)
    GROUP BY v.vintage, p.mob
)
SELECT
    l.vintage,
    l.mob,
    s.loans,
    s.orig_amount,
    SUM(l.net_loss) OVER (PARTITION BY l.vintage ORDER BY l.mob) / s.orig_amount AS cum_net_loss_rate
FROM losses_by_mob l
JOIN vintage_size s USING (vintage)
ORDER BY l.vintage, l.mob;
