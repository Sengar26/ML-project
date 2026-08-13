-- Phase 1b — product x store x week panel.
--
-- Run through DuckDB by src/build_panel.py, which parameterises the three paths and the
-- two filter thresholds, then counts rows between stages to record what each filter costs.
-- Written as a sequence of tables rather than one nested query precisely so those
-- intermediate counts are available: "record what each data filter costs" is a working
-- agreement, not an afterthought.
--
-- Metric definitions, matching the README:
--   realised price = sum(SALES_VALUE) / sum(QUANTITY)                        [quantity-weighted]
--   base price     = sum(SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / sum(QUANTITY)
--   discount depth = 1 - realised / base
--
-- RETAIL_DISC and COUPON_MATCH_DISC are stored NEGATIVE. abs() is deliberate, and the
-- builder asserts base >= realised afterwards rather than trusting it.

-- Stage 1 -----------------------------------------------------------------------------
-- Transaction rows with a usable unit price. QUANTITY <= 0 rows (returns, weighted items
-- recorded with zero count) cannot produce a per-unit price and are dropped here.
CREATE OR REPLACE TABLE txn_clean AS
SELECT
    PRODUCT_ID,
    STORE_ID,
    WEEK_NO,
    QUANTITY,
    SALES_VALUE,
    SALES_VALUE + abs(RETAIL_DISC) + abs(COUPON_MATCH_DISC) AS base_value
FROM read_csv_auto($txn_path)
WHERE QUANTITY > 0
  AND SALES_VALUE > 0;

-- Stage 2 -----------------------------------------------------------------------------
-- Aggregate to the panel grain. Cells absent here are UNOBSERVED, not zero: the
-- transaction file records sales, and it cannot distinguish "stocked, sold none" from
-- "not stocked". They are left absent rather than imputed as zeros, which would put
-- log(0) into the demand equation and fabricate variation the data does not contain.
CREATE OR REPLACE TABLE cell AS
SELECT
    PRODUCT_ID,
    STORE_ID,
    WEEK_NO,
    sum(QUANTITY)                        AS units,
    sum(SALES_VALUE)                     AS revenue,
    count(*)                             AS n_baskets,
    sum(SALES_VALUE) / sum(QUANTITY)     AS realised_price,
    sum(base_value)  / sum(QUANTITY)     AS base_price
FROM txn_clean
GROUP BY 1, 2, 3;

-- Stage 3 -----------------------------------------------------------------------------
-- Promo calendar. causal_data carries rows only where there was promotional activity, so
-- an absent row means no display and no mailer. Flags are cast to VARCHAR because the
-- original release codes them as characters and some redistributions as integers.
CREATE OR REPLACE TABLE cell_promo AS
WITH causal AS (
    SELECT
        PRODUCT_ID,
        STORE_ID,
        WEEK_NO,
        max(CASE WHEN CAST(display AS VARCHAR) NOT IN ('0', '') THEN 1 ELSE 0 END) AS display_flag,
        max(CASE WHEN CAST(mailer  AS VARCHAR) NOT IN ('0', '') THEN 1 ELSE 0 END) AS mailer_flag
    FROM read_csv_auto($causal_path)
    GROUP BY 1, 2, 3
)
SELECT
    c.*,
    coalesce(z.display_flag, 0) AS display_flag,
    coalesce(z.mailer_flag, 0)  AS mailer_flag,
    CASE WHEN coalesce(z.display_flag, 0) + coalesce(z.mailer_flag, 0) > 0 THEN 1 ELSE 0 END
        AS promo,
    1 - c.realised_price / c.base_price AS discount_depth
FROM cell c
LEFT JOIN causal z
       ON c.PRODUCT_ID = z.PRODUCT_ID
      AND c.STORE_ID   = z.STORE_ID
      AND c.WEEK_NO    = z.WEEK_NO;

-- Stage 4 -----------------------------------------------------------------------------
-- Product hierarchy. INNER JOIN: a product with no row in product.csv has no department,
-- and cannot be grouped or interpreted downstream.
CREATE OR REPLACE TABLE cell_labeled AS
SELECT
    c.*,
    p.DEPARTMENT,
    p.COMMODITY_DESC,
    p.BRAND
FROM cell_promo c
JOIN read_csv_auto($product_path) p USING (PRODUCT_ID);

-- Stage 5 -----------------------------------------------------------------------------
-- Filter A: products observed in at least $min_weeks distinct weeks. Two-way fixed effects
-- need within-product variation over time; a product seen in a handful of weeks contributes
-- dummies without contributing identification.
CREATE OR REPLACE TABLE cell_hist AS
SELECT * FROM cell_labeled
WHERE PRODUCT_ID IN (
    SELECT PRODUCT_ID FROM cell_labeled
    GROUP BY PRODUCT_ID
    HAVING count(DISTINCT WEEK_NO) >= $min_weeks
);

-- Stage 6 -----------------------------------------------------------------------------
-- Filter B: products whose realised price actually moves. With no price variation there is
-- no elasticity to estimate — the product contributes only noise and a fixed effect.
CREATE OR REPLACE TABLE cell_var AS
SELECT * FROM cell_hist
WHERE PRODUCT_ID IN (
    SELECT PRODUCT_ID FROM cell_hist
    GROUP BY PRODUCT_ID
    HAVING stddev_samp(realised_price) / nullif(avg(realised_price), 0) >= $min_cv
);

-- Stage 7 -----------------------------------------------------------------------------
-- Final panel, with the analysis columns and the Phase 4c instrument.
--
-- The instrument is the mean log price of the same product across all OTHER stores in the
-- same week — leave-one-out, so a store's own price is never inside its own instrument.
-- It is NULL where the product appears in only one store that week; those rows stay in the
-- panel and drop out of the IV specification only.
CREATE OR REPLACE TABLE panel AS
SELECT
    PRODUCT_ID,
    STORE_ID,
    WEEK_NO,
    units,
    revenue,
    n_baskets,
    realised_price,
    base_price,
    discount_depth,
    display_flag,
    mailer_flag,
    promo,
    DEPARTMENT,
    COMMODITY_DESC,
    BRAND,
    ln(units)           AS log_units,
    ln(realised_price)  AS log_price,
    ln(base_price)      AS log_base_price,
    CASE WHEN count(*) OVER w > 1
         THEN (sum(ln(realised_price)) OVER w - ln(realised_price))
              / (count(*) OVER w - 1)
    END                 AS log_iv_price,
    count(*) OVER w - 1 AS n_other_stores
FROM cell_var
WINDOW w AS (PARTITION BY PRODUCT_ID, WEEK_NO);
