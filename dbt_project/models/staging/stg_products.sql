-- ============================================================
-- dbt staging model : stg_products.sql
-- Généré par SmartMigrate Schema Analyst
-- Table source : products  →  Table cible : products
-- Complexité   : medium
-- ============================================================

{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('erp_legacy', 'products') }}
),

transformed AS (
    SELECT
        prod_id  AS  legacy_prod_id,
        prod_ref  AS  reference,
        prod_name  AS  name,
        prod_category  AS  category,
        CAST(prod_price AS NUMERIC)  AS  unit_price,
        (UPPER(TRIM(prod_currency)))  AS  currency_code  -- normalize_country,
        CAST(prod_stock AS NUMERIC)  AS  stock_qty,
        prod_unit  AS  unit,
        (CASE WHEN UPPER(prod_active) IN ('Y','1','YES','TRUE') THEN 1 ELSE 0 END)  AS  is_active  -- normalize_bool,
        prod_created  AS  created_at
    FROM source
    WHERE 1=1
      AND prod_price >= 0 OR prod_price IS NULL  -- exclure prix négatifs
      AND prod_stock >= 0 OR prod_stock IS NULL  -- exclure stocks négatifs (bug legacy)
)

SELECT * FROM transformed