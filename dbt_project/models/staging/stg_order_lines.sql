-- ============================================================
-- dbt staging model : stg_order_lines.sql
-- Généré par SmartMigrate Schema Analyst
-- Table source : order_lines  →  Table cible : order_lines
-- Complexité   : medium
-- ============================================================

{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('erp_legacy', 'order_lines') }}
),

transformed AS (
    SELECT
        line_id  AS  legacy_line_id,
        line_ord_id  AS  order_id,
        line_prod_id  AS  product_id,
        CAST(line_qty AS NUMERIC)  AS  quantity,
        line_unit_price  AS  unit_price,
        CAST(line_discount AS NUMERIC)  AS  discount_pct
    FROM source
    WHERE 1=1
      AND line_qty > 0 OR line_qty IS NULL       -- exclure quantités nulles ou négatives
)

SELECT * FROM transformed