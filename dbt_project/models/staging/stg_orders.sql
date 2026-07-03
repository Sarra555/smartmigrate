-- ============================================================
-- dbt staging model : stg_orders.sql
-- Généré par SmartMigrate Schema Analyst
-- Table source : orders  →  Table cible : orders
-- Complexité   : high
-- ============================================================

{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('erp_legacy', 'orders') }}
),

transformed AS (
    SELECT
        ord_id  AS  legacy_ord_id,
        ord_ref  AS  order_reference,
        ord_cst_id  AS  customer_id,
        ord_date  AS  order_date,
        ord_exp_deliver  AS  expected_delivery,
        ord_delivered  AS  delivered_at,
        CASE WHEN ord_status IN ('P','PEND','pending','0') THEN 'pending' WHEN ord_status IN ('C','CONF','confirmed','1') THEN 'confirmed' WHEN ord_status IN ('S','SHIP','shipped','2') THEN 'shipped' WHEN ord_status IN ('D','DELIV','delivered','3') THEN 'delivered' WHEN ord_status IN ('X','CANC','cancelled','9') THEN 'cancelled' ELSE 'pending' END  AS  status  -- normalize_status,
        ord_total_ht  AS  subtotal_ht,
        ord_tva  AS  tva_rate,
        ord_channel  AS  channel,
        ord_notes  AS  notes
    FROM source
    WHERE 1=1
)

SELECT * FROM transformed