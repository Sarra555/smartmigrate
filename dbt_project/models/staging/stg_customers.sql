-- ============================================================
-- dbt staging model : stg_customers.sql
-- Généré par SmartMigrate Schema Analyst
-- Table source : customers  →  Table cible : customers
-- Complexité   : high
-- ============================================================

{{ config(materialized='table') }}

WITH source AS (
    SELECT * FROM {{ source('erp_legacy', 'customers') }}
),

transformed AS (
    SELECT
        cst_id  AS  legacy_cst_id,
        cst_fname  AS  first_name,
        cst_lname  AS  last_name,
        cst_email  AS  email,
        CASE WHEN cst_status IN ('1','A','active') THEN 'active' ELSE 'inactive' END  AS  status  -- normalize_status,
        cst_country  AS  country_code,
        cst_phone  AS  phone,
        cst_created_dt  AS  created_at
    FROM source
    WHERE 1=1
)

SELECT * FROM transformed