-- ============================================================
-- SmartMigrate — Schéma CIBLE (Cloud PostgreSQL / Supabase)
-- Nommage moderne, types corrects, contraintes explicites
-- Comparé au schéma source legacy pour le schema mapping agent
-- ============================================================

-- Extension UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ────────────────────────────────────────────────────────────
-- 1. RÉFÉRENTIELS
-- ────────────────────────────────────────────────────────────

CREATE TABLE dim_country (
    country_code   CHAR(2)      PRIMARY KEY,          -- ISO 3166-1 alpha-2
    country_name   VARCHAR(100) NOT NULL,
    region         VARCHAR(50)
);

CREATE TABLE dim_currency (
    currency_code  CHAR(3)      PRIMARY KEY,          -- ISO 4217
    currency_name  VARCHAR(50)  NOT NULL,
    symbol         VARCHAR(5)
);

CREATE TABLE dim_status_customer (
    status_code    VARCHAR(10)  PRIMARY KEY,          -- 'active' | 'inactive'
    label          VARCHAR(50)  NOT NULL
);

CREATE TABLE dim_status_order (
    status_code    VARCHAR(20)  PRIMARY KEY,          -- 'pending' | 'confirmed' | ...
    label          VARCHAR(50)  NOT NULL,
    is_terminal    BOOLEAN      DEFAULT FALSE
);

-- ────────────────────────────────────────────────────────────
-- 2. CLIENTS
-- Source legacy : cst_* colonnes, formats incohérents
-- Cible : nommage explicite, types corrects, pays normalisé
-- ────────────────────────────────────────────────────────────

CREATE TABLE customers (
    customer_id    SERIAL        PRIMARY KEY,
    legacy_cst_id  INTEGER       UNIQUE,               -- FK vers source pour reconciliation
    first_name     VARCHAR(100),
    last_name      VARCHAR(100),
    company_name   VARCHAR(200),
    email          VARCHAR(255)  UNIQUE,
    phone          VARCHAR(30),                        -- format E.164 normalisé
    country_code   CHAR(2)       REFERENCES dim_country(country_code),
    city           VARCHAR(100),
    address        TEXT,
    segment        VARCHAR(20)   CHECK (segment IN ('B2B','B2C','VIP')),
    status         VARCHAR(10)   NOT NULL DEFAULT 'active'
                                 REFERENCES dim_status_customer(status_code),
    is_company     BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at     DATE,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW(),
    notes          TEXT
);

-- ────────────────────────────────────────────────────────────
-- 3. PRODUITS
-- Source : prod_* , prix sans devise, stock peut être négatif
-- Cible  : devise normalisée, stock >= 0, booléen propre
-- ────────────────────────────────────────────────────────────

CREATE TABLE products (
    product_id     SERIAL        PRIMARY KEY,
    legacy_prod_id INTEGER       UNIQUE,
    reference      VARCHAR(50)   NOT NULL UNIQUE,
    name           VARCHAR(200)  NOT NULL,
    category       VARCHAR(100),
    unit_price     NUMERIC(12,2) CHECK (unit_price >= 0),
    currency_code  CHAR(3)       NOT NULL DEFAULT 'TND'
                                 REFERENCES dim_currency(currency_code),
    stock_qty      INTEGER       NOT NULL DEFAULT 0 CHECK (stock_qty >= 0),
    unit           VARCHAR(20),
    is_active      BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at     DATE,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- 4. FOURNISSEURS
-- ────────────────────────────────────────────────────────────

CREATE TABLE suppliers (
    supplier_id    SERIAL        PRIMARY KEY,
    legacy_sup_id  INTEGER       UNIQUE,
    name           VARCHAR(200)  NOT NULL,
    country_code   CHAR(2)       REFERENCES dim_country(country_code),
    contact_name   VARCHAR(150),
    email          VARCHAR(255),
    phone          VARCHAR(30),
    rating         SMALLINT      CHECK (rating BETWEEN 1 AND 5),
    is_active      BOOLEAN       NOT NULL DEFAULT TRUE,
    partner_since  DATE,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- 5. COMMANDES
-- Source : ord_* , statuts encodés (1/0/P/CONF/...), dates mixtes
-- Cible  : statut normalisé, dates typées, TVA séparée
-- ────────────────────────────────────────────────────────────

CREATE TABLE orders (
    order_id           SERIAL        PRIMARY KEY,
    legacy_ord_id      INTEGER       UNIQUE,
    order_reference    VARCHAR(30)   NOT NULL UNIQUE,
    customer_id        INTEGER       NOT NULL REFERENCES customers(customer_id),
    order_date         DATE          NOT NULL,
    expected_delivery  DATE,
    delivered_at       DATE,
    status             VARCHAR(20)   NOT NULL DEFAULT 'pending'
                                     REFERENCES dim_status_order(status_code),
    subtotal_ht        NUMERIC(14,2) CHECK (subtotal_ht >= 0),
    tva_rate           NUMERIC(5,2)  CHECK (tva_rate >= 0),
    total_ttc          NUMERIC(14,2) GENERATED ALWAYS AS
                         (subtotal_ht * (1 + tva_rate / 100)) STORED,
    channel            VARCHAR(20)   CHECK (channel IN ('WEB','TEL','STORE','API')),
    notes              TEXT,
    migrated_at        TIMESTAMPTZ   DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- 6. LIGNES DE COMMANDE
-- ────────────────────────────────────────────────────────────

CREATE TABLE order_lines (
    line_id        SERIAL        PRIMARY KEY,
    legacy_line_id INTEGER       UNIQUE,
    order_id       INTEGER       NOT NULL REFERENCES orders(order_id),
    product_id     INTEGER       NOT NULL REFERENCES products(product_id),
    quantity       INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price     NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    discount_pct   NUMERIC(5,2)  NOT NULL DEFAULT 0 CHECK (discount_pct BETWEEN 0 AND 100),
    line_total     NUMERIC(14,2) GENERATED ALWAYS AS
                     (quantity * unit_price * (1 - discount_pct / 100)) STORED,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- 7. TABLE DE RECONCILIATION (meta-migration)
-- Trace chaque enregistrement migré pour audit
-- ────────────────────────────────────────────────────────────

CREATE TABLE migration_log (
    log_id          SERIAL        PRIMARY KEY,
    migration_run   VARCHAR(50)   NOT NULL,            -- ex: "run_2024-01-15_v1"
    table_name      VARCHAR(50)   NOT NULL,
    legacy_id       INTEGER,
    target_id       INTEGER,
    status          VARCHAR(20)   NOT NULL             -- 'success' | 'skipped' | 'error'
                    CHECK (status IN ('success','skipped','error')),
    error_message   TEXT,
    agent_decision  TEXT,                              -- ce que l'agent AI a décidé
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

-- Index pour performance de reconciliation
CREATE INDEX idx_migration_log_table  ON migration_log(table_name);
CREATE INDEX idx_migration_log_run    ON migration_log(migration_run);
CREATE INDEX idx_orders_customer      ON orders(customer_id);
CREATE INDEX idx_order_lines_order    ON order_lines(order_id);
CREATE INDEX idx_order_lines_product  ON order_lines(product_id);

-- ────────────────────────────────────────────────────────────
-- 8. DONNÉES DE RÉFÉRENCE INITIALES
-- ────────────────────────────────────────────────────────────

INSERT INTO dim_country VALUES
('TN','Tunisie','Afrique du Nord'),
('FR','France','Europe de l''Ouest'),
('DZ','Algérie','Afrique du Nord'),
('MA','Maroc','Afrique du Nord'),
('DE','Allemagne','Europe de l''Ouest'),
('US','États-Unis','Amérique du Nord'),
('CN','Chine','Asie de l''Est'),
('IT','Italie','Europe de l''Ouest'),
('ES','Espagne','Europe de l''Ouest'),
('TR','Turquie','Moyen-Orient');

INSERT INTO dim_currency VALUES
('TND','Dinar Tunisien','DT'),
('EUR','Euro','€'),
('USD','Dollar Américain','$');

INSERT INTO dim_status_customer VALUES
('active','Client actif'),
('inactive','Client inactif');

INSERT INTO dim_status_order VALUES
('pending',  'En attente',   FALSE),
('confirmed','Confirmée',    FALSE),
('shipped',  'Expédiée',     FALSE),
('delivered','Livrée',       TRUE),
('cancelled','Annulée',      TRUE);
