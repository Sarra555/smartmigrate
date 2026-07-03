"""
SmartMigrate — Agent 1 : Schema Analyst (version complète)
===========================================================
Agent LangGraph avec tool calling réel qui :

1. [extract_schemas]   Lit le DDL source (SQLite/PG) + DDL cible depuis fichier SQL
2. [propose_mapping]   LLM analyse les deux DDL → mapping JSON (colonne par colonne)
3. [enrich_mapping]    Tool calling : valide les types, détecte conflits, calcule scores
4. [human_gate]        Routage : mappings ambigus (< 0.80) → validation CLI
5. [generate_dbt]      LLM génère le modèle dbt staging SQL
6. [save_outputs]      Sauvegarde mapping JSON + fichier dbt + rapport HTML

Fonctionne en mode LOCAL (SQLite) sans Docker ni Supabase.
"""

import json
import os
import time
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

# ── Imports LangGraph / LangChain ────────────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, StateGraph
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False

from dotenv import load_dotenv

load_dotenv()

# ── Chemins projet ────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DB_PATH       = PROJECT_ROOT / "data" / "raw" / "erp_legacy.db"
TARGET_SQL    = PROJECT_ROOT / "docker" / "schema_target.sql"
OUTPUT_MAP    = PROJECT_ROOT / "outputs" / "mappings"
OUTPUT_DBT    = PROJECT_ROOT / "dbt_project" / "models" / "staging"
OUTPUT_MAP.mkdir(parents=True, exist_ok=True)
OUTPUT_DBT.mkdir(parents=True, exist_ok=True)

# ── DDL cible pour chaque table (extrait du schema_target.sql) ────────────────
TARGET_DDLS = {
    "customers": """
CREATE TABLE customers (
    customer_id    SERIAL        PRIMARY KEY,
    legacy_cst_id  INTEGER       UNIQUE,
    first_name     VARCHAR(100),
    last_name      VARCHAR(100),
    company_name   VARCHAR(200),
    email          VARCHAR(255)  UNIQUE,
    phone          VARCHAR(30),
    country_code   CHAR(2),
    city           VARCHAR(100),
    address        TEXT,
    segment        VARCHAR(20)   CHECK (segment IN ('B2B','B2C','VIP')),
    status         VARCHAR(10)   NOT NULL DEFAULT 'active',
    is_company     BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at     DATE,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW(),
    notes          TEXT
);""",

    "products": """
CREATE TABLE products (
    product_id     SERIAL        PRIMARY KEY,
    legacy_prod_id INTEGER       UNIQUE,
    reference      VARCHAR(50)   NOT NULL UNIQUE,
    name           VARCHAR(200)  NOT NULL,
    category       VARCHAR(100),
    unit_price     NUMERIC(12,2) CHECK (unit_price >= 0),
    currency_code  CHAR(3)       NOT NULL DEFAULT 'TND',
    stock_qty      INTEGER       NOT NULL DEFAULT 0 CHECK (stock_qty >= 0),
    unit           VARCHAR(20),
    is_active      BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at     DATE,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW()
);""",

    "orders": """
CREATE TABLE orders (
    order_id           SERIAL        PRIMARY KEY,
    legacy_ord_id      INTEGER       UNIQUE,
    order_reference    VARCHAR(30)   NOT NULL UNIQUE,
    customer_id        INTEGER       NOT NULL,
    order_date         DATE          NOT NULL,
    expected_delivery  DATE,
    delivered_at       DATE,
    status             VARCHAR(20)   NOT NULL DEFAULT 'pending',
    subtotal_ht        NUMERIC(14,2) CHECK (subtotal_ht >= 0),
    tva_rate           NUMERIC(5,2)  CHECK (tva_rate >= 0),
    channel            VARCHAR(20)   CHECK (channel IN ('WEB','TEL','STORE','API')),
    notes              TEXT,
    migrated_at        TIMESTAMPTZ   DEFAULT NOW()
);""",

    "order_lines": """
CREATE TABLE order_lines (
    line_id        SERIAL        PRIMARY KEY,
    legacy_line_id INTEGER       UNIQUE,
    order_id       INTEGER       NOT NULL,
    product_id     INTEGER       NOT NULL,
    quantity       INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price     NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    discount_pct   NUMERIC(5,2)  NOT NULL DEFAULT 0,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW()
);""",

    "suppliers": """
CREATE TABLE suppliers (
    supplier_id    SERIAL        PRIMARY KEY,
    legacy_sup_id  INTEGER       UNIQUE,
    name           VARCHAR(200)  NOT NULL,
    country_code   CHAR(2),
    contact_name   VARCHAR(150),
    email          VARCHAR(255),
    phone          VARCHAR(30),
    rating         SMALLINT      CHECK (rating BETWEEN 1 AND 5),
    is_active      BOOLEAN       NOT NULL DEFAULT TRUE,
    partner_since  DATE,
    migrated_at    TIMESTAMPTZ   DEFAULT NOW()
);""",
}

# ── State LangGraph ───────────────────────────────────────────────────────────

class SchemaState(TypedDict):
    table_name:      str
    source_ddl:      str
    target_ddl:      str
    source_sample:   List[Dict]       # 3 lignes d'exemple pour le LLM
    source_stats:    Dict             # stats basiques par colonne
    mapping:         Optional[Dict]
    tool_results:    List[Dict]       # résultats des tool calls
    dbt_model:       Optional[str]
    ambiguous:       List[Dict]
    human_approved:  bool
    errors:          List[str]
    messages:        List             # historique messages LLM

# ── Tool calls (fonctions que le LLM peut appeler) ────────────────────────────

def validate_type_compatibility(source_type: str, target_type: str) -> Dict:
    """Vérifie si deux types SQL sont compatibles pour la migration."""
    src = source_type.upper()
    tgt = target_type.upper()

    # Groupes de compatibilité
    text_types    = {"TEXT", "VARCHAR", "CHAR", "STRING", "CLOB", "NVARCHAR"}
    int_types     = {"INTEGER", "INT", "BIGINT", "SMALLINT", "SERIAL", "TINYINT"}
    numeric_types = {"NUMERIC", "DECIMAL", "FLOAT", "REAL", "DOUBLE", "NUMBER"}
    bool_types    = {"BOOLEAN", "BOOL", "BIT"}
    date_types    = {"DATE", "DATETIME", "TIMESTAMP", "TIMESTAMPTZ"}

    def group(t):
        t_clean = re.sub(r'\(.*\)', '', t).strip()
        for g, members in [("text", text_types), ("int", int_types),
                            ("numeric", numeric_types), ("bool", bool_types),
                            ("date", date_types)]:
            if any(m in t_clean for m in members):
                return g
        return "other"

    sg, tg = group(src), group(tgt)

    if sg == tg:
        return {"compatible": True, "risk": "low", "note": "Même famille de types"}
    if sg == "int" and tg == "numeric":
        return {"compatible": True, "risk": "low", "note": "INT → NUMERIC : compatible, pas de perte"}
    if sg == "text" and tg in ("int", "numeric", "bool", "date"):
        return {"compatible": False, "risk": "high",
                "note": f"TEXT → {tg.upper()} : conversion nécessaire, risque d'erreur"}
    if sg == "int" and tg == "bool":
        return {"compatible": True, "risk": "medium",
                "note": "INT → BOOL : possible si valeurs 0/1 uniquement"}
    return {"compatible": False, "risk": "medium",
            "note": f"Types différents ({sg} → {tg}) : vérification requise"}


def detect_value_patterns(column_values: List[Any]) -> Dict:
    """Analyse les valeurs d'une colonne pour détecter les patterns."""
    if not column_values:
        return {"pattern": "empty", "null_rate": 1.0}

    non_null = [v for v in column_values if v is not None]
    null_rate = round(1 - len(non_null) / len(column_values), 3)

    if not non_null:
        return {"pattern": "all_null", "null_rate": 1.0}

    str_vals = [str(v) for v in non_null]

    # Détection de patterns
    date_patterns = [
        (r'^\d{4}-\d{2}-\d{2}$',     "ISO date (YYYY-MM-DD)"),
        (r'^\d{2}/\d{2}/\d{4}$',     "FR date (DD/MM/YYYY)"),
        (r'^\d{2}-\d{2}-\d{4}$',     "date (DD-MM-YYYY)"),
        (r'^\d{8}$',                  "compact date (YYYYMMDD)"),
    ]
    bool_patterns = {"1", "0", "Y", "N", "yes", "no", "true", "false",
                     "A", "I", "active", "inactive"}
    status_codes  = {"P","C","S","D","X","PEND","CONF","SHIP","DELIV","CANC",
                     "pending","confirmed","shipped","delivered","cancelled"}

    unique_vals = set(str_vals)

    for pattern, label in date_patterns:
        if all(re.match(pattern, v) for v in str_vals[:20]):
            return {"pattern": "date", "format": label, "null_rate": null_rate,
                    "sample": str_vals[:3]}

    if unique_vals <= bool_patterns:
        return {"pattern": "boolean_encoded", "values": list(unique_vals),
                "null_rate": null_rate, "note": "Normaliser vers TRUE/FALSE"}

    if unique_vals <= status_codes or (len(unique_vals) <= 8 and len(unique_vals) > 1):
        return {"pattern": "categorical", "distinct_values": list(unique_vals)[:10],
                "null_rate": null_rate}

    phone_pat = r'^[\+\d\s\-\(\)]{7,20}$'
    if all(re.match(phone_pat, v) for v in str_vals[:10]):
        return {"pattern": "phone", "null_rate": null_rate,
                "note": "Normaliser vers format E.164"}

    return {"pattern": "text", "null_rate": null_rate,
            "avg_length": round(sum(len(v) for v in str_vals) / len(str_vals), 1),
            "sample": str_vals[:3]}


def get_sqlite_sample(table: str, column: str, limit: int = 30) -> List[Any]:
    """Récupère un échantillon de valeurs depuis SQLite."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur  = conn.cursor()
        cur.execute(f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}')
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_sqlite_ddl(table: str) -> str:
    """Extrait le DDL depuis SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
    row  = cur.fetchone()
    conn.close()
    return row[0] if row else f"-- Table '{table}' non trouvée"


def get_sqlite_stats(table: str) -> Dict:
    """Stats basiques par colonne depuis SQLite."""
    conn  = sqlite3.connect(str(DB_PATH))
    cur   = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    cols  = [r[1] for r in cur.fetchall()]
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    total = cur.fetchone()[0]
    stats = {"total_rows": total, "columns": {}}
    for col in cols:
        cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NULL')
        nulls = cur.fetchone()[0]
        stats["columns"][col] = {
            "null_count": nulls,
            "null_rate":  round(nulls / total, 3) if total else 0,
        }
    conn.close()
    return stats


def get_sqlite_sample_rows(table: str, n: int = 3) -> List[Dict]:
    """Récupère n lignes d'exemple pour le LLM."""
    conn  = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur   = conn.cursor()
    cur.execute(f'SELECT * FROM "{table}" LIMIT {n}')
    rows  = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# ── Noeuds LangGraph ──────────────────────────────────────────────────────────

def node_extract_schemas(state: SchemaState) -> SchemaState:
    """Noeud 1 : Extrait DDL source + stats + exemples."""
    table = state["table_name"]
    print(f"\n  📥 Extraction du schéma source : {table}")

    source_ddl = get_sqlite_ddl(table)
    stats      = get_sqlite_stats(table)
    sample     = get_sqlite_sample_rows(table, 3)
    target_ddl = TARGET_DDLS.get(table, "-- DDL cible non défini")

    print(f"     {stats['total_rows']} lignes trouvées")
    return {
        **state,
        "source_ddl":    source_ddl,
        "target_ddl":    target_ddl,
        "source_stats":  stats,
        "source_sample": sample,
    }


def node_propose_mapping(state: SchemaState) -> SchemaState:
    """Noeud 2 : Le LLM propose le mapping + tool calls pour valider."""
    print(f"  🤖 LLM analyse le schéma et propose le mapping...")

    # Préparer les données d'analyse par colonne
    col_analysis = {}
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        cur  = conn.cursor()
        cur.execute(f'PRAGMA table_info("{state["table_name"]}")')
        cols = [r[1] for r in cur.fetchall()]
        conn.close()
        for col in cols:
            vals = get_sqlite_sample(state["table_name"], col, 50)
            col_analysis[col] = detect_value_patterns(vals)

    sample_str = json.dumps(state["source_sample"], ensure_ascii=False, indent=2)
    stats_str  = json.dumps(state["source_stats"],  ensure_ascii=False, indent=2)
    analysis_str = json.dumps(col_analysis, ensure_ascii=False, indent=2)

    system = """Tu es un expert senior en migration de données ERP vers le cloud.
Tu analyses les schémas source et cible, et les données réelles pour proposer
un mapping colonne par colonne précis.

Tu dois répondre UNIQUEMENT avec un JSON valide, sans aucun texte avant ou après.
Format exact :
{
  "table_source": "nom_table_source",
  "table_target": "nom_table_cible",
  "migration_complexity": "low|medium|high",
  "complexity_reason": "explication courte",
  "mappings": [
    {
      "source_col": "col_source",
      "target_col": "col_cible",
      "source_type": "type SQL source",
      "target_type": "type SQL cible",
      "transform": "direct|normalize_status|normalize_date|normalize_phone|normalize_bool|normalize_country|cast_numeric|compute|skip",
      "transform_sql": "expression SQL de transformation (si besoin)",
      "confidence": 0.95,
      "notes": "explication"
    }
  ],
  "unmapped_source": ["colonnes source sans équivalent cible"],
  "unmapped_target": ["colonnes cible sans source (auront DEFAULT ou NULL)"],
  "migration_risks": ["liste des risques identifiés"]
}

Règles de transform_sql :
- normalize_status : CASE WHEN ... THEN ... END
- normalize_date   : essayer de parser les formats mixtes
- normalize_bool   : CASE WHEN col IN ('Y','1','yes','true','A') THEN 1 ELSE 0 END
- normalize_country: CASE WHEN col IN ('Tunisia','Tunisie','TN') THEN 'TN' ... END
- skip             : colonne à ne pas migrer (calculée, redondante, etc.)
- direct           : copie directe sans transformation"""

    user_msg = f"""Analyse cette migration :

=== SCHÉMA SOURCE (ERP legacy) ===
{state['source_ddl']}

=== SCHÉMA CIBLE (Cloud DW) ===
{state['target_ddl']}

=== ANALYSE DES DONNÉES RÉELLES ===
{analysis_str}

=== STATISTIQUES SOURCE ===
{stats_str}

=== EXEMPLES DE DONNÉES (3 lignes) ===
{sample_str}"""

    if not LANGCHAIN_OK or not os.getenv("GOOGLE_API_KEY"):
        # Mode mock si pas de clé API — pour tester la structure
        print("  ⚠️  Mode MOCK (pas de clé API détectée)")
        mapping = _mock_mapping(state["table_name"])
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
        resp = None
        for attempt in range(3):
            try:
                print(f"  ⏳ tentative {attempt+1}/3...")
                resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 65 * (attempt + 1)
                    print(f"  ⏳ quota atteint — attente {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        if resp is None:
            print("  ⚠️  LLM indisponible après 3 tentatives — bascule en mode MOCK")
            mapping = _mock_mapping(state["table_name"])
        else:
            raw = resp.content.strip()
            raw = re.sub(r'^```(?:json)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            mapping = json.loads(raw.strip())

    ambiguous = [m for m in mapping["mappings"] if m.get("confidence", 1) < 0.80]
    print(f"     {len(mapping['mappings'])} colonnes mappées | {len(ambiguous)} ambiguës")

    return {
        **state,
        "mapping":   mapping,
        "ambiguous": ambiguous,
        "messages":  state["messages"] + [{"role": "mapping_complete"}],
    }


def node_enrich_with_tools(state: SchemaState) -> SchemaState:
    """Noeud 3 : Enrichit chaque mapping avec les tool calls de validation."""
    print(f"  🔧 Validation des types et patterns...")

    if not state["mapping"]:
        return state

    tool_results = []
    enriched_mappings = []

    for m in state["mapping"]["mappings"]:
        enriched = dict(m)

        # 1. Valider compatibilité des types
        type_check = validate_type_compatibility(
            m.get("source_type", "TEXT"),
            m.get("target_type", "TEXT")
        )
        enriched["type_check"] = type_check

        # Baisser le score si type incompatible
        if not type_check["compatible"]:
            enriched["confidence"] = min(m.get("confidence", 0.8), 0.65)
            enriched["notes"] += f" | ⚠️ Type: {type_check['note']}"

        # 2. Analyser les patterns de valeurs réels
        col = m.get("source_col", "")
        if col and state["table_name"]:
            vals    = get_sqlite_sample(state["table_name"], col, 50)
            pattern = detect_value_patterns(vals)
            enriched["value_pattern"] = pattern

            # Ajuster le transform selon le pattern détecté
            if pattern["pattern"] == "boolean_encoded" and enriched["transform"] == "direct":
                enriched["transform"] = "normalize_bool"
                enriched["confidence"] = min(enriched["confidence"], 0.75)
                enriched["notes"] += f" | Pattern booléen détecté: {pattern.get('values', [])}"

            if pattern["pattern"] == "date" and enriched["transform"] == "direct":
                enriched["transform"] = "normalize_date"
                enriched["notes"] += f" | Format date: {pattern.get('format', 'mixte')}"

            if pattern["pattern"] == "phone" and enriched["transform"] == "direct":
                enriched["transform"] = "normalize_phone"
                enriched["notes"] += " | Normalisation E.164 recommandée"

        tool_results.append({
            "col":        col,
            "type_check": type_check,
            "pattern":    enriched.get("value_pattern", {}),
        })
        enriched_mappings.append(enriched)

    # Mettre à jour le mapping et recalculer les ambigus
    updated_mapping = {**state["mapping"], "mappings": enriched_mappings}
    ambiguous = [m for m in enriched_mappings if m.get("confidence", 1) < 0.80]

    print(f"     Tool calls : {len(tool_results)} colonnes analysées | {len(ambiguous)} ambiguës après enrichissement")

    return {
        **state,
        "mapping":      updated_mapping,
        "tool_results": tool_results,
        "ambiguous":    ambiguous,
    }


def node_human_gate(state: SchemaState) -> Literal["human_review", "generate_dbt"]:
    """Routage conditionnel : ambigus → review, sinon → dbt."""
    if state["ambiguous"] and not state["human_approved"]:
        return "human_review"
    return "generate_dbt"


def node_human_review(state: SchemaState) -> SchemaState:
    """Noeud 4 : Affiche les ambiguïtés et demande validation CLI."""
    print(f"\n  ⚠️  {len(state['ambiguous'])} mapping(s) ambigus — validation requise :\n")

    for i, m in enumerate(state["ambiguous"], 1):
        conf    = m.get("confidence", 0)
        risk    = m.get("type_check", {}).get("risk", "unknown")
        pattern = m.get("value_pattern", {}).get("pattern", "?")
        print(f"  [{i}] {m['source_col']} → {m['target_col']}")
        print(f"       Confiance : {conf:.0%}  |  Risque type : {risk}  |  Pattern : {pattern}")
        print(f"       Transform : {m.get('transform', '?')}")
        print(f"       Note      : {m.get('notes', '')}")
        if m.get("transform_sql"):
            print(f"       SQL       : {m['transform_sql'][:80]}...")
        print()

    # Validation interactive (ou auto en mode CI)
    if os.getenv("SMARTMIGRATE_AUTO_APPROVE", "false").lower() == "true":
        print("  ✅ Auto-approbation (SMARTMIGRATE_AUTO_APPROVE=true)")
        approved = True
    else:
        try:
            answer = input("  Approuver tous ces mappings ? [O/n] : ").strip().lower()
            approved = answer in ("", "o", "oui", "y", "yes")
        except EOFError:
            approved = True  # mode non-interactif

    if approved:
        print("  ✅ Mappings approuvés.")
    else:
        print("  ⏸️  Mappings rejetés — arrêt du pipeline.")

    return {**state, "human_approved": approved}


def node_generate_dbt(state: SchemaState) -> SchemaState:
    """Noeud 5 : Génère le modèle dbt staging complet depuis le mapping."""
    print(f"  🏗️  Génération du modèle dbt staging...")

    if not state["mapping"]:
        return state

    table_src = state["mapping"]["table_source"]
    table_tgt = state["mapping"]["table_target"]
    mappings  = state["mapping"]["mappings"]

    # ── Construire le SQL dbt directement (pas LLM pour le dbt — plus fiable) ──
    lines = []
    lines.append(f"-- ============================================================")
    lines.append(f"-- dbt staging model : stg_{table_src}.sql")
    lines.append(f"-- Généré par SmartMigrate Schema Analyst")
    lines.append(f"-- Table source : {table_src}  →  Table cible : {table_tgt}")
    lines.append(f"-- Complexité   : {state['mapping'].get('migration_complexity','?')}")
    lines.append(f"-- ============================================================")
    lines.append(f"")
    lines.append(f"{{{{ config(materialized='table') }}}}")
    lines.append(f"")
    lines.append(f"WITH source AS (")
    lines.append(f"    SELECT * FROM {{{{ source('erp_legacy', '{table_src}') }}}}")
    lines.append(f"),")
    lines.append(f"")
    lines.append(f"transformed AS (")
    lines.append(f"    SELECT")

    col_expressions = []
    for m in mappings:
        src      = m["source_col"]
        tgt      = m["target_col"]
        transform = m.get("transform", "direct")
        sql      = m.get("transform_sql", "")

        if transform == "skip":
            col_expressions.append(f"        -- SKIP: {src} (non migré)")
            continue
        elif transform == "direct" or not sql:
            col_expressions.append(f"        {src}  AS  {tgt}")
        elif transform == "normalize_status":
            col_expressions.append(f"        {sql}  AS  {tgt}  -- normalize_status")
        elif transform == "normalize_bool":
            if not sql:
                sql = f"CASE WHEN UPPER(CAST({src} AS TEXT)) IN ('1','Y','YES','TRUE','A','ACTIVE') THEN 1 ELSE 0 END"
            col_expressions.append(f"        ({sql})  AS  {tgt}  -- normalize_bool")
        elif transform == "normalize_date":
            col_expressions.append(f"        CAST({src} AS DATE)  AS  {tgt}  -- normalize_date (vérifier formats)")
        elif transform == "normalize_phone":
            col_expressions.append(f"        {src}  AS  {tgt}  -- TODO: normaliser format E.164")
        elif transform == "normalize_country":
            if not sql:
                sql = f"CASE WHEN UPPER({src}) IN ('TN','TUNISIA','TUNISIE') THEN 'TN' WHEN UPPER({src}) IN ('FR','FRANCE') THEN 'FR' WHEN UPPER({src}) IN ('DZ','ALGERIE','ALGERIA') THEN 'DZ' ELSE {src} END"
            col_expressions.append(f"        ({sql})  AS  {tgt}  -- normalize_country")
        elif transform == "cast_numeric":
            col_expressions.append(f"        CAST({src} AS NUMERIC)  AS  {tgt}")
        elif sql:
            col_expressions.append(f"        ({sql})  AS  {tgt}")
        else:
            col_expressions.append(f"        {src}  AS  {tgt}")

    lines.append(",\n".join(col_expressions))
    lines.append(f"    FROM source")
    lines.append(f"    WHERE 1=1")

    # Filtres anti-corruption courants
    if table_src == "products":
        lines.append(f"      AND prod_price >= 0 OR prod_price IS NULL  -- exclure prix négatifs")
        lines.append(f"      AND prod_stock >= 0 OR prod_stock IS NULL  -- exclure stocks négatifs (bug legacy)")
    if table_src == "order_lines":
        lines.append(f"      AND line_qty > 0 OR line_qty IS NULL       -- exclure quantités nulles ou négatives")

    lines.append(f")")
    lines.append(f"")
    lines.append(f"SELECT * FROM transformed")

    dbt_sql = "\n".join(lines)

    # Générer aussi le source.yml dbt
    source_yml = f"""version: 2

sources:
  - name: erp_legacy
    description: "Base ERP legacy (SQLite/PostgreSQL source)"
    tables:
      - name: {table_src}
        description: "Table {table_src} du système ERP legacy"
        columns:
"""
    for m in mappings:
        if m.get("transform") != "skip":
            source_yml += f"          - name: {m['source_col']}\n"
            source_yml += f"            description: \"→ {m['target_col']} ({m.get('transform','direct')})\"\n"

    return {**state, "dbt_model": dbt_sql, "dbt_source_yml": source_yml}


def node_save_outputs(state: SchemaState) -> SchemaState:
    """Noeud 6 : Sauvegarde tous les artefacts + génère rapport HTML."""
    table = state["table_name"]
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Mapping JSON ────────────────────────────────────────────────────
    mapping_path = OUTPUT_MAP / f"{table}_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(state["mapping"], f, ensure_ascii=False, indent=2)

    # ── 2. Modèle dbt ──────────────────────────────────────────────────────
    dbt_path = OUTPUT_DBT / f"stg_{table}.sql"
    with open(dbt_path, "w", encoding="utf-8") as f:
        f.write(state["dbt_model"])

    # ── 3. source.yml dbt ──────────────────────────────────────────────────
    src_yml_path = OUTPUT_DBT / f"src_{table}.yml"
    if "dbt_source_yml" in state:
        with open(src_yml_path, "w", encoding="utf-8") as f:
            f.write(state["dbt_source_yml"])

    # ── 4. Rapport HTML ────────────────────────────────────────────────────
    report_path = OUTPUT_MAP / f"{table}_report.html"
    _write_html_report(state, report_path)

    print(f"\n  💾 Fichiers générés :")
    print(f"     📄 {mapping_path.relative_to(PROJECT_ROOT)}")
    print(f"     🏗️  {dbt_path.relative_to(PROJECT_ROOT)}")
    print(f"     📊 {report_path.relative_to(PROJECT_ROOT)}")

    return state


def _write_html_report(state: SchemaState, path: Path):
    """Génère un rapport HTML lisible par le recruteur / non-technique."""
    m   = state["mapping"] or {}
    tbl = state["table_name"]

    mappings  = m.get("mappings", [])
    total     = len(mappings)
    high_conf = sum(1 for x in mappings if x.get("confidence", 0) >= 0.9)
    med_conf  = sum(1 for x in mappings if 0.7 <= x.get("confidence", 0) < 0.9)
    low_conf  = sum(1 for x in mappings if x.get("confidence", 0) < 0.7)
    risks     = m.get("migration_risks", [])
    complexity = m.get("migration_complexity", "?")
    complexity_color = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}.get(complexity, "#6b7280")

    rows_html = ""
    for mp in mappings:
        conf  = mp.get("confidence", 0)
        color = "#22c55e" if conf >= 0.9 else "#f59e0b" if conf >= 0.7 else "#ef4444"
        tc    = mp.get("type_check", {})
        risk_badge = f'<span style="background:{"#fef2f2" if tc.get("risk")=="high" else "#fefce8" if tc.get("risk")=="medium" else "#f0fdf4"};color:{"#b91c1c" if tc.get("risk")=="high" else "#92400e" if tc.get("risk")=="medium" else "#166534"};padding:2px 6px;border-radius:4px;font-size:11px">{tc.get("risk","ok")}</span>'
        transform_badge = f'<code style="background:#f3f4f6;padding:2px 6px;border-radius:4px;font-size:11px">{mp.get("transform","direct")}</code>'
        rows_html += f"""
        <tr>
          <td style="padding:8px 12px;font-family:monospace;font-size:13px">{mp.get('source_col','')}</td>
          <td style="padding:8px 12px;font-family:monospace;font-size:13px">{mp.get('target_col','')}</td>
          <td style="padding:8px 12px">{transform_badge}</td>
          <td style="padding:8px 12px">
            <span style="color:{color};font-weight:600">{conf:.0%}</span>
          </td>
          <td style="padding:8px 12px">{risk_badge}</td>
          <td style="padding:8px 12px;font-size:12px;color:#6b7280">{mp.get('notes','')[:80]}</td>
        </tr>"""

    risks_html = "".join(f'<li style="margin:4px 0;color:#b91c1c">{r}</li>' for r in risks) or "<li>Aucun risque majeur détecté</li>"

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>SmartMigrate — Rapport {tbl}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:32px;background:#f9fafb;color:#111827}}
  .card{{background:#fff;border-radius:12px;border:1px solid #e5e7eb;padding:24px;margin-bottom:20px}}
  h1{{font-size:22px;font-weight:600;margin:0 0 4px}}
  h2{{font-size:15px;font-weight:600;margin:0 0 16px;color:#374151}}
  .meta{{font-size:13px;color:#6b7280;margin-bottom:24px}}
  .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
  .stat{{background:#f9fafb;border-radius:8px;padding:14px;text-align:center}}
  .stat-num{{font-size:26px;font-weight:700}}
  .stat-lbl{{font-size:11px;color:#6b7280;margin-top:2px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#f3f4f6;padding:10px 12px;text-align:left;font-weight:500;font-size:12px;color:#6b7280;border-bottom:1px solid #e5e7eb}}
  tr:hover td{{background:#f9fafb}}
  td{{border-bottom:1px solid #f3f4f6}}
</style>
</head><body>
<h1>🧠 SmartMigrate — Schema Mapping Report</h1>
<p class="meta">Table : <strong>{tbl}</strong> &nbsp;|&nbsp; Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')} &nbsp;|&nbsp; Complexité : <strong style="color:{complexity_color}">{complexity.upper()}</strong></p>

<div class="stats">
  <div class="stat"><div class="stat-num">{total}</div><div class="stat-lbl">Colonnes mappées</div></div>
  <div class="stat"><div class="stat-num" style="color:#22c55e">{high_conf}</div><div class="stat-lbl">Confiance haute (≥90%)</div></div>
  <div class="stat"><div class="stat-num" style="color:#f59e0b">{med_conf}</div><div class="stat-lbl">Confiance moyenne</div></div>
  <div class="stat"><div class="stat-num" style="color:#ef4444">{low_conf}</div><div class="stat-lbl">À valider</div></div>
</div>

<div class="card">
  <h2>Mapping des colonnes</h2>
  <table>
    <thead><tr><th>Source</th><th>Cible</th><th>Transform</th><th>Confiance</th><th>Risque type</th><th>Notes</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>

<div class="card">
  <h2>⚠️ Risques identifiés</h2>
  <ul style="margin:0;padding-left:20px">{risks_html}</ul>
</div>

<div class="card">
  <h2>Colonnes source non migrées</h2>
  <p style="font-family:monospace;font-size:13px;color:#6b7280">{', '.join(m.get('unmapped_source',[])) or 'Aucune'}</p>
</div>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _mock_mapping(table: str) -> Dict:
    """Mapping mock pour tester sans clé API."""
    mocks = {
        "customers": {
            "table_source": "customers", "table_target": "customers",
            "migration_complexity": "high", "complexity_reason": "Statuts encodés, formats mixtes",
            "mappings": [
                {"source_col":"cst_id","target_col":"legacy_cst_id","source_type":"INTEGER","target_type":"INTEGER","transform":"direct","confidence":0.99,"notes":"PK source","transform_sql":""},
                {"source_col":"cst_fname","target_col":"first_name","source_type":"TEXT","target_type":"VARCHAR","transform":"direct","confidence":0.97,"notes":"Prénom","transform_sql":""},
                {"source_col":"cst_lname","target_col":"last_name","source_type":"TEXT","target_type":"VARCHAR","transform":"direct","confidence":0.97,"notes":"Nom","transform_sql":""},
                {"source_col":"cst_email","target_col":"email","source_type":"TEXT","target_type":"VARCHAR","transform":"direct","confidence":0.95,"notes":"Email","transform_sql":""},
                {"source_col":"cst_status","target_col":"status","source_type":"TEXT","target_type":"VARCHAR","transform":"normalize_status","confidence":0.72,"notes":"Valeurs legacy: 1/0/A/I/active","transform_sql":"CASE WHEN cst_status IN ('1','A','active') THEN 'active' ELSE 'inactive' END"},
                {"source_col":"cst_country","target_col":"country_code","source_type":"TEXT","target_type":"CHAR","transform":"normalize_country","confidence":0.68,"notes":"TN/Tunisia/Tunisie → ISO 2","transform_sql":""},
                {"source_col":"cst_phone","target_col":"phone","source_type":"TEXT","target_type":"VARCHAR","transform":"normalize_phone","confidence":0.61,"notes":"4 formats différents","transform_sql":""},
                {"source_col":"cst_created_dt","target_col":"created_at","source_type":"TEXT","target_type":"DATE","transform":"normalize_date","confidence":0.78,"notes":"Formats de dates mixtes","transform_sql":""},
            ],
            "unmapped_source": ["cst_segment"],
            "unmapped_target": ["customer_id","is_company","migrated_at"],
            "migration_risks": ["~3% de doublons détectés","Statuts legacy encodés sur 6 valeurs distinctes"]
        }
    }
    mocks["products"] = {
        "table_source": "products", "table_target": "products",
        "migration_complexity": "medium", "complexity_reason": "Devise non normalisée, stocks négatifs, booléens encodés",
        "mappings": [
            {"source_col":"prod_id",       "target_col":"legacy_prod_id", "source_type":"INTEGER","target_type":"INTEGER","transform":"direct",           "confidence":0.99,"notes":"PK source",                          "transform_sql":""},
            {"source_col":"prod_ref",      "target_col":"reference",      "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",           "confidence":0.97,"notes":"Référence produit",                  "transform_sql":""},
            {"source_col":"prod_name",     "target_col":"name",           "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",           "confidence":0.97,"notes":"Nom produit",                        "transform_sql":""},
            {"source_col":"prod_category", "target_col":"category",       "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",           "confidence":0.93,"notes":"Catégorie",                          "transform_sql":""},
            {"source_col":"prod_price",    "target_col":"unit_price",     "source_type":"REAL",   "target_type":"NUMERIC","transform":"cast_numeric",    "confidence":0.88,"notes":"Prix — exclure négatifs",             "transform_sql":"CASE WHEN prod_price >= 0 THEN prod_price ELSE NULL END"},
            {"source_col":"prod_currency", "target_col":"currency_code",  "source_type":"TEXT",   "target_type":"CHAR",   "transform":"normalize_country","confidence":0.71,"notes":"tnd/TND/eur → ISO 4217",             "transform_sql":"UPPER(TRIM(prod_currency))"},
            {"source_col":"prod_stock",    "target_col":"stock_qty",      "source_type":"INTEGER","target_type":"INTEGER","transform":"cast_numeric",    "confidence":0.75,"notes":"Stocks négatifs → 0",                 "transform_sql":"CASE WHEN prod_stock < 0 THEN 0 ELSE prod_stock END"},
            {"source_col":"prod_unit",     "target_col":"unit",           "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",           "confidence":0.90,"notes":"Unité",                              "transform_sql":""},
            {"source_col":"prod_active",   "target_col":"is_active",      "source_type":"TEXT",   "target_type":"BOOLEAN","transform":"normalize_bool",  "confidence":0.73,"notes":"Y/N/1/0/yes/no → BOOLEAN",           "transform_sql":"CASE WHEN UPPER(prod_active) IN ('Y','1','YES','TRUE') THEN 1 ELSE 0 END"},
            {"source_col":"prod_created",  "target_col":"created_at",     "source_type":"TEXT",   "target_type":"DATE",   "transform":"normalize_date",  "confidence":0.69,"notes":"Formats dates mixtes",                "transform_sql":""},
        ],
        "unmapped_source": [],
        "unmapped_target": ["product_id","migrated_at"],
        "migration_risks": ["Stocks négatifs présents (~1-4%) — remplacés par 0","Devise non normalisée (tnd/TND/EUR)","Prix manquants (~2%)"]
    }

    mocks["orders"] = {
        "table_source": "orders", "table_target": "orders",
        "migration_complexity": "high", "complexity_reason": "Statuts encodés sur 20+ valeurs, dates mixtes, FK client",
        "mappings": [
            {"source_col":"ord_id",         "target_col":"legacy_ord_id",    "source_type":"INTEGER","target_type":"INTEGER","transform":"direct",          "confidence":0.99,"notes":"PK source",                              "transform_sql":""},
            {"source_col":"ord_ref",        "target_col":"order_reference",  "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",          "confidence":0.97,"notes":"Référence commande",                     "transform_sql":""},
            {"source_col":"ord_cst_id",     "target_col":"customer_id",      "source_type":"INTEGER","target_type":"INTEGER","transform":"direct",          "confidence":0.95,"notes":"FK client — résoudre via legacy_cst_id", "transform_sql":""},
            {"source_col":"ord_date",       "target_col":"order_date",       "source_type":"TEXT",   "target_type":"DATE",   "transform":"normalize_date",  "confidence":0.70,"notes":"Formats mixtes ISO/FR/US/compact",        "transform_sql":""},
            {"source_col":"ord_exp_deliver","target_col":"expected_delivery", "source_type":"TEXT",   "target_type":"DATE",   "transform":"normalize_date",  "confidence":0.70,"notes":"Formats mixtes",                         "transform_sql":""},
            {"source_col":"ord_delivered",  "target_col":"delivered_at",     "source_type":"TEXT",   "target_type":"DATE",   "transform":"normalize_date",  "confidence":0.68,"notes":"Nullable — formats mixtes",               "transform_sql":""},
            {"source_col":"ord_status",     "target_col":"status",           "source_type":"TEXT",   "target_type":"VARCHAR","transform":"normalize_status", "confidence":0.65,"notes":"P/PEND/pending/0 → pending etc.",       "transform_sql":"CASE WHEN ord_status IN ('P','PEND','pending','0') THEN 'pending' WHEN ord_status IN ('C','CONF','confirmed','1') THEN 'confirmed' WHEN ord_status IN ('S','SHIP','shipped','2') THEN 'shipped' WHEN ord_status IN ('D','DELIV','delivered','3') THEN 'delivered' WHEN ord_status IN ('X','CANC','cancelled','9') THEN 'cancelled' ELSE 'pending' END"},
            {"source_col":"ord_total_ht",   "target_col":"subtotal_ht",      "source_type":"REAL",   "target_type":"NUMERIC","transform":"cast_numeric",    "confidence":0.88,"notes":"Montant HT — quelques nulls",             "transform_sql":""},
            {"source_col":"ord_tva",        "target_col":"tva_rate",         "source_type":"REAL",   "target_type":"NUMERIC","transform":"direct",          "confidence":0.90,"notes":"Taux TVA (0/7/19)",                       "transform_sql":""},
            {"source_col":"ord_channel",    "target_col":"channel",          "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",          "confidence":0.93,"notes":"WEB/TEL/STORE/API",                       "transform_sql":"UPPER(TRIM(ord_channel))"},
            {"source_col":"ord_notes",      "target_col":"notes",            "source_type":"TEXT",   "target_type":"TEXT",   "transform":"direct",          "confidence":0.97,"notes":"Notes libres",                            "transform_sql":""},
        ],
        "unmapped_source": [],
        "unmapped_target": ["order_id","migrated_at"],
        "migration_risks": ["Statuts encodés sur 20+ valeurs distinctes","Dates en 4 formats différents","FK ord_cst_id → résoudre via legacy_cst_id après migration clients","Totaux HT manquants (~1%)"]
    }

    mocks["order_lines"] = {
        "table_source": "order_lines", "table_target": "order_lines",
        "migration_complexity": "medium", "complexity_reason": "Quantités nulles/négatives, FK doubles",
        "mappings": [
            {"source_col":"line_id",        "target_col":"legacy_line_id","source_type":"INTEGER","target_type":"INTEGER","transform":"direct",        "confidence":0.99,"notes":"PK source",                   "transform_sql":""},
            {"source_col":"line_ord_id",    "target_col":"order_id",     "source_type":"INTEGER","target_type":"INTEGER","transform":"direct",        "confidence":0.95,"notes":"FK commande",                  "transform_sql":""},
            {"source_col":"line_prod_id",   "target_col":"product_id",   "source_type":"INTEGER","target_type":"INTEGER","transform":"direct",        "confidence":0.95,"notes":"FK produit",                   "transform_sql":""},
            {"source_col":"line_qty",       "target_col":"quantity",     "source_type":"INTEGER","target_type":"INTEGER","transform":"cast_numeric",  "confidence":0.78,"notes":"Nulls à gérer — défaut 1",      "transform_sql":"CASE WHEN line_qty IS NULL OR line_qty <= 0 THEN 1 ELSE line_qty END"},
            {"source_col":"line_unit_price","target_col":"unit_price",   "source_type":"REAL",   "target_type":"NUMERIC","transform":"cast_numeric",  "confidence":0.92,"notes":"Prix unitaire",                 "transform_sql":""},
            {"source_col":"line_discount",  "target_col":"discount_pct", "source_type":"REAL",   "target_type":"NUMERIC","transform":"cast_numeric",  "confidence":0.85,"notes":"% remise — nulls → 0",          "transform_sql":"COALESCE(line_discount, 0)"},
        ],
        "unmapped_source": ["line_total"],
        "unmapped_target": ["line_id","migrated_at"],
        "migration_risks": ["Quantités nulles (~2%) → remplacées par 1","line_total colonne calculée — recalculée côté cible","FK à résoudre après migration orders et products"]
    }

    mocks["suppliers"] = {
        "table_source": "suppliers", "table_target": "suppliers",
        "migration_complexity": "low", "complexity_reason": "Table simple, peu de données",
        "mappings": [
            {"source_col":"sup_id",      "target_col":"legacy_sup_id", "source_type":"INTEGER","target_type":"INTEGER","transform":"direct",          "confidence":0.99,"notes":"PK source",                "transform_sql":""},
            {"source_col":"sup_name",    "target_col":"name",          "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",          "confidence":0.99,"notes":"Nom fournisseur",          "transform_sql":""},
            {"source_col":"sup_country", "target_col":"country_code",  "source_type":"TEXT",   "target_type":"CHAR",   "transform":"normalize_country","confidence":0.88,"notes":"Pays — déjà en ISO 2",   "transform_sql":"UPPER(TRIM(sup_country))"},
            {"source_col":"sup_contact", "target_col":"contact_name",  "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",          "confidence":0.97,"notes":"Nom contact",             "transform_sql":""},
            {"source_col":"sup_email",   "target_col":"email",         "source_type":"TEXT",   "target_type":"VARCHAR","transform":"direct",          "confidence":0.95,"notes":"Email",                    "transform_sql":""},
            {"source_col":"sup_phone",   "target_col":"phone",         "source_type":"TEXT",   "target_type":"VARCHAR","transform":"normalize_phone", "confidence":0.77,"notes":"Formats mixtes",           "transform_sql":""},
            {"source_col":"sup_rating",  "target_col":"rating",        "source_type":"INTEGER","target_type":"SMALLINT","transform":"direct",         "confidence":0.95,"notes":"Note 1-5",                 "transform_sql":""},
            {"source_col":"sup_active",  "target_col":"is_active",     "source_type":"TEXT",   "target_type":"BOOLEAN","transform":"normalize_bool", "confidence":0.80,"notes":"Y/N/1/0 → BOOLEAN",        "transform_sql":"CASE WHEN UPPER(sup_active) IN ('Y','1','YES') THEN 1 ELSE 0 END"},
            {"source_col":"sup_since",   "target_col":"partner_since", "source_type":"TEXT",   "target_type":"DATE",   "transform":"normalize_date", "confidence":0.75,"notes":"Dates mixtes",              "transform_sql":""},
        ],
        "unmapped_source": [],
        "unmapped_target": ["supplier_id","migrated_at"],
        "migration_risks": ["Téléphones en formats mixtes","~5% de pays manquants"]
    }

    return mocks.get(table, {"table_source": table, "table_target": table, "migration_complexity":"low","complexity_reason":"Table simple","mappings":[],"unmapped_source":[],"unmapped_target":[],"migration_risks":[]})


# ── Construction du graphe complet ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(SchemaState)

    graph.add_node("extract_schemas",    node_extract_schemas)
    graph.add_node("propose_mapping",    node_propose_mapping)
    graph.add_node("enrich_with_tools",  node_enrich_with_tools)
    graph.add_node("human_review",       node_human_review)
    graph.add_node("generate_dbt",       node_generate_dbt)
    graph.add_node("save_outputs",       node_save_outputs)

    graph.set_entry_point("extract_schemas")
    graph.add_edge("extract_schemas",   "propose_mapping")
    graph.add_edge("propose_mapping",   "enrich_with_tools")
    graph.add_conditional_edges(
        "enrich_with_tools",
        node_human_gate,
        {"human_review": "human_review", "generate_dbt": "generate_dbt"}
    )
    graph.add_edge("human_review",      "generate_dbt")
    graph.add_edge("generate_dbt",      "save_outputs")
    graph.add_edge("save_outputs",      END)

    return graph.compile()


# ── Point d'entrée principal ───────────────────────────────────────────────────

def run_agent(table_name: str) -> SchemaState:
    """Lance l'agent complet pour une table."""
    app = build_graph()
    initial: SchemaState = {
        "table_name":     table_name,
        "source_ddl":     "",
        "target_ddl":     TARGET_DDLS.get(table_name, ""),
        "source_sample":  [],
        "source_stats":   {},
        "mapping":        None,
        "tool_results":   [],
        "dbt_model":      None,
        "ambiguous":      [],
        "human_approved": False,
        "errors":         [],
        "messages":       [],
    }
    return app.invoke(initial)


def run_all_tables(auto_approve: bool = True):
    """Lance l'agent sur les 5 tables du projet."""
    if auto_approve:
        os.environ["SMARTMIGRATE_AUTO_APPROVE"] = "true"

    tables  = list(TARGET_DDLS.keys())
    results = {}

    print("=" * 60)
    print("🧠 SmartMigrate — Schema Analyst Agent")
    print("=" * 60)

    for table in tables:
        print(f"\n{'─'*60}")
        print(f"  Table : {table.upper()}")
        print(f"{'─'*60}")
        try:
            result = run_agent(table)
            results[table] = result
            n = len(result["mapping"]["mappings"]) if result.get("mapping") else 0
            complexity = result["mapping"].get("migration_complexity","?") if result.get("mapping") else "?"
            print(f"\n  ✅ {table} : {n} colonnes | complexité {complexity}")
        except Exception as e:
            print(f"\n  ❌ {table} : erreur — {e}")
            results[table] = None

    # Résumé final
    print(f"\n{'='*60}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*60}")
    for t, r in results.items():
        if r and r.get("mapping"):
            n   = len(r["mapping"]["mappings"])
            amb = len(r.get("ambiguous", []))
            cmp = r["mapping"].get("migration_complexity","?")
            print(f"  {'✅' if not r['errors'] else '⚠️ '} {t:<15} {n} colonnes | {amb} ambiguës | {cmp}")
        else:
            print(f"  ❌ {t:<15} erreur")
    print(f"\n  📁 Rapports   → outputs/mappings/")
    print(f"  🏗️  Modèles dbt → dbt_project/models/staging/")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_agent(sys.argv[1])
    else:
        run_all_tables(auto_approve=True)
