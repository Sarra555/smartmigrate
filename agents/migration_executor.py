"""
SmartMigrate — Agent 3 : Migration Executor
=============================================
Le grand final : exécute vraiment la migration ERP → Cloud.

Ce que fait cet agent :
1. [preflight_check]     Vérifie les outputs des Agents 1 et 2 avant de démarrer
2. [transform_data]      Applique toutes les transformations du mapping (dates, statuts, bools...)
3. [load_to_target]      Charge les données transformées en base cible (SQLite cloud simulé)
4. [reconcile]           Compare source vs destination (counts, distributions, samples)
5. [detect_drift]        Détecte les anomalies post-migration (données corrompues silencieuses)
6. [generate_narrative]  Rapport exécutif LLM en langage naturel pour les stakeholders
7. [generate_report]     Dashboard HTML final avec toutes les métriques

Stack : LangGraph + Pandas + SQLite (target simulé) + Gemini/MOCK
"""

import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, StateGraph
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False

from dotenv import load_dotenv
load_dotenv()

# ── Chemins ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB    = PROJECT_ROOT / "data" / "raw" / "erp_legacy.db"
TARGET_DB    = PROJECT_ROOT / "data" / "raw" / "cloud_target.db"   # SQLite simule le cloud
OUTPUT_MIG   = PROJECT_ROOT / "outputs" / "migration"
OUTPUT_MIG.mkdir(parents=True, exist_ok=True)

MAPPINGS_DIR = PROJECT_ROOT / "outputs" / "mappings"
QUALITY_DIR  = PROJECT_ROOT / "outputs" / "quality"

RUN_ID = os.getenv("MIGRATION_RUN_ID", f"run_{datetime.now().strftime('%Y%m%d_%H%M')}")

TABLES_ORDER = ["customers", "products", "suppliers", "orders", "order_lines"]

# ── State ──────────────────────────────────────────────────────────────────────
class MigrationState(TypedDict):
    table_name:       str
    source_df:        Optional[Any]
    transformed_df:   Optional[Any]
    mapping:          Optional[Dict]
    quality_report:   Optional[Dict]
    preflight_ok:     bool
    load_stats:       Dict
    reconciliation:   Dict
    drift_issues:     List[Dict]
    migration_status: str            # "success" | "partial" | "failed"
    narrative:        str
    errors:           List[str]

# ── Helpers transformation ─────────────────────────────────────────────────────

def normalize_date(val: Any) -> Optional[str]:
    """Convertit tous les formats de dates legacy en ISO YYYY-MM-DD."""
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    patterns = [
        (r"^(\d{4})-(\d{2})-(\d{2})$",    lambda m: f"{m[0]}-{m[1]}-{m[2]}"),
        (r"^(\d{2})/(\d{2})/(\d{4})$",    lambda m: f"{m[2]}-{m[1]}-{m[0]}"),
        (r"^(\d{2})-(\d{2})-(\d{4})$",    lambda m: f"{m[2]}-{m[0]}-{m[1]}"),
        (r"^(\d{4})(\d{2})(\d{2})$",       lambda m: f"{m[0]}-{m[1]}-{m[2]}"),
    ]
    for pattern, formatter in patterns:
        m = re.match(pattern, s)
        if m:
            try:
                result = formatter(m.groups())
                datetime.strptime(result, "%Y-%m-%d")
                return result
            except ValueError:
                continue
    return None


def normalize_status_customer(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "active"
    v = str(val).strip().lower()
    if v in ("1", "a", "active", "yes", "y"):
        return "active"
    return "inactive"


def normalize_status_order(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "pending"
    v = str(val).strip().upper()
    mapping = {
        "P": "pending",   "PEND": "pending",   "PENDING": "pending",   "0": "pending",
        "C": "confirmed", "CONF": "confirmed",  "CONFIRMED": "confirmed","1": "confirmed",
        "S": "shipped",   "SHIP": "shipped",    "SHIPPED": "shipped",    "2": "shipped",
        "D": "delivered", "DELIV": "delivered", "DELIVERED": "delivered","3": "delivered",
        "X": "cancelled", "CANC": "cancelled",  "CANCELLED": "cancelled","9": "cancelled",
    }
    return mapping.get(v, "pending")


def normalize_bool(val: Any) -> int:
    if pd.isna(val) or val is None:
        return 0
    v = str(val).strip().upper()
    return 1 if v in ("1", "Y", "YES", "TRUE", "A", "ACTIVE") else 0


def normalize_country(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "TN"
    mapping = {
        "tunisia": "TN", "tunisie": "TN", "tn": "TN",
        "france": "FR",  "fr": "FR",
        "algerie": "DZ", "algeria": "DZ", "dz": "DZ",
        "maroc": "MA",   "morocco": "MA", "ma": "MA",
        "allemagne": "DE","germany": "DE", "de": "DE",
        "us": "US",      "usa": "US",     "united states": "US",
        "cn": "CN",      "china": "CN",
        "it": "IT",      "italy": "IT",   "italie": "IT",
        "es": "ES",      "spain": "ES",   "espagne": "ES",
        "tr": "TR",      "turkey": "TR",  "turquie": "TR",
    }
    return mapping.get(val.strip().lower(), val.upper()[:2])


def normalize_currency(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "TND"
    return str(val).strip().upper()[:3]


def normalize_phone(val: Any) -> Optional[str]:
    if pd.isna(val) or val is None:
        return None
    digits = re.sub(r"[^\d+]", "", str(val))
    if digits.startswith("00216"):
        digits = "+" + digits[2:]
    elif digits.startswith("216") and len(digits) >= 11:
        digits = "+" + digits
    elif not digits.startswith("+") and len(digits) == 8:
        digits = "+216" + digits
    return digits if len(digits) >= 8 else None

# ── Noeud 1 : Preflight check ─────────────────────────────────────────────────

def node_preflight(state: MigrationState) -> MigrationState:
    table = state["table_name"]
    print(f"\n  ✈️  Preflight check : {table}")

    errors = []

    # Charger le mapping Agent 1
    mapping_path = MAPPINGS_DIR / f"{table}_mapping.json"
    if not mapping_path.exists():
        errors.append(f"Mapping manquant : {mapping_path}")
        mapping = None
    else:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        print(f"     ✓ Mapping trouvé : {len(mapping.get('mappings', []))} colonnes")

    # Charger le rapport qualité Agent 2
    quality_path = QUALITY_DIR / f"{table}_quality.json"
    if not quality_path.exists():
        print(f"     ⚠️  Rapport qualité absent — migration sans gate qualité")
        quality = None
    else:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        score   = quality.get("score", 0)
        print(f"     ✓ Qualité : score={score:.0%} | approved={quality.get('approved')}")

    # Charger les données source
    try:
        conn = sqlite3.connect(str(SOURCE_DB))
        df   = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        conn.close()
        print(f"     ✓ Source : {len(df)} lignes · {len(df.columns)} colonnes")
    except Exception as e:
        errors.append(f"Source DB inaccessible : {e}")
        df = None

    preflight_ok = len(errors) == 0 and df is not None

    return {
        **state,
        "source_df":      df,
        "mapping":        mapping,
        "quality_report": quality,
        "preflight_ok":   preflight_ok,
        "errors":         state["errors"] + errors
    }


def route_preflight(state: MigrationState) -> Literal["ok", "failed"]:
    return "ok" if state["preflight_ok"] else "failed"

# ── Noeud 2 : Transformation ───────────────────────────────────────────────────

TRANSFORMERS = {
    "customers": lambda df: (
        df.assign(
            legacy_cst_id  = df["cst_id"],
            first_name     = df["cst_fname"],
            last_name      = df["cst_lname"],
            company_name   = df["cst_company"],
            email          = df["cst_email"].str.lower().str.strip(),
            phone          = df["cst_phone"].apply(normalize_phone),
            country_code   = df["cst_country"].apply(normalize_country),
            city           = df["cst_city"],
            address        = df["cst_address"],
            segment        = df["cst_segment"].where(df["cst_segment"].isin(["B2B","B2C","VIP"])),
            status         = df["cst_status"].apply(normalize_status_customer),
            is_company     = df["cst_company"].notna().astype(int),
            created_at     = df["cst_created_dt"].apply(normalize_date),
            notes          = df["cst_notes"],
        )[[
            "legacy_cst_id","first_name","last_name","company_name",
            "email","phone","country_code","city","address",
            "segment","status","is_company","created_at","notes"
        ]]
        .drop_duplicates(subset=["email"])
        .dropna(subset=["legacy_cst_id"])
    ),

    "products": lambda df: (
        df.assign(
            legacy_prod_id = df["prod_id"],
            reference      = df["prod_ref"],
            name           = df["prod_name"],
            category       = df["prod_category"],
            unit_price     = pd.to_numeric(df["prod_price"], errors="coerce").clip(lower=0),
            currency_code  = df["prod_currency"].apply(normalize_currency),
            stock_qty      = pd.to_numeric(df["prod_stock"], errors="coerce").fillna(0).clip(lower=0).astype(int),
            unit           = df["prod_unit"],
            is_active      = df["prod_active"].apply(normalize_bool),
            created_at     = df["prod_created"].apply(normalize_date),
        )[[
            "legacy_prod_id","reference","name","category",
            "unit_price","currency_code","stock_qty","unit","is_active","created_at"
        ]]
        .dropna(subset=["legacy_prod_id","reference","name"])
    ),

    "suppliers": lambda df: (
        df.assign(
            legacy_sup_id  = df["sup_id"],
            name           = df["sup_name"],
            country_code   = df["sup_country"].apply(normalize_country),
            contact_name   = df["sup_contact"],
            email          = df["sup_email"],
            phone          = df["sup_phone"].apply(normalize_phone),
            rating         = pd.to_numeric(df["sup_rating"], errors="coerce").fillna(0).clip(lower=0, upper=5),
            is_active      = df["sup_active"].apply(normalize_bool),
            partner_since  = df["sup_since"].apply(normalize_date),
        )[[
            "legacy_sup_id","name","country_code","contact_name",
            "email","phone","rating","is_active","partner_since"
        ]]
        .dropna(subset=["legacy_sup_id","name"])
    ),

    "orders": lambda df: (
        df.assign(
            legacy_ord_id    = df["ord_id"],
            order_reference  = df["ord_ref"],
            customer_id      = df["ord_cst_id"],
            order_date       = df["ord_date"].apply(normalize_date),
            expected_delivery= df["ord_exp_deliver"].apply(normalize_date),
            delivered_at     = df["ord_delivered"].apply(normalize_date),
            status           = df["ord_status"].apply(normalize_status_order),
            subtotal_ht      = pd.to_numeric(df["ord_total_ht"], errors="coerce").clip(lower=0),
            tva_rate         = pd.to_numeric(df["ord_tva"], errors="coerce").fillna(19.0),
            channel          = df["ord_channel"].str.upper().str.strip().where(
                                   df["ord_channel"].str.upper().str.strip().isin(["WEB","TEL","STORE","API"])
                               ),
            notes            = df["ord_notes"],
        )[[
            "legacy_ord_id","order_reference","customer_id",
            "order_date","expected_delivery","delivered_at",
            "status","subtotal_ht","tva_rate","channel","notes"
        ]]
        .dropna(subset=["legacy_ord_id","order_reference","order_date"])
    ),

    "order_lines": lambda df: (
        df.assign(
            legacy_line_id = df["line_id"],
            order_id       = df["line_ord_id"],
            product_id     = df["line_prod_id"],
            quantity       = pd.to_numeric(df["line_qty"], errors="coerce").fillna(1).clip(lower=1).astype(int),
            unit_price     = pd.to_numeric(df["line_unit_price"], errors="coerce").clip(lower=0),
            discount_pct   = pd.to_numeric(df["line_discount"], errors="coerce").fillna(0).clip(0,100),
        )[[
            "legacy_line_id","order_id","product_id",
            "quantity","unit_price","discount_pct"
        ]]
        .dropna(subset=["legacy_line_id","order_id","product_id","unit_price"])
    ),
}


def node_transform(state: MigrationState) -> MigrationState:
    table  = state["table_name"]
    df_src = state["source_df"]
    print(f"  🔄 Transformation des données...")

    try:
        transformer = TRANSFORMERS.get(table)
        if transformer is None:
            raise ValueError(f"Pas de transformer pour '{table}'")

        df_transformed = transformer(df_src)

        n_src  = len(df_src)
        n_dst  = len(df_transformed)
        n_drop = n_src - n_dst
        print(f"     {n_src} → {n_dst} lignes ({n_drop} filtrées/dédupliquées)")

        return {**state, "transformed_df": df_transformed}

    except Exception as e:
        return {**state, "errors": state["errors"] + [f"transform: {str(e)}"]}

# ── Noeud 3 : Load vers la cible ──────────────────────────────────────────────

def node_load(state: MigrationState) -> MigrationState:
    table = state["table_name"]
    df    = state["transformed_df"]
    print(f"  📥 Chargement vers la base cible...")

    if df is None or len(df) == 0:
        return {**state, "errors": state["errors"] + ["load: DataFrame vide"], "load_stats": {}}

    try:
        conn = sqlite3.connect(str(TARGET_DB))

        # Écriture avec replace
        df.to_sql(table, conn, if_exists="replace", index=False)

        # Vérification immédiate
        n_loaded = pd.read_sql_query(f'SELECT COUNT(*) as n FROM "{table}"', conn)["n"][0]

        # Log de migration
        log_df = pd.DataFrame([{
            "migration_run": RUN_ID,
            "table_name":    table,
            "source_count":  len(state["source_df"]),
            "target_count":  n_loaded,
            "status":        "success",
            "migrated_at":   datetime.now().isoformat()
        }])
        log_df.to_sql("migration_log", conn, if_exists="append", index=False)
        conn.close()

        stats = {
            "source_count":  len(state["source_df"]),
            "target_count":  int(n_loaded),
            "filtered":      len(state["source_df"]) - int(n_loaded),
            "load_rate":     round(int(n_loaded) / len(state["source_df"]), 4)
        }
        print(f"     ✅ {n_loaded} lignes chargées ({stats['load_rate']:.1%} du source)")
        return {**state, "load_stats": stats}

    except Exception as e:
        return {**state, "errors": state["errors"] + [f"load: {str(e)}"], "load_stats": {}}

# ── Noeud 4 : Réconciliation ───────────────────────────────────────────────────

def node_reconcile(state: MigrationState) -> MigrationState:
    table = state["table_name"]
    print(f"  🔍 Réconciliation source ↔ destination...")

    try:
        # Source
        src_conn = sqlite3.connect(str(SOURCE_DB))
        df_src   = pd.read_sql_query(f'SELECT * FROM "{table}"', src_conn)
        src_conn.close()

        # Target
        tgt_conn = sqlite3.connect(str(TARGET_DB))
        df_tgt   = pd.read_sql_query(f'SELECT * FROM "{table}"', tgt_conn)
        tgt_conn.close()

        recon = {
            "source_count":    len(df_src),
            "target_count":    len(df_tgt),
            "count_match":     len(df_src) >= len(df_tgt),
            "count_diff":      len(df_src) - len(df_tgt),
            "column_checks":   {},
            "sample_check":    {}
        }

        # Comparaison des colonnes numériques
        for col in df_tgt.columns:
            if pd.api.types.is_numeric_dtype(df_tgt[col]):
                tgt_vals = df_tgt[col].dropna()
                if len(tgt_vals) > 0:
                    recon["column_checks"][col] = {
                        "target_min":  float(tgt_vals.min()),
                        "target_max":  float(tgt_vals.max()),
                        "target_mean": round(float(tgt_vals.mean()), 2),
                        "null_rate":   round(df_tgt[col].isna().sum() / len(df_tgt), 4)
                    }

        # Vérification des valeurs uniques sur colonnes clés
        for col in df_tgt.columns:
            if any(kw in col.lower() for kw in ["id", "ref", "email"]):
                n_dup = int(df_tgt[col].dropna().duplicated().sum())
                recon["sample_check"][col] = {
                    "unique_ok": n_dup == 0,
                    "duplicates": n_dup
                }

        # Status global
        issues = sum(1 for v in recon["sample_check"].values() if not v["unique_ok"])
        recon["overall_ok"] = (recon["count_diff"] <= len(df_src) * 0.05) and issues == 0

        print(f"     Source: {recon['source_count']} | Target: {recon['target_count']} | Diff: {recon['count_diff']}")
        print(f"     Réconciliation: {'✅ OK' if recon['overall_ok'] else '⚠️ Anomalies détectées'}")

        return {**state, "reconciliation": recon}

    except Exception as e:
        return {**state, "errors": state["errors"] + [f"reconcile: {str(e)}"], "reconciliation": {}}

# ── Noeud 5 : Détection de drift ──────────────────────────────────────────────

def node_detect_drift(state: MigrationState) -> MigrationState:
    table   = state["table_name"]
    df_tgt  = state["transformed_df"]
    print(f"  📡 Détection de drift post-migration...")

    drift_issues = []

    if df_tgt is None:
        return {**state, "drift_issues": drift_issues}

    # 1. Dates invalides après transformation
    for col in df_tgt.columns:
        if "date" in col or "at" in col.split("_")[-1:]:
            if df_tgt[col].dtype == object:
                n_null = df_tgt[col].isna().sum()
                null_rate = n_null / len(df_tgt)
                if null_rate > 0.10:
                    drift_issues.append({
                        "type":    "DATE_CONVERSION_LOSS",
                        "column":  col,
                        "detail":  f"{null_rate*100:.1f}% des dates n'ont pas pu être converties",
                        "severity":"MEDIUM"
                    })

    # 2. Valeurs hors domaine après normalisation
    if "status" in df_tgt.columns:
        valid_statuses = {
            "customers":   {"active", "inactive"},
            "orders":      {"pending","confirmed","shipped","delivered","cancelled"},
        }
        expected = valid_statuses.get(table, set())
        if expected:
            invalid = ~df_tgt["status"].isin(expected)
            n_inv   = int(invalid.sum())
            if n_inv > 0:
                drift_issues.append({
                    "type":    "INVALID_STATUS_POST_TRANSFORM",
                    "column":  "status",
                    "detail":  f"{n_inv} lignes avec statut non reconnu après transformation",
                    "severity":"HIGH"
                })

    # 3. Prix négatifs résiduels
    for col in ["unit_price", "subtotal_ht", "discount_pct"]:
        if col in df_tgt.columns:
            n_neg = int((pd.to_numeric(df_tgt[col], errors="coerce") < 0).sum())
            if n_neg > 0:
                drift_issues.append({
                    "type":    "NEGATIVE_VALUE_POST_TRANSFORM",
                    "column":  col,
                    "detail":  f"{n_neg} valeurs négatives après transformation",
                    "severity":"HIGH"
                })

    # 4. Taux de rétention
    n_src = state["load_stats"].get("source_count", 1)
    n_tgt = state["load_stats"].get("target_count", 0)
    retention = n_tgt / n_src if n_src > 0 else 0
    if retention < 0.90:
        drift_issues.append({
            "type":    "LOW_RETENTION_RATE",
            "column":  "ALL",
            "detail":  f"Seulement {retention:.1%} des lignes source migrées ({n_src - n_tgt} perdues)",
            "severity":"HIGH" if retention < 0.80 else "MEDIUM"
        })

    status = "failed" if any(d["severity"]=="HIGH" for d in drift_issues) else \
             "partial" if drift_issues else "success"

    print(f"     {len(drift_issues)} problèmes de drift | Statut: {status}")
    return {**state, "drift_issues": drift_issues, "migration_status": status}

# ── Noeud 6 : Narratif LLM ────────────────────────────────────────────────────

def node_narrative(state: MigrationState) -> MigrationState:
    table  = state["table_name"]
    stats  = state["load_stats"]
    recon  = state["reconciliation"]
    drift  = state["drift_issues"]
    status = state["migration_status"]

    # Mode MOCK
    if not LANGCHAIN_OK or not os.getenv("GOOGLE_API_KEY"):
        retention = stats.get("load_rate", 0)
        status_fr = {"success":"réussie ✅","partial":"partielle ⚠️","failed":"échouée ❌"}.get(status,"?")
        narrative = (
            f"**Migration de la table `{table}` — {status_fr}**\n\n"
            f"**{stats.get('target_count',0):,}** enregistrements migrés sur "
            f"**{stats.get('source_count',0):,}** (taux de rétention : **{retention:.1%}**).\n\n"
        )
        if drift:
            narrative += "**Points d'attention :**\n"
            for d in drift:
                narrative += f"- {d['column']} : {d['detail']}\n"
        else:
            narrative += "Aucun problème de drift détecté. Les données sont conformes en base cible."

        if stats.get("filtered",0) > 0:
            narrative += f"\n\n**{stats['filtered']} lignes filtrées** lors de la transformation (doublons éliminés, valeurs invalides écartées)."

        return {**state, "narrative": narrative}

    # Vrai LLM
    system = """Tu es expert en migration de données. Tu écris des rapports exécutifs 
clairs et précis pour des managers non-techniques. Sois concis (6-8 lignes), 
factuel et actionnable. Utilise des chiffres précis."""

    user_msg = f"""Table migrée : {table}
Statut : {status}
Lignes source : {stats.get('source_count',0):,}
Lignes migrées : {stats.get('target_count',0):,}
Taux rétention : {stats.get('load_rate',0):.1%}
Problèmes drift : {json.dumps(drift[:3], ensure_ascii=False)}
Réconciliation OK : {recon.get('overall_ok', False)}

Écris le résumé exécutif en français."""

    try:
        llm  = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,
                                       google_api_key=os.getenv("GOOGLE_API_KEY"))
        resp = None
        for attempt in range(3):
            try:
                resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 60 * (attempt + 1)
                    print(f"  ⏳ quota — attente {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        narrative = resp.content.strip() if resp else "Rapport non disponible."
    except Exception as e:
        narrative = f"[LLM indisponible : {str(e)[:60]}]"

    return {**state, "narrative": narrative}

# ── Noeud 7 : Rapport HTML ────────────────────────────────────────────────────

def node_report(state: MigrationState) -> MigrationState:
    table     = state["table_name"]
    stats     = state["load_stats"]
    recon     = state["reconciliation"]
    drift     = state["drift_issues"]
    status    = state["migration_status"]
    narrative = state["narrative"]
    quality   = state["quality_report"] or {}

    status_color = {"success":"#16a34a","partial":"#d97706","failed":"#dc2626"}.get(status,"#64748b")
    status_icon  = {"success":"✅","partial":"⚠️","failed":"❌"}.get(status,"?")
    status_bg    = {"success":"#dcfce7","partial":"#fef3c7","failed":"#fee2e2"}.get(status,"#f1f5f9")

    retention  = stats.get("load_rate", 0)
    ret_color  = "#16a34a" if retention >= 0.95 else "#d97706" if retention >= 0.85 else "#dc2626"

    # Drift HTML
    drift_html = ""
    for d in drift:
        sev_color = {"HIGH":"#dc2626","MEDIUM":"#d97706"}.get(d["severity"],"#64748b")
        drift_html += f"""<tr>
          <td><span style="color:{sev_color};font-weight:600">{d['severity']}</span></td>
          <td><code>{d['column']}</code></td>
          <td>{d['type']}</td>
          <td style="font-size:12px">{d['detail']}</td>
        </tr>"""

    # Recon HTML
    recon_html = ""
    for col, info in (recon.get("sample_check") or {}).items():
        icon = "✅" if info["unique_ok"] else f"❌ {info['duplicates']} doublons"
        recon_html += f"<tr><td><code>{col}</code></td><td>Unicité</td><td>{icon}</td></tr>"

    narrative_html = narrative.replace("**","<b>").replace("\n","<br>")

    # Quality score badge
    q_score = quality.get("score", None)
    q_badge = f'<span style="background:#f1f5f9;padding:4px 10px;border-radius:6px;font-size:13px">Qualité pré-migration : <b>{q_score:.0%}</b></span>' if q_score else ""

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Migration Report — {table}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f8fafc; color:#0f172a; padding:28px; }}
  h1 {{ font-size:22px; font-weight:700; margin-bottom:4px; }}
  h2 {{ font-size:14px; font-weight:600; margin-bottom:12px; color:#374151; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:20px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px; }}
  .card {{ background:white; border:1px solid #e2e8f0; border-radius:12px; padding:20px 24px; }}
  .stat {{ text-align:center; }}
  .stat-num {{ font-size:32px; font-weight:700; }}
  .stat-lbl {{ font-size:12px; color:#64748b; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th {{ background:#f8fafc; padding:8px 12px; text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; border-bottom:1px solid #e2e8f0; }}
  td {{ padding:8px 12px; border-bottom:1px solid #f8fafc; vertical-align:top; }}
  code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:11px; }}
  .narrative {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:18px 22px; font-size:13px; line-height:1.8; margin-bottom:20px; }}
  .bar-wrap {{ background:#f1f5f9; border-radius:99px; height:8px; margin-top:6px; }}
  .bar-fill  {{ height:8px; border-radius:99px; }}
</style></head><body>

<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:12px">
  <div>
    <h1>🚀 Migration Report — <code>{table}</code></h1>
    <p style="font-size:13px;color:#64748b">Run: {RUN_ID} · {datetime.now().strftime('%Y-%m-%d %H:%M')} · SmartMigrate</p>
  </div>
  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    {q_badge}
    <div style="background:{status_bg};padding:10px 18px;border-radius:10px;font-size:15px;font-weight:600;color:{status_color}">
      {status_icon} Migration {status.upper()}
    </div>
  </div>
</div>

<div class="grid3">
  <div class="card stat">
    <div class="stat-num">{stats.get('source_count',0):,}</div>
    <div class="stat-lbl">Lignes source (ERP)</div>
  </div>
  <div class="card stat">
    <div class="stat-num" style="color:{ret_color}">{stats.get('target_count',0):,}</div>
    <div class="stat-lbl">Lignes migrées (Cloud)</div>
  </div>
  <div class="card stat">
    <div class="stat-num" style="color:{ret_color}">{retention:.1%}</div>
    <div class="stat-lbl">Taux de rétention</div>
    <div class="bar-wrap"><div class="bar-fill" style="background:{ret_color};width:{min(retention*100,100):.0f}%"></div></div>
  </div>
</div>

<div class="narrative">
  <strong>📝 Rapport exécutif</strong><br><br>
  {narrative_html}
</div>

<div class="grid2">
  <div class="card">
    <h2>🔍 Réconciliation Source ↔ Target</h2>
    <table>
      <tr>
        <td>Lignes source</td>
        <td><strong>{recon.get('source_count',0):,}</strong></td>
      </tr>
      <tr>
        <td>Lignes target</td>
        <td><strong>{recon.get('target_count',0):,}</strong></td>
      </tr>
      <tr>
        <td>Différence</td>
        <td style="color:{'#16a34a' if recon.get('count_diff',0)<=0 else '#d97706'}">
          {recon.get('count_diff',0):+,} lignes
        </td>
      </tr>
      <tr>
        <td>Lignes filtrées</td>
        <td>{stats.get('filtered',0)} (doublons/invalides)</td>
      </tr>
      <tr>
        <td>Statut global</td>
        <td>{'✅ OK' if recon.get('overall_ok') else '⚠️ Vérifier'}</td>
      </tr>
    </table>
    {'<br><table><tr><th>Colonne</th><th>Règle</th><th>Résultat</th></tr>' + recon_html + '</table>' if recon_html else ''}
  </div>

  <div class="card">
    <h2>📡 Drift Post-Migration</h2>
    {f'''<table>
      <tr><th>Sévérité</th><th>Colonne</th><th>Type</th><th>Détail</th></tr>
      {drift_html}
    </table>''' if drift_html else '<p style="color:#16a34a;padding:20px;text-align:center">✅ Aucun drift détecté</p>'}
  </div>
</div>

</body></html>"""

    # Sauvegarde
    html_path = OUTPUT_MIG / f"{table}_migration_report.html"
    html_path.write_text(html, encoding="utf-8")

    # JSON log
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            return super().default(obj)

    log = {
        "run_id": RUN_ID, "table": table, "status": status,
        "load_stats": stats, "reconciliation": {k:v for k,v in recon.items() if k != "samples"},
        "drift_issues": drift, "errors": state["errors"],
        "timestamp": datetime.now().isoformat()
    }
    (OUTPUT_MIG / f"{table}_migration_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2, cls=NpEncoder), encoding="utf-8"
    )

    print(f"  💾 Rapport : outputs/migration/{table}_migration_report.html")
    return state


def node_preflight_failed(state: MigrationState) -> MigrationState:
    print(f"  ❌ Preflight échoué : {state['errors']}")
    return {**state, "migration_status": "failed"}

# ── Graph ──────────────────────────────────────────────────────────────────────

def build_migration_graph():
    graph = StateGraph(MigrationState)

    graph.add_node("preflight",         node_preflight)
    graph.add_node("preflight_failed",  node_preflight_failed)
    graph.add_node("transform",         node_transform)
    graph.add_node("load",              node_load)
    graph.add_node("reconcile",         node_reconcile)
    graph.add_node("detect_drift",      node_detect_drift)
    graph.add_node("narrative",         node_narrative)
    graph.add_node("report",            node_report)

    graph.set_entry_point("preflight")
    graph.add_conditional_edges("preflight", route_preflight,
                                {"ok": "transform", "failed": "preflight_failed"})
    graph.add_edge("preflight_failed", END)
    graph.add_edge("transform",   "load")
    graph.add_edge("load",        "reconcile")
    graph.add_edge("reconcile",   "detect_drift")
    graph.add_edge("detect_drift","narrative")
    graph.add_edge("narrative",   "report")
    graph.add_edge("report",      END)

    return graph.compile()

# ── Runner + Rapport final global ─────────────────────────────────────────────

def generate_final_dashboard(summary: Dict):
    """Dashboard HTML global — toutes les tables d'un coup."""
    rows_html = ""
    total_src = total_tgt = 0

    for table, s in summary.items():
        status     = s.get("status","?")
        icon       = {"success":"✅","partial":"⚠️","failed":"❌"}.get(status,"?")
        color      = {"success":"#16a34a","partial":"#d97706","failed":"#dc2626"}.get(status,"#64748b")
        stats      = s.get("load_stats", {})
        src        = stats.get("source_count", 0)
        tgt        = stats.get("target_count", 0)
        ret        = stats.get("load_rate", 0)
        drift_n    = len(s.get("drift_issues",[]))
        total_src += src
        total_tgt += tgt

        rows_html += f"""<tr>
          <td><code>{table}</code></td>
          <td><span style="color:{color};font-weight:600">{icon} {status.upper()}</span></td>
          <td>{src:,}</td><td>{tgt:,}</td>
          <td style="color:{'#16a34a' if ret>=0.95 else '#d97706'}">{ret:.1%}</td>
          <td>{'<span style="color:#dc2626">'+str(drift_n)+'</span>' if drift_n else '✅ 0'}</td>
          <td><a href="{table}_migration_report.html" style="color:#2563eb">Voir →</a></td>
        </tr>"""

    global_retention = total_tgt / total_src if total_src > 0 else 0
    n_success = sum(1 for s in summary.values() if s.get("status")=="success")

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"><title>SmartMigrate — Dashboard Final</title>
<style>
  * {{ box-sizing:border-box;margin:0;padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#0f172a;padding:28px; }}
  .header {{ background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;border-radius:16px;padding:32px 36px;margin-bottom:24px; }}
  .header h1 {{ font-size:24px;font-weight:700;margin-bottom:6px; }}
  .header p  {{ font-size:13px;opacity:0.8; }}
  .stats {{ display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px; }}
  .stat {{ background:rgba(255,255,255,0.12);border-radius:10px;padding:16px; }}
  .stat-n {{ font-size:28px;font-weight:700; }}
  .stat-l {{ font-size:12px;opacity:0.75;margin-top:2px; }}
  .card {{ background:white;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin-bottom:20px; }}
  table {{ width:100%;border-collapse:collapse;font-size:13px; }}
  th {{ background:#f8fafc;padding:10px 14px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;border-bottom:1px solid #e2e8f0; }}
  td {{ padding:10px 14px;border-bottom:1px solid #f8fafc; }}
  code {{ background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:12px; }}
  .footer {{ text-align:center;font-size:12px;color:#94a3b8;margin-top:20px; }}
</style></head><body>

<div class="header">
  <h1>🚀 SmartMigrate — Dashboard Migration Final</h1>
  <p>Run : {RUN_ID} · ERP Legacy → Cloud Data Warehouse · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  <div class="stats">
    <div class="stat"><div class="stat-n">{n_success}/{len(summary)}</div><div class="stat-l">Tables réussies</div></div>
    <div class="stat"><div class="stat-n">{total_src:,}</div><div class="stat-l">Lignes source</div></div>
    <div class="stat"><div class="stat-n">{total_tgt:,}</div><div class="stat-l">Lignes migrées</div></div>
    <div class="stat"><div class="stat-n">{global_retention:.1%}</div><div class="stat-l">Rétention globale</div></div>
  </div>
</div>

<div class="card">
  <table>
    <tr><th>Table</th><th>Statut</th><th>Source</th><th>Migré</th><th>Rétention</th><th>Drift</th><th>Rapport</th></tr>
    {rows_html}
  </table>
</div>

<div class="footer">SmartMigrate · Sarra Aouadi · Data Migration & AI Engineer</div>
</body></html>"""

    path = OUTPUT_MIG / "dashboard_final.html"
    path.write_text(html, encoding="utf-8")
    print(f"\n  🏁 Dashboard final : outputs/migration/dashboard_final.html")


def run_migration_executor(tables: List[str] = None) -> Dict:
    if tables is None:
        tables = TABLES_ORDER

    print("\n" + "="*60)
    print("  🚀  SmartMigrate — Migration Executor Agent")
    print(f"  Run ID : {RUN_ID}")
    print("="*60)

    app     = build_migration_graph()
    summary = {}

    for table in tables:
        print(f"\n{'─'*60}")
        print(f"  Table : {table.upper()}")
        print(f"{'─'*60}")

        initial: MigrationState = {
            "table_name":       table,
            "source_df":        None,
            "transformed_df":   None,
            "mapping":          None,
            "quality_report":   None,
            "preflight_ok":     False,
            "load_stats":       {},
            "reconciliation":   {},
            "drift_issues":     [],
            "migration_status": "failed",
            "narrative":        "",
            "errors":           []
        }

        try:
            result = app.invoke(initial)
            summary[table] = {
                "status":       result["migration_status"],
                "load_stats":   result["load_stats"],
                "drift_issues": result["drift_issues"],
                "errors":       result["errors"]
            }
        except Exception as e:
            print(f"  ❌ Erreur fatale : {e}")
            summary[table] = {"status":"failed","load_stats":{},"drift_issues":[],"errors":[str(e)]}

    # Résumé final
    print(f"\n{'='*60}")
    print("  RÉSUMÉ FINAL — MIGRATION COMPLÈTE")
    print(f"{'='*60}")

    for table, s in summary.items():
        icon  = {"success":"✅","partial":"⚠️","failed":"❌"}.get(s["status"],"?")
        stats = s.get("load_stats",{})
        print(f"  {icon} {table:<15} {stats.get('target_count',0):>5} lignes | {stats.get('load_rate',0):.1%} rétention | {len(s['drift_issues'])} drift")

    generate_final_dashboard(summary)
    print("="*60)
    return summary


if __name__ == "__main__":
    import sys
    tables = sys.argv[1:] if len(sys.argv) > 1 else None
    run_migration_executor(tables)
