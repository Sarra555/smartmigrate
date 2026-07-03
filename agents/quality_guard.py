"""
SmartMigrate — Agent 2 : Data Quality Guard
=============================================
Agent LangGraph qui résout le vrai problème #2 du métier migration :
"On découvre les problèmes de qualité APRÈS la migration — trop tard."

Ce que fait cet agent AVANT chaque migration :
1. [profile_data]      Analyse statistique complète de chaque colonne
2. [detect_anomalies]  Détecte les patterns problématiques (doublons, nulls critiques, formats)
3. [generate_rules]    Génère automatiquement les règles de validation (style Great Expectations)
4. [score_quality]     Calcule un score de qualité 0→1 par table et global
5. [human_gate]        Bloque la migration si score < seuil (défaut 0.80)
6. [generate_report]   Rapport HTML détaillé lisible par les non-techniques

Stack : LangGraph + Pandas + Great Expectations (sans serveur) + Gemini/MOCK
"""

import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

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

# ── Chemins projet ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "raw" / "erp_legacy.db"
OUTPUT_QA    = PROJECT_ROOT / "outputs" / "quality"
OUTPUT_QA.mkdir(parents=True, exist_ok=True)

QUALITY_THRESHOLD = float(os.getenv("QUALITY_SCORE_THRESHOLD", "0.80"))

TABLES = ["customers", "products", "orders", "order_lines", "suppliers"]

# Colonnes critiques par table (nulls interdits en production)
CRITICAL_COLUMNS = {
    "customers":   ["cst_id", "cst_email"],
    "products":    ["prod_id", "prod_ref", "prod_name"],
    "orders":      ["ord_id", "ord_cst_id", "ord_date", "ord_status"],
    "order_lines": ["line_id", "line_ord_id", "line_prod_id", "line_unit_price"],
    "suppliers":   ["sup_id", "sup_name"],
}

# ── State LangGraph ────────────────────────────────────────────────────────────

class QualityState(TypedDict):
    table_name:     str
    df:             Optional[Any]           # DataFrame pandas
    profile:        Optional[Dict]          # Stats par colonne
    anomalies:      List[Dict]              # Anomalies détectées
    rules:          List[Dict]              # Règles de validation générées
    rule_results:   List[Dict]              # Résultats de chaque règle
    quality_score:  float                   # Score global 0→1
    score_detail:   Dict                    # Détail du score par dimension
    approved:       bool                    # Approuvé pour migration ?
    llm_narrative:  str                     # Explication LLM en langage naturel
    errors:         List[str]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_table(table: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df   = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    conn.close()
    return df


def detect_date_format(series: pd.Series) -> Dict:
    """Détecte les formats de dates dans une colonne TEXT."""
    formats = {
        "ISO (YYYY-MM-DD)":   r"^\d{4}-\d{2}-\d{2}$",
        "FR (DD/MM/YYYY)":    r"^\d{2}/\d{2}/\d{4}$",
        "US (MM-DD-YYYY)":    r"^\d{2}-\d{2}-\d{4}$",
        "Compact (YYYYMMDD)": r"^\d{8}$",
    }
    samples    = series.dropna().astype(str).head(200)
    found      = {}
    for label, pattern in formats.items():
        count = samples.str.match(pattern).sum()
        if count > 0:
            found[label] = int(count)
    return found


def detect_bool_encoding(series: pd.Series) -> List[str]:
    """Détecte les valeurs booléennes encodées."""
    bool_vals = {"1","0","Y","N","yes","no","true","false","A","I","active","inactive"}
    unique    = set(series.dropna().astype(str).str.upper().unique())
    if unique and unique <= {v.upper() for v in bool_vals}:
        return list(unique)
    return []


def detect_status_encoding(series: pd.Series) -> List[str]:
    """Détecte les statuts encodés de multiples façons."""
    unique = list(series.dropna().astype(str).unique())
    if 2 <= len(unique) <= 15:
        return unique
    return []


# ── Noeud 1 : Profile des données ─────────────────────────────────────────────

def node_profile_data(state: QualityState) -> QualityState:
    table = state["table_name"]
    print(f"\n  📊 Profiling des données : {table}")

    try:
        df      = load_table(table)
        n_rows  = len(df)
        profile = {"table": table, "total_rows": n_rows, "columns": {}}

        for col in df.columns:
            series   = df[col]
            n_null   = int(series.isna().sum())
            n_unique = int(series.nunique())
            dtype    = str(series.dtype)

            col_info = {
                "dtype":       dtype,
                "null_count":  n_null,
                "null_rate":   round(n_null / n_rows, 4) if n_rows else 0,
                "unique_count":n_unique,
                "unique_rate": round(n_unique / n_rows, 4) if n_rows else 0,
            }

            # Analyse numérique
            if pd.api.types.is_numeric_dtype(series):
                col_info["min"]  = float(series.min()) if not series.isna().all() else None
                col_info["max"]  = float(series.max()) if not series.isna().all() else None
                col_info["mean"] = round(float(series.mean()), 2) if not series.isna().all() else None
                col_info["negative_count"] = int((series < 0).sum())

            # Analyse textuelle
            if dtype == "object":
                str_series = series.dropna().astype(str)
                col_info["sample_values"] = str_series.head(5).tolist()
                col_info["avg_length"]    = round(str_series.str.len().mean(), 1) if len(str_series) > 0 else 0

                # Détections spéciales
                date_fmts = detect_date_format(series)
                if date_fmts:
                    col_info["date_formats"] = date_fmts
                    col_info["mixed_dates"]  = len(date_fmts) > 1

                bool_enc = detect_bool_encoding(series)
                if bool_enc:
                    col_info["boolean_encoding"] = bool_enc

                status_enc = detect_status_encoding(series)
                if status_enc and not bool_enc:
                    col_info["status_encoding"] = status_enc

            profile["columns"][col] = col_info

        print(f"     {n_rows} lignes · {len(df.columns)} colonnes profilées")
        return {**state, "df": df, "profile": profile}

    except Exception as e:
        return {**state, "errors": state["errors"] + [f"profile: {str(e)}"]}


# ── Noeud 2 : Détection des anomalies ─────────────────────────────────────────

def node_detect_anomalies(state: QualityState) -> QualityState:
    table   = state["table_name"]
    profile = state["profile"]
    df      = state["df"]
    print(f"  🔍 Détection des anomalies...")

    anomalies = []
    critical  = CRITICAL_COLUMNS.get(table, [])

    for col, info in profile["columns"].items():
        # 1. Nulls sur colonnes critiques
        if col in critical and info["null_rate"] > 0:
            anomalies.append({
                "type":     "CRITICAL_NULLS",
                "severity": "HIGH",
                "column":   col,
                "detail":   f"{info['null_count']} nulls sur colonne critique ({info['null_rate']*100:.1f}%)",
                "impact":   "Bloquant — insertion impossible en base cible (NOT NULL)"
            })

        # 2. Taux de nulls élevé sur colonnes non-critiques
        elif info["null_rate"] > 0.15:
            anomalies.append({
                "type":     "HIGH_NULL_RATE",
                "severity": "MEDIUM",
                "column":   col,
                "detail":   f"{info['null_rate']*100:.1f}% de valeurs manquantes",
                "impact":   "Perte de données potentielle après migration"
            })

        # 3. Formats de dates mixtes
        if info.get("mixed_dates"):
            fmts = info["date_formats"]
            anomalies.append({
                "type":     "MIXED_DATE_FORMATS",
                "severity": "HIGH",
                "column":   col,
                "detail":   f"{len(fmts)} formats différents : {list(fmts.keys())}",
                "impact":   "Erreur de conversion CAST — dates perdues ou corrompues"
            })

        # 4. Booléens encodés
        if info.get("boolean_encoding"):
            anomalies.append({
                "type":     "BOOLEAN_ENCODING",
                "severity": "MEDIUM",
                "column":   col,
                "detail":   f"Valeurs encodées : {info['boolean_encoding']}",
                "impact":   "Doit être normalisé en TRUE/FALSE avant migration"
            })

        # 5. Statuts multiples encodages
        if info.get("status_encoding") and len(info["status_encoding"]) > 3:
            anomalies.append({
                "type":     "STATUS_ENCODING",
                "severity": "HIGH",
                "column":   col,
                "detail":   f"{len(info['status_encoding'])} valeurs distinctes : {info['status_encoding'][:8]}",
                "impact":   "CASE WHEN requis — risque de statuts non mappés"
            })

        # 6. Valeurs négatives sur numériques
        if info.get("negative_count", 0) > 0:
            anomalies.append({
                "type":     "NEGATIVE_VALUES",
                "severity": "MEDIUM",
                "column":   col,
                "detail":   f"{info['negative_count']} valeurs négatives (min={info.get('min')})",
                "impact":   "Violation contrainte CHECK >= 0 en base cible"
            })

    # 7. Doublons globaux
    if df is not None:
        n_dup = int(df.duplicated().sum())
        if n_dup > 0:
            anomalies.append({
                "type":     "DUPLICATE_ROWS",
                "severity": "HIGH",
                "column":   "ALL",
                "detail":   f"{n_dup} lignes en double",
                "impact":   "Doublons migrés = données corrompues en production"
            })

        # 8. Doublons sur colonnes email / ref
        for col in df.columns:
            if any(kw in col.lower() for kw in ["email", "ref", "mail"]):
                non_null  = df[col].dropna()
                n_dup_col = int(non_null.duplicated().sum())
                if n_dup_col > 0:
                    anomalies.append({
                        "type":     "DUPLICATE_KEY",
                        "severity": "HIGH",
                        "column":   col,
                        "detail":   f"{n_dup_col} doublons sur colonne unique",
                        "impact":   "Violation contrainte UNIQUE en base cible"
                    })

    high_count   = sum(1 for a in anomalies if a["severity"] == "HIGH")
    medium_count = sum(1 for a in anomalies if a["severity"] == "MEDIUM")
    print(f"     {len(anomalies)} anomalies : {high_count} HIGH · {medium_count} MEDIUM")

    return {**state, "anomalies": anomalies}


# ── Noeud 3 : Génération des règles de validation ─────────────────────────────

def node_generate_rules(state: QualityState) -> QualityState:
    table    = state["table_name"]
    profile  = state["profile"]
    critical = CRITICAL_COLUMNS.get(table, [])
    print(f"  📋 Génération des règles de validation...")

    rules = []

    for col, info in profile["columns"].items():
        # Règle 1 : Not null sur colonnes critiques
        if col in critical:
            rules.append({
                "rule_id":   f"{col}__not_null",
                "column":    col,
                "type":      "expect_column_values_to_not_be_null",
                "severity":  "CRITICAL",
                "threshold": 1.0,   # 0 null toléré
                "description": f"La colonne {col} ne doit contenir aucun NULL"
            })

        # Règle 2 : Taux de nulls acceptable
        elif info["null_rate"] > 0:
            max_null = min(info["null_rate"] * 1.2, 0.30)  # 20% de tolérance
            rules.append({
                "rule_id":   f"{col}__null_rate",
                "column":    col,
                "type":      "expect_column_values_null_rate_below",
                "severity":  "WARNING",
                "threshold": round(max_null, 3),
                "description": f"Taux de null < {max_null*100:.0f}% pour {col}"
            })

        # Règle 3 : Valeurs dans un set connu (statuts)
        if info.get("status_encoding"):
            rules.append({
                "rule_id":    f"{col}__valid_values",
                "column":     col,
                "type":       "expect_column_values_to_be_in_set",
                "severity":   "HIGH",
                "valid_set":  info["status_encoding"],
                "description": f"Valeurs de {col} dans l'ensemble connu"
            })

        # Règle 4 : Pas de valeurs négatives
        if info.get("negative_count", 0) >= 0 and info["dtype"] != "object":
            if col not in critical:
                rules.append({
                    "rule_id":   f"{col}__non_negative",
                    "column":    col,
                    "type":      "expect_column_values_to_be_between",
                    "severity":  "HIGH",
                    "min_value": 0,
                    "description": f"{col} doit être >= 0"
                })

        # Règle 5 : Unicité sur colonnes identifiantes
        if info["unique_rate"] > 0.98 and info["null_rate"] < 0.01:
            if any(kw in col.lower() for kw in ["id", "ref", "email", "code"]):
                rules.append({
                    "rule_id":   f"{col}__unique",
                    "column":    col,
                    "type":      "expect_column_values_to_be_unique",
                    "severity":  "HIGH",
                    "description": f"{col} doit être unique"
                })

    print(f"     {len(rules)} règles générées")
    return {**state, "rules": rules}


# ── Noeud 4 : Exécution des règles + scoring ──────────────────────────────────

def node_score_quality(state: QualityState) -> QualityState:
    df    = state["df"]
    rules = state["rules"]
    print(f"  🧮 Calcul du score de qualité...")

    results = []
    n_rows  = len(df)

    for rule in rules:
        col    = rule["column"]
        r_type = rule["type"]
        passed = False
        detail = ""

        try:
            if col not in df.columns:
                continue

            series = df[col]

            if r_type == "expect_column_values_to_not_be_null":
                n_null = int(series.isna().sum())
                passed = n_null == 0
                detail = f"{n_null} nulls trouvés" if not passed else "✓ Aucun null"

            elif r_type == "expect_column_values_null_rate_below":
                rate   = series.isna().sum() / n_rows
                passed = rate <= rule["threshold"]
                detail = f"Taux null : {rate*100:.1f}% (seuil {rule['threshold']*100:.0f}%)"

            elif r_type == "expect_column_values_to_be_in_set":
                valid      = set(str(v) for v in rule["valid_set"])
                n_invalid  = int(series.dropna().astype(str).apply(lambda x: x not in valid).sum())
                passed     = n_invalid == 0
                detail     = f"{n_invalid} valeurs hors ensemble" if not passed else "✓ Toutes les valeurs sont valides"

            elif r_type == "expect_column_values_to_be_between":
                non_null  = series.dropna()
                n_below   = int((non_null < rule["min_value"]).sum())
                passed    = n_below == 0
                detail    = f"{n_below} valeurs < {rule['min_value']}" if not passed else f"✓ Toutes >= {rule['min_value']}"

            elif r_type == "expect_column_values_to_be_unique":
                non_null  = series.dropna()
                n_dup     = int(non_null.duplicated().sum())
                passed    = n_dup == 0
                detail    = f"{n_dup} doublons" if not passed else "✓ Toutes les valeurs sont uniques"

        except Exception as e:
            detail = f"Erreur : {str(e)}"
            passed = False

        results.append({
            **rule,
            "passed": passed,
            "detail": detail
        })

    # ── Calcul du score ───────────────────────────────────────────────────────
    # Pondération par sévérité
    weights = {"CRITICAL": 3.0, "HIGH": 2.0, "WARNING": 1.0}

    total_weight  = sum(weights.get(r["severity"], 1) for r in results)
    passed_weight = sum(weights.get(r["severity"], 1) for r in results if r["passed"])

    score = round(passed_weight / total_weight, 4) if total_weight > 0 else 1.0

    # Score par dimension
    dims = {}
    for sev in ["CRITICAL", "HIGH", "WARNING"]:
        sub = [r for r in results if r["severity"] == sev]
        if sub:
            dims[sev] = {
                "total":  len(sub),
                "passed": sum(1 for r in sub if r["passed"]),
                "score":  round(sum(1 for r in sub if r["passed"]) / len(sub), 3)
            }

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = len(results) - n_pass
    print(f"     Score : {score:.2%} | {n_pass} règles OK · {n_fail} échouées")

    return {**state, "rule_results": results, "quality_score": score, "score_detail": dims}


# ── Noeud 5 : Gate — bloquer ou autoriser ─────────────────────────────────────

def node_quality_gate(state: QualityState) -> Literal["blocked", "approved"]:
    score = state["quality_score"]
    if score >= QUALITY_THRESHOLD:
        return "approved"
    return "blocked"


def node_handle_blocked(state: QualityState) -> QualityState:
    score = state["quality_score"]
    print(f"\n  🚫 MIGRATION BLOQUÉE — Score {score:.2%} < seuil {QUALITY_THRESHOLD:.2%}")
    print(f"     Corriger les anomalies HIGH/CRITICAL avant de relancer.")
    return {**state, "approved": False}


def node_handle_approved(state: QualityState) -> QualityState:
    score = state["quality_score"]
    print(f"\n  ✅ MIGRATION AUTORISÉE — Score {score:.2%} >= seuil {QUALITY_THRESHOLD:.2%}")
    return {**state, "approved": True}


# ── Noeud 6 : Narratif LLM ────────────────────────────────────────────────────

def node_llm_narrative(state: QualityState) -> QualityState:
    """Génère une explication en langage naturel lisible par les non-techniques."""
    table    = state["table_name"]
    score    = state["quality_score"]
    approved = state["approved"]
    anomalies= state["anomalies"]
    n_rows   = state["profile"]["total_rows"]

    # Mode MOCK si pas de clé API
    if not LANGCHAIN_OK or not os.getenv("GOOGLE_API_KEY"):
        high_anoms = [a for a in anomalies if a["severity"] == "HIGH"]
        status_txt = "autorisée" if approved else "BLOQUÉE"
        narrative  = (
            f"Analyse de la table **{table}** ({n_rows} lignes) — "
            f"Score de qualité : **{score:.0%}** — Migration {status_txt}.\n\n"
        )
        if high_anoms:
            narrative += "**Problèmes critiques détectés :**\n"
            for a in high_anoms[:3]:
                narrative += f"- {a['column']} : {a['detail']} → {a['impact']}\n"
        else:
            narrative += "Aucun problème bloquant détecté. Données prêtes pour migration."
        return {**state, "llm_narrative": narrative}

    # Vrai LLM
    system = """Tu es un expert en qualité de données. Tu expliques les résultats
d'une analyse de qualité en langage clair, lisible par un manager non-technique.
Sois concis (5-8 lignes max), précis et actionnable."""

    anomaly_summary = json.dumps(anomalies[:5], ensure_ascii=False, indent=2)
    user_msg = f"""Table : {table} | {n_rows} lignes | Score qualité : {score:.0%} | Migration : {'AUTORISÉE' if approved else 'BLOQUÉE'}

Anomalies principales :
{anomaly_summary}

Écris un résumé exécutif en français pour le responsable de projet."""

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

        narrative = resp.content.strip() if resp else "Rapport narratif non disponible."
    except Exception as e:
        narrative = f"[LLM indisponible : {str(e)[:80]}]"

    return {**state, "llm_narrative": narrative}


# ── Noeud 7 : Rapport HTML ────────────────────────────────────────────────────

def node_generate_report(state: QualityState) -> QualityState:
    table    = state["table_name"]
    score    = state["quality_score"]
    approved = state["approved"]
    profile  = state["profile"]
    anomalies= state["anomalies"]
    results  = state["rule_results"]
    narrative= state["llm_narrative"]
    dims     = state["score_detail"]

    score_color = "#16a34a" if score >= 0.85 else "#d97706" if score >= 0.70 else "#dc2626"
    status_txt  = "✅ AUTORISÉE" if approved else "🚫 BLOQUÉE"
    status_bg   = "#dcfce7" if approved else "#fee2e2"

    # Anomalies HTML
    anom_html = ""
    for a in anomalies:
        sev_color = {"HIGH":"#dc2626","MEDIUM":"#d97706","LOW":"#2563eb"}.get(a["severity"],"#64748b")
        anom_html += f"""
        <tr>
          <td><span style="color:{sev_color};font-weight:600">{a['severity']}</span></td>
          <td><code>{a['column']}</code></td>
          <td>{a['type']}</td>
          <td>{a['detail']}</td>
          <td style="color:#64748b;font-size:11px">{a['impact']}</td>
        </tr>"""

    # Rules HTML
    rules_html = ""
    for r in results:
        icon   = "✅" if r["passed"] else "❌"
        bg     = "#f0fdf4" if r["passed"] else "#fef2f2"
        rules_html += f"""
        <tr style="background:{bg}">
          <td>{icon}</td>
          <td><code>{r['rule_id']}</code></td>
          <td><span style="font-size:11px;padding:2px 6px;border-radius:4px;background:#f1f5f9">{r['severity']}</span></td>
          <td style="font-size:12px">{r['description']}</td>
          <td style="font-size:12px;color:#475569">{r.get('detail','')}</td>
        </tr>"""

    # Score dims HTML
    dims_html = ""
    for sev, d in dims.items():
        color  = {"CRITICAL":"#dc2626","HIGH":"#d97706","WARNING":"#2563eb"}.get(sev,"#64748b")
        pct    = int(d["score"] * 100)
        dims_html += f"""
        <div style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="font-size:13px;font-weight:500;color:{color}">{sev}</span>
            <span style="font-size:13px;font-weight:600">{d['passed']}/{d['total']} ({pct}%)</span>
          </div>
          <div style="background:#f1f5f9;border-radius:99px;height:8px">
            <div style="background:{color};height:8px;border-radius:99px;width:{pct}%"></div>
          </div>
        </div>"""

    # Narrative (markdown simple → HTML)
    narrative_html = narrative.replace("**", "<strong>").replace("**", "</strong>").replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Quality Report — {table}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f8fafc; color:#0f172a; padding:28px; }}
  h1 {{ font-size:22px; font-weight:700; margin-bottom:4px; }}
  h2 {{ font-size:14px; font-weight:600; margin-bottom:12px; color:#374151; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }}
  .card {{ background:white; border:1px solid #e2e8f0; border-radius:12px; padding:20px 24px; }}
  .stat {{ text-align:center; }}
  .stat-num {{ font-size:32px; font-weight:700; }}
  .stat-lbl {{ font-size:12px; color:#64748b; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th {{ background:#f8fafc; padding:8px 12px; text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:#64748b; border-bottom:1px solid #e2e8f0; }}
  td {{ padding:8px 12px; border-bottom:1px solid #f1f5f9; vertical-align:top; }}
  code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:11.5px; }}
  .narrative {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:16px 20px; font-size:13px; line-height:1.7; margin-bottom:20px; }}
</style></head><body>

<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
  <div>
    <h1>🛡️ Data Quality Report — <code>{table}</code></h1>
    <p style="font-size:13px;color:#64748b">Généré par SmartMigrate Quality Guard · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  </div>
  <div style="background:{status_bg};padding:10px 20px;border-radius:10px;font-size:15px;font-weight:600">
    {status_txt}
  </div>
</div>

<div class="grid3">
  <div class="card stat">
    <div class="stat-num" style="color:{score_color}">{score:.0%}</div>
    <div class="stat-lbl">Score de qualité global</div>
  </div>
  <div class="card stat">
    <div class="stat-num">{profile['total_rows']:,}</div>
    <div class="stat-lbl">Lignes analysées</div>
  </div>
  <div class="card stat">
    <div class="stat-num" style="color:#dc2626">{sum(1 for a in anomalies if a['severity']=='HIGH')}</div>
    <div class="stat-lbl">Anomalies HIGH</div>
  </div>
</div>

<div class="narrative">
  <strong>📝 Résumé exécutif</strong><br><br>
  {narrative_html}
</div>

<div class="grid2">
  <div class="card">
    <h2>Score par dimension</h2>
    {dims_html}
  </div>
  <div class="card">
    <h2>Profil de la table</h2>
    <table>
      <tr><th>Colonne</th><th>Type</th><th>Nulls</th><th>Uniques</th></tr>
      {''.join(f'<tr><td><code>{c}</code></td><td style="color:#64748b;font-size:11px">{i["dtype"]}</td><td style="color:{"#dc2626" if i["null_rate"]>0.1 else "#374151"}">{i["null_rate"]*100:.1f}%</td><td>{i["unique_count"]}</td></tr>' for c, i in profile["columns"].items())}
    </table>
  </div>
</div>

<div class="card" style="margin-bottom:20px">
  <h2 style="margin-bottom:16px">⚠️ Anomalies détectées ({len(anomalies)})</h2>
  <table>
    <tr><th>Sévérité</th><th>Colonne</th><th>Type</th><th>Détail</th><th>Impact</th></tr>
    {anom_html if anom_html else '<tr><td colspan="5" style="text-align:center;color:#16a34a;padding:16px">✅ Aucune anomalie détectée</td></tr>'}
  </table>
</div>

<div class="card">
  <h2 style="margin-bottom:16px">📋 Règles de validation ({len(results)} règles)</h2>
  <table>
    <tr><th>Statut</th><th>Règle</th><th>Sévérité</th><th>Description</th><th>Résultat</th></tr>
    {rules_html}
  </table>
</div>

</body></html>"""

    path = OUTPUT_QA / f"{table}_quality_report.html"
    path.write_text(html, encoding="utf-8")

    # Sauvegarde JSON
    json_path = OUTPUT_QA / f"{table}_quality.json"
    import numpy as np
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)
    json_path.write_text(json.dumps({
        "table":         table,
        "score":         score,
        "approved":      approved,
        "anomalies":     anomalies,
        "rule_results":  results,
        "score_detail":  dims,
        "generated_at":  datetime.now().isoformat()
    }, ensure_ascii=False, indent=2, cls=NpEncoder), encoding="utf-8")

    print(f"  💾 Rapport sauvegardé : outputs/quality/{table}_quality_report.html")
    return state


# ── Construction du graphe ─────────────────────────────────────────────────────

def build_quality_graph():
    graph = StateGraph(QualityState)

    graph.add_node("profile_data",      node_profile_data)
    graph.add_node("detect_anomalies",  node_detect_anomalies)
    graph.add_node("generate_rules",    node_generate_rules)
    graph.add_node("score_quality",     node_score_quality)
    graph.add_node("handle_blocked",    node_handle_blocked)
    graph.add_node("handle_approved",   node_handle_approved)
    graph.add_node("llm_narrative",     node_llm_narrative)
    graph.add_node("generate_report",   node_generate_report)

    graph.set_entry_point("profile_data")
    graph.add_edge("profile_data",     "detect_anomalies")
    graph.add_edge("detect_anomalies", "generate_rules")
    graph.add_edge("generate_rules",   "score_quality")

    graph.add_conditional_edges(
        "score_quality",
        node_quality_gate,
        {"blocked": "handle_blocked", "approved": "handle_approved"}
    )

    graph.add_edge("handle_blocked",  "llm_narrative")
    graph.add_edge("handle_approved", "llm_narrative")
    graph.add_edge("llm_narrative",   "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ── Runner principal ───────────────────────────────────────────────────────────

def run_quality_guard(tables: List[str] = None) -> Dict:
    if tables is None:
        tables = TABLES

    print("\n" + "="*60)
    print("  🛡️  SmartMigrate — Data Quality Guard Agent")
    print("="*60)

    app     = build_quality_graph()
    summary = {}

    for table in tables:
        print(f"\n{'─'*60}")
        print(f"  Table : {table.upper()}")
        print(f"{'─'*60}")

        initial: QualityState = {
            "table_name":    table,
            "df":            None,
            "profile":       None,
            "anomalies":     [],
            "rules":         [],
            "rule_results":  [],
            "quality_score": 0.0,
            "score_detail":  {},
            "approved":      False,
            "llm_narrative": "",
            "errors":        []
        }

        try:
            result = app.invoke(initial)
            summary[table] = {
                "score":     result["quality_score"],
                "approved":  result["approved"],
                "anomalies": len(result["anomalies"]),
                "errors":    result["errors"]
            }
        except Exception as e:
            print(f"  ❌ Erreur : {e}")
            summary[table] = {"score": 0, "approved": False, "anomalies": 0, "errors": [str(e)]}

    # Résumé final
    print(f"\n{'='*60}")
    print("  RÉSUMÉ QUALITÉ — TOUTES LES TABLES")
    print(f"{'='*60}")

    all_approved = True
    for table, s in summary.items():
        icon = "✅" if s["approved"] else "🚫"
        print(f"  {icon} {table:<15} score={s['score']:.0%}  anomalies={s['anomalies']}")
        if not s["approved"]:
            all_approved = False

    print(f"\n  {'✅ PIPELINE PRÊT — toutes les tables approuvées' if all_approved else '🚫 PIPELINE BLOQUÉ — corriger les tables refusées'}")
    print(f"  Rapports HTML → outputs/quality/")
    print("="*60)

    return summary


if __name__ == "__main__":
    import sys
    tables = sys.argv[1:] if len(sys.argv) > 1 else None
    run_quality_guard(tables)
