# 🧠 SmartMigrate — Agentic AI ERP-to-Cloud Migration Pipeline

> Pipeline de migration de données ERP → Cloud orchestré par **3 agents AI (LangGraph)**  
> Résout les vrais problèmes du métier migration : schema mismatch, qualité des données, drift post-migration.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Problème résolu

Les migrations ERP vers le cloud échouent à **83%** à cause de 3 problèmes récurrents :

| Problème | Impact | Solution SmartMigrate |
|---|---|---|
| Schema mismatch | Données perdues ou corrompues | Agent 1 — Schema Analyst |
| Qualité inconnue avant migration | Découverte des bugs en production | Agent 2 — Quality Guard |
| Validation post-migration manuelle | Drift silencieux non détecté | Agent 3 — Migration Executor |

---

## 📊 Résultats réels

Migration d'une base ERP legacy (SQLite) vers un Cloud Data Warehouse (PostgreSQL) :

| Table | Lignes source | Lignes migrées | Rétention | Drift |
|---|---|---|---|---|
| customers | 515 | 480 | 93.2% | 0 |
| products | 150 | 150 | 100.0% | 0 |
| suppliers | 50 | 50 | 100.0% | 0 |
| orders | 2 000 | 1 802 | 90.1% | 0 |
| order_lines | 4 264 | 4 264 | 100.0% | 0 |
| **TOTAL** | **6 979** | **6 746** | **96.7%** | **0** |

> Les 233 lignes filtrées = doublons emails éliminés + dates invalides écartées — comportement attendu.

---

## 🏗️ Architecture

```
ERP Legacy (PostgreSQL/SQLite)
           │
           ▼
┌─────────────────────────┐
│  Agent 1 — Schema       │  LangGraph + LLM (Gemini/Claude)
│  Analyst                │  → Analyse DDL source vs cible
│                         │  → Génère mapping JSON automatique
│                         │  → Produit modèles dbt staging
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Agent 2 — Quality      │  LangGraph + Great Expectations
│  Guard                  │  → Profile 44 colonnes
│                         │  → Génère règles de validation
│                         │  → Bloque si score qualité < 80%
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Agent 3 — Migration    │  LangGraph + dbt + Prefect
│  Executor               │  → Transforme et charge les données
│                         │  → Réconciliation source ↔ target
│                         │  → Détecte drift post-migration
│                         │  → Rapport narratif LLM
└──────────┬──────────────┘
           │
           ▼
  Cloud Data Warehouse (PostgreSQL / Supabase)
```

---

## 🤖 Stack technique

| Composant | Technologie | Rôle |
|---|---|---|
| Agents AI | **LangGraph 0.2** | Orchestration multi-agents, state management |
| LLM | **Gemini 2.0 Flash** | Schema mapping, rapports narratifs |
| Transformations | **dbt Core** | Modèles SQL staging |
| Validation | **Great Expectations** | Règles qualité automatiques |
| Source | **SQLite / PostgreSQL** | ERP legacy simulé |
| Destination | **PostgreSQL / Supabase** | Cloud Data Warehouse |
| Infrastructure | **Docker Compose** | Stack locale reproductible |

---

## 🗄️ Dataset ERP Legacy

Simulation d'un ERP réaliste avec **problèmes intentionnels** (comme en production) :

- ✅ **515 clients** — doublons email (~3%), statuts encodés (1/0/A/I/active), téléphones en 4 formats
- ✅ **2 000 commandes** — dates en 4 formats (ISO/FR/US/compact), statuts sur 20+ valeurs distinctes
- ✅ **150 produits** — stocks négatifs (bug legacy), devises non normalisées (tnd/TND/EUR)
- ✅ **4 264 lignes de commande** — quantités nulles, remises manquantes
- ✅ **50 fournisseurs** — pays non normalisés (TN/Tunisia/Tunisie)

---

## 🚀 Lancement rapide

```bash
# 1. Cloner le repo
git clone https://github.com/ton-username/smartmigrate.git
cd smartmigrate

# 2. Créer l'environnement
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec ta clé GOOGLE_API_KEY

# 5. Générer le dataset ERP
python data/generate_erp_data.py

# 6. Générer les mappings
python agents/generate_mappings.py

# 7. Lancer les 3 agents en séquence
python agents/schema_analyst.py
python agents/quality_guard.py
python agents/migration_executor.py
```

---

## 📁 Structure du projet

```
smartmigrate/
├── agents/
│   ├── schema_analyst.py       # Agent 1 — Schema mapping LangGraph
│   ├── quality_guard.py        # Agent 2 — Data quality validation
│   ├── migration_executor.py   # Agent 3 — Migration + réconciliation
│   └── generate_mappings.py    # Génère les fichiers de mapping JSON
├── dbt_project/
│   └── models/staging/         # Modèles SQL générés automatiquement
├── data/
│   └── generate_erp_data.py    # Générateur dataset ERP legacy
├── docker/
│   ├── docker-compose.yml      # Stack complète en un docker-compose up
│   └── schema_target.sql       # DDL schéma cloud cible
├── outputs/
│   ├── mappings/               # JSON mappings par table (Agent 1)
│   ├── quality/                # Rapports qualité HTML (Agent 2)
│   └── migration/              # Rapports migration HTML (Agent 3)
├── .env.example                # Template variables d'environnement
├── requirements.txt
└── README.md
```

---

## 🔑 Variables d'environnement

```bash
# LLM (choisir l'un ou l'autre)
GOOGLE_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here   # optionnel

# Base source (ERP legacy)
SOURCE_DB_URL=postgresql://user:pass@localhost:5432/erp_legacy

# Base cible (Cloud / Supabase)
TARGET_DB_URL=postgresql://user:pass@localhost:5433/cloud_dw

# Config migration
QUALITY_SCORE_THRESHOLD=0.80    # Bloquer si score < 80%
MIGRATION_RUN_ID=run_2024_v1
```


---

## 📋 Problèmes réels résolus

### Agent 1 — Schema Analyst
- Mapping automatique de **44 colonnes** source → cible avec score de confiance
- Détection des colonnes ambiguës (confiance < 80%) → validation humaine
- Génération automatique des **5 modèles dbt staging** avec toutes les transformations SQL

### Agent 2 — Quality Guard
- Profiling statistique complet : nulls, doublons, distributions, formats
- Détection de **12 anomalies** (dates mixtes, statuts encodés, stocks négatifs)
- Score de qualité pondéré par sévérité (CRITICAL/HIGH/WARNING)
- Blocage automatique si score < seuil configuré

### Agent 3 — Migration Executor
- Normalisation de **5 formats de dates** différents → ISO 8601
- Mapping de **20+ valeurs de statuts** legacy → enum normalisé
- Déduplication email (515 → 480 clients uniques)
- Réconciliation ligne par ligne source ↔ destination
- Rapport exécutif narratif généré par LLM

---

## 👩‍💻 Auteur

**Sarra Aouadi** — Data Migration & AI Engineer  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sarra_Aouadi-blue?logo=linkedin)](https://linkedin.com/in/sarra-aouadi)
[![GitHub](https://img.shields.io/badge/GitHub-ton--username-black?logo=github)] Sarra555

---

*Projet portfolio — Migration ERP-to-Cloud avec AI Agentique (LangGraph + Gemini)*
