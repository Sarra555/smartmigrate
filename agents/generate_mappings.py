"""
generate_mappings.py
Génère les fichiers JSON de mapping pour les 5 tables.
Lance ce script une seule fois pour alimenter l'Agent 3.
"""
import json
from pathlib import Path

OUTPUT = Path("outputs/mappings")
OUTPUT.mkdir(parents=True, exist_ok=True)

MAPPINGS = {

"customers": {
    "table_source": "customers", "table_target": "customers",
    "migration_complexity": "high",
    "complexity_reason": "Doublons email, statuts encodés, dates mixtes, pays non normalisés",
    "mappings": [
        {"source_col":"cst_id",       "target_col":"legacy_cst_id", "transform":"direct",           "confidence":0.99, "notes":"PK source"},
        {"source_col":"cst_fname",    "target_col":"first_name",    "transform":"direct",           "confidence":0.97, "notes":"Prénom"},
        {"source_col":"cst_lname",    "target_col":"last_name",     "transform":"direct",           "confidence":0.97, "notes":"Nom"},
        {"source_col":"cst_company",  "target_col":"company_name",  "transform":"direct",           "confidence":0.95, "notes":"Société"},
        {"source_col":"cst_email",    "target_col":"email",         "transform":"lowercase",        "confidence":0.95, "notes":"Email — lowercase + strip"},
        {"source_col":"cst_phone",    "target_col":"phone",         "transform":"normalize_phone",  "confidence":0.61, "notes":"Formats mixtes → E.164"},
        {"source_col":"cst_country",  "target_col":"country_code",  "transform":"normalize_country","confidence":0.68, "notes":"TN/Tunisia/Tunisie → ISO 2"},
        {"source_col":"cst_city",     "target_col":"city",          "transform":"direct",           "confidence":0.95, "notes":"Ville"},
        {"source_col":"cst_address",  "target_col":"address",       "transform":"direct",           "confidence":0.90, "notes":"Adresse"},
        {"source_col":"cst_status",   "target_col":"status",        "transform":"normalize_status", "confidence":0.72, "notes":"1/0/A/I/active → active/inactive"},
        {"source_col":"cst_segment",  "target_col":"segment",       "transform":"direct",           "confidence":0.90, "notes":"B2B/B2C/VIP"},
        {"source_col":"cst_created_dt","target_col":"created_at",   "transform":"normalize_date",   "confidence":0.65, "notes":"Formats dates mixtes → ISO"},
        {"source_col":"cst_notes",    "target_col":"notes",         "transform":"direct",           "confidence":0.97, "notes":"Notes libres"},
    ],
    "unmapped_source": [],
    "unmapped_target": ["customer_id","migrated_at"],
    "migration_risks": [
        "~3% doublons email → dédupliqués",
        "Pays non normalisés (TN/Tunisia/Tunisie)",
        "Statuts encodés sur 6 valeurs distinctes",
        "Dates en 4 formats différents"
    ]
},

"products": {
    "table_source": "products", "table_target": "products",
    "migration_complexity": "medium",
    "complexity_reason": "Stocks négatifs, devise non normalisée, booléens encodés",
    "mappings": [
        {"source_col":"prod_id",      "target_col":"legacy_prod_id","transform":"direct",           "confidence":0.99, "notes":"PK source"},
        {"source_col":"prod_ref",     "target_col":"reference",     "transform":"direct",           "confidence":0.97, "notes":"Référence produit"},
        {"source_col":"prod_name",    "target_col":"name",          "transform":"direct",           "confidence":0.97, "notes":"Nom produit"},
        {"source_col":"prod_category","target_col":"category",      "transform":"direct",           "confidence":0.93, "notes":"Catégorie"},
        {"source_col":"prod_price",   "target_col":"unit_price",    "transform":"cast_numeric",     "confidence":0.88, "notes":"Prix — clip(lower=0)"},
        {"source_col":"prod_currency","target_col":"currency_code", "transform":"normalize_country","confidence":0.71, "notes":"tnd/TND/eur → ISO 4217"},
        {"source_col":"prod_stock",   "target_col":"stock_qty",     "transform":"cast_numeric",     "confidence":0.75, "notes":"Stocks négatifs → 0"},
        {"source_col":"prod_unit",    "target_col":"unit",          "transform":"direct",           "confidence":0.90, "notes":"Unité"},
        {"source_col":"prod_active",  "target_col":"is_active",     "transform":"normalize_bool",   "confidence":0.73, "notes":"Y/N/1/0 → BOOLEAN"},
        {"source_col":"prod_created", "target_col":"created_at",    "transform":"normalize_date",   "confidence":0.69, "notes":"Formats dates mixtes"},
    ],
    "unmapped_source": [],
    "unmapped_target": ["product_id","migrated_at"],
    "migration_risks": [
        "Stocks négatifs (~1-4%) → remplacés par 0",
        "Devise non normalisée (tnd/TND/EUR)",
        "Prix manquants (~2%)"
    ]
},

"orders": {
    "table_source": "orders", "table_target": "orders",
    "migration_complexity": "high",
    "complexity_reason": "Statuts encodés sur 20+ valeurs, dates mixtes, FK client",
    "mappings": [
        {"source_col":"ord_id",         "target_col":"legacy_ord_id",     "transform":"direct",          "confidence":0.99, "notes":"PK source"},
        {"source_col":"ord_ref",        "target_col":"order_reference",   "transform":"direct",          "confidence":0.97, "notes":"Référence commande"},
        {"source_col":"ord_cst_id",     "target_col":"customer_id",       "transform":"direct",          "confidence":0.95, "notes":"FK client"},
        {"source_col":"ord_date",       "target_col":"order_date",        "transform":"normalize_date",  "confidence":0.70, "notes":"Formats mixtes ISO/FR/US/compact"},
        {"source_col":"ord_exp_deliver","target_col":"expected_delivery",  "transform":"normalize_date",  "confidence":0.70, "notes":"Formats mixtes"},
        {"source_col":"ord_delivered",  "target_col":"delivered_at",      "transform":"normalize_date",  "confidence":0.68, "notes":"Nullable — formats mixtes"},
        {"source_col":"ord_status",     "target_col":"status",            "transform":"normalize_status","confidence":0.65, "notes":"P/PEND/pending/0 → pending etc."},
        {"source_col":"ord_total_ht",   "target_col":"subtotal_ht",       "transform":"cast_numeric",    "confidence":0.88, "notes":"Montant HT"},
        {"source_col":"ord_tva",        "target_col":"tva_rate",          "transform":"direct",          "confidence":0.90, "notes":"Taux TVA (0/7/19)"},
        {"source_col":"ord_channel",    "target_col":"channel",           "transform":"uppercase",       "confidence":0.93, "notes":"WEB/TEL/STORE/API"},
        {"source_col":"ord_notes",      "target_col":"notes",             "transform":"direct",          "confidence":0.97, "notes":"Notes libres"},
    ],
    "unmapped_source": [],
    "unmapped_target": ["order_id","migrated_at"],
    "migration_risks": [
        "Statuts encodés sur 20+ valeurs distinctes",
        "Dates en 4 formats différents",
        "FK ord_cst_id → résoudre via legacy_cst_id",
        "Totaux HT manquants (~1%)"
    ]
},

"order_lines": {
    "table_source": "order_lines", "table_target": "order_lines",
    "migration_complexity": "medium",
    "complexity_reason": "Quantités nulles, FK doubles",
    "mappings": [
        {"source_col":"line_id",        "target_col":"legacy_line_id","transform":"direct",       "confidence":0.99, "notes":"PK source"},
        {"source_col":"line_ord_id",    "target_col":"order_id",     "transform":"direct",       "confidence":0.95, "notes":"FK commande"},
        {"source_col":"line_prod_id",   "target_col":"product_id",   "transform":"direct",       "confidence":0.95, "notes":"FK produit"},
        {"source_col":"line_qty",       "target_col":"quantity",     "transform":"cast_numeric", "confidence":0.78, "notes":"Nulls → 1"},
        {"source_col":"line_unit_price","target_col":"unit_price",   "transform":"cast_numeric", "confidence":0.92, "notes":"Prix unitaire"},
        {"source_col":"line_discount",  "target_col":"discount_pct", "transform":"cast_numeric", "confidence":0.85, "notes":"% remise — nulls → 0"},
    ],
    "unmapped_source": ["line_total"],
    "unmapped_target": ["line_id","migrated_at"],
    "migration_risks": [
        "Quantités nulles (~2%) → remplacées par 1",
        "line_total colonne calculée côté cible"
    ]
},

"suppliers": {
    "table_source": "suppliers", "table_target": "suppliers",
    "migration_complexity": "low",
    "complexity_reason": "Table simple, peu de données",
    "mappings": [
        {"source_col":"sup_id",     "target_col":"legacy_sup_id", "transform":"direct",           "confidence":0.99, "notes":"PK source"},
        {"source_col":"sup_name",   "target_col":"name",          "transform":"direct",           "confidence":0.99, "notes":"Nom fournisseur"},
        {"source_col":"sup_country","target_col":"country_code",  "transform":"normalize_country","confidence":0.88, "notes":"Pays ISO 2"},
        {"source_col":"sup_contact","target_col":"contact_name",  "transform":"direct",           "confidence":0.97, "notes":"Nom contact"},
        {"source_col":"sup_email",  "target_col":"email",         "transform":"direct",           "confidence":0.95, "notes":"Email"},
        {"source_col":"sup_phone",  "target_col":"phone",         "transform":"normalize_phone",  "confidence":0.77, "notes":"Formats mixtes"},
        {"source_col":"sup_rating", "target_col":"rating",        "transform":"direct",           "confidence":0.95, "notes":"Note 1-5"},
        {"source_col":"sup_active", "target_col":"is_active",     "transform":"normalize_bool",   "confidence":0.80, "notes":"Y/N/1/0 → BOOLEAN"},
        {"source_col":"sup_since",  "target_col":"partner_since", "transform":"normalize_date",   "confidence":0.75, "notes":"Dates mixtes"},
    ],
    "unmapped_source": [],
    "unmapped_target": ["supplier_id","migrated_at"],
    "migration_risks": [
        "Téléphones en formats mixtes",
        "~5% de pays manquants"
    ]
}

}

# Génération des fichiers
print("Génération des mappings JSON...")
for table, mapping in MAPPINGS.items():
    path = OUTPUT / f"{table}_mapping.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✅ {path.name}")

print(f"\n✅ {len(MAPPINGS)} fichiers générés dans outputs/mappings/")
print("Tu peux maintenant relancer : python agents/migration_executor.py")
