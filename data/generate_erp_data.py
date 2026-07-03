"""
SmartMigrate — ERP Legacy Dataset Generator
Simule une vraie base ERP legacy avec des problèmes réels :
- Données manquantes
- Formats incohérents (dates, téléphones)
- Doublons
- Valeurs aberrantes
- Colonnes mal nommées (legacy naming)
- Données encodées en chaîne (ex: statut "1" au lieu de "active")
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import sqlite3
import os
from datetime import datetime, timedelta

fake = Faker(['fr_FR', 'en_US'])
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/raw"
DB_PATH    = OUTPUT_DIR + "/erp_legacy.db"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def rand_date(start="2018-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    return s + timedelta(days=random.randint(0, (e - s).days))

def messy_phone():
    """Génère des téléphones dans des formats incohérents (legacy réaliste)"""
    formats = [
        lambda: f"+216 {random.randint(20,99)} {random.randint(100,999)} {random.randint(100,999)}",
        lambda: f"00216{random.randint(20000000,99999999)}",
        lambda: f"{random.randint(20,99)}-{random.randint(100,999)}-{random.randint(100,999)}",
        lambda: str(random.randint(20000000, 99999999)),
        lambda: None,  # manquant
    ]
    return random.choices(formats, weights=[30, 20, 20, 20, 10])[0]()

def messy_date(d):
    """Formats de dates incohérents dans les vieux ERP"""
    if d is None or random.random() < 0.03:
        return None
    formats = [
        d.strftime("%Y-%m-%d"),
        d.strftime("%d/%m/%Y"),
        d.strftime("%m-%d-%Y"),
        d.strftime("%d-%m-%Y"),
        d.strftime("%Y%m%d"),
    ]
    return random.choices(formats, weights=[50, 25, 10, 10, 5])[0]

# ── 1. CLIENTS (cst_ prefix = legacy naming convention) ──────────────────────

def generate_customers(n=500):
    rows = []
    for i in range(1, n + 1):
        is_company  = random.random() < 0.4
        first       = fake.first_name() if not is_company else None
        last        = fake.last_name()  if not is_company else None
        company     = fake.company()    if is_company     else (fake.company() if random.random() < 0.2 else None)
        created     = rand_date("2018-01-01", "2023-12-31")
        country_raw = random.choices(
            ["TN", "Tunisia", "Tunisie", "FR", "France", "DZ", "MA", "DE", "US"],
            weights=[25, 10, 10, 15, 5, 10, 8, 7, 10]
        )[0]
        status_raw  = random.choices(["1", "0", "A", "I", "active", "inactive", None],
                                     weights=[35, 15, 20, 10, 10, 5, 5])[0]

        rows.append({
            "cst_id":         i,
            "cst_fname":      first,
            "cst_lname":      last,
            "cst_company":    company,
            "cst_email":      fake.email() if random.random() > 0.05 else None,
            "cst_phone":      messy_phone(),
            "cst_country":    country_raw,
            "cst_city":       fake.city(),
            "cst_address":    fake.address().replace("\n", ", ") if random.random() > 0.1 else None,
            "cst_status":     status_raw,
            "cst_segment":    random.choice(["B2B", "B2C", "B2B", "B2C", "VIP", None]),
            "cst_created_dt": messy_date(created),
            "cst_notes":      fake.sentence() if random.random() < 0.2 else None,
        })

    # Injecter des doublons réalistes (~3%)
    n_dup = int(n * 0.03)
    dupes = random.sample(rows, n_dup)
    for d in dupes:
        dup = d.copy()
        dup["cst_id"]    = n + rows.index(d) + 1
        dup["cst_email"] = d["cst_email"]          # même email = doublon
        dup["cst_phone"] = messy_phone()            # téléphone légèrement différent
        rows.append(dup)

    return pd.DataFrame(rows)

# ── 2. PRODUITS ───────────────────────────────────────────────────────────────

CATEGORIES = ["Informatique", "Fournitures", "Mobilier", "Logiciels", "Services", "Électronique"]

def generate_products(n=150):
    rows = []
    for i in range(1, n + 1):
        cat   = random.choice(CATEGORIES)
        price = round(random.uniform(5, 5000), random.choice([0, 2]))
        rows.append({
            "prod_id":       i,
            "prod_ref":      f"REF-{cat[:3].upper()}-{i:04d}",
            "prod_name":     fake.catch_phrase()[:60],
            "prod_category": cat if random.random() > 0.05 else None,
            "prod_price":    price if random.random() > 0.02 else None,   # quelques prix manquants
            "prod_currency": random.choices(["TND", "EUR", "USD", "tnd", "eur"], weights=[50,25,15,7,3])[0],
            "prod_stock":    random.randint(0, 500) if random.random() > 0.04 else -1,  # stock négatif (bug legacy)
            "prod_unit":     random.choice(["pièce", "pcs", "unit", "box", "kg", None]),
            "prod_active":   random.choices(["Y", "N", "1", "0", "yes", "no"], weights=[50,15,15,8,7,5])[0],
            "prod_created":  messy_date(rand_date("2017-01-01", "2023-01-01")),
        })
    return pd.DataFrame(rows)

# ── 3. COMMANDES ──────────────────────────────────────────────────────────────

def generate_orders(customers_df, n=2000):
    cst_ids = customers_df["cst_id"].tolist()
    rows    = []

    # Statuts encodés legacy
    status_map = {
        "pending":   ["P", "PEND", "pending", "0"],
        "confirmed": ["C", "CONF", "confirmed", "1"],
        "shipped":   ["S", "SHIP", "shipped", "2"],
        "delivered": ["D", "DELIV", "delivered", "3"],
        "cancelled": ["X", "CANC", "cancelled", "9"],
    }
    flat_statuses, weights_st = [], []
    for k, vals in status_map.items():
        w = {"pending":10,"confirmed":20,"shipped":15,"delivered":45,"cancelled":10}[k]
        for v in vals:
            flat_statuses.append(v)
            weights_st.append(w / len(vals))

    for i in range(1, n + 1):
        order_date    = rand_date("2019-01-01", "2024-06-30")
        expected_days = random.randint(1, 14)
        delivered     = order_date + timedelta(days=expected_days + random.randint(-2, 10))
        total         = round(random.uniform(20, 15000), 2)

        rows.append({
            "ord_id":          i,
            "ord_ref":         f"ORD-{order_date.year}-{i:05d}",
            "ord_cst_id":      random.choice(cst_ids),
            "ord_date":        messy_date(order_date),
            "ord_exp_deliver": messy_date(order_date + timedelta(days=expected_days)),
            "ord_delivered":   messy_date(delivered) if random.random() > 0.15 else None,
            "ord_status":      random.choices(flat_statuses, weights=weights_st)[0],
            "ord_total_ht":    total if random.random() > 0.01 else None,
            "ord_tva":         random.choices([19.0, 7.0, 0.0, None], weights=[60, 20, 15, 5])[0],
            "ord_channel":     random.choices(["WEB", "TEL", "STORE", "API", None], weights=[40,25,25,5,5])[0],
            "ord_notes":       fake.sentence() if random.random() < 0.1 else None,
        })

    return pd.DataFrame(rows)

# ── 4. LIGNES DE COMMANDE ─────────────────────────────────────────────────────

def generate_order_lines(orders_df, products_df):
    ord_ids  = orders_df["ord_id"].tolist()
    prod_ids = products_df["prod_id"].tolist()
    rows     = []
    line_id  = 1

    for oid in ord_ids:
        n_lines = random.choices([1,2,3,4,5], weights=[40,30,15,10,5])[0]
        prods   = random.sample(prod_ids, min(n_lines, len(prod_ids)))
        for pid in prods:
            qty        = random.randint(1, 20)
            unit_price = round(random.uniform(5, 3000), 2)
            rows.append({
                "line_id":        line_id,
                "line_ord_id":    oid,
                "line_prod_id":   pid,
                "line_qty":       qty if random.random() > 0.02 else None,
                "line_unit_price":unit_price,
                "line_discount":  random.choices([0, 5, 10, 15, 20, None], weights=[50,15,15,10,5,5])[0],
                "line_total":     round(qty * unit_price, 2) if random.random() > 0.03 else None,
            })
            line_id += 1

    return pd.DataFrame(rows)

# ── 5. FOURNISSEURS ───────────────────────────────────────────────────────────

def generate_suppliers(n=50):
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "sup_id":      i,
            "sup_name":    fake.company(),
            "sup_country": random.choices(["TN","CN","DE","FR","IT","ES","US","TR"], weights=[20,20,10,15,10,10,10,5])[0],
            "sup_contact": fake.name() if random.random() > 0.1 else None,
            "sup_email":   fake.company_email() if random.random() > 0.08 else None,
            "sup_phone":   messy_phone(),
            "sup_rating":  random.choices([1,2,3,4,5,None], weights=[5,10,20,35,25,5])[0],
            "sup_active":  random.choices(["Y","N","1","0"], weights=[70,10,15,5])[0],
            "sup_since":   messy_date(rand_date("2010-01-01","2022-12-31")),
        })
    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🏭 Génération du dataset ERP legacy...")

    customers   = generate_customers(500)
    products    = generate_products(150)
    orders      = generate_orders(customers, 2000)
    order_lines = generate_order_lines(orders, products)
    suppliers   = generate_suppliers(50)

    # Sauvegarde CSV (pour inspection)
    customers.to_csv(f"{OUTPUT_DIR}/erp_customers.csv",   index=False, encoding="utf-8")
    products.to_csv(f"{OUTPUT_DIR}/erp_products.csv",     index=False, encoding="utf-8")
    orders.to_csv(f"{OUTPUT_DIR}/erp_orders.csv",         index=False, encoding="utf-8")
    order_lines.to_csv(f"{OUTPUT_DIR}/erp_order_lines.csv", index=False, encoding="utf-8")
    suppliers.to_csv(f"{OUTPUT_DIR}/erp_suppliers.csv",   index=False, encoding="utf-8")

    # Sauvegarde SQLite (source ERP simulée)
    conn = sqlite3.connect(DB_PATH)
    customers.to_sql("customers",   conn, if_exists="replace", index=False)
    products.to_sql("products",     conn, if_exists="replace", index=False)
    orders.to_sql("orders",         conn, if_exists="replace", index=False)
    order_lines.to_sql("order_lines", conn, if_exists="replace", index=False)
    suppliers.to_sql("suppliers",   conn, if_exists="replace", index=False)
    conn.close()

    # Rapport de synthèse
    print("\n✅ Dataset généré avec succès !\n")
    print("=" * 55)
    tables = {
        "customers":   customers,
        "products":    products,
        "orders":      orders,
        "order_lines": order_lines,
        "suppliers":   suppliers,
    }
    for name, df in tables.items():
        nulls = df.isnull().sum().sum()
        print(f"  📋 {name:<15} {len(df):>5} lignes  |  {nulls:>4} valeurs nulles")

    print("=" * 55)
    print(f"\n📁 Fichiers CSV  → {OUTPUT_DIR}/")
    print(f"🗄️  Base SQLite   → {DB_PATH}")
    print("\n⚠️  Problèmes intentionnels injectés :")
    print("  • Formats de dates mixtes (ISO, FR, US, compact)")
    print("  • Téléphones incohérents (4 formats différents)")
    print("  • Statuts encodés legacy (1/0, Y/N, text, code)")
    print("  • ~3% de doublons clients (même email)")
    print("  • Stocks négatifs (bug legacy)")
    print("  • Pays non normalisés (TN / Tunisia / Tunisie)")
    print("  • Prix et quantités manquants (~1-4%)")
    print("  • Nommage legacy (cst_, ord_, prod_, sup_)")

if __name__ == "__main__":
    main()
