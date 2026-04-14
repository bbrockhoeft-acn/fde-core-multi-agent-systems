"""
Environment smoke test — verifies all 8 required helper functions work correctly.
Run from the project/ directory:  python smoke_test.py
No API key required — these are all database/file operations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from project_starter import (
    db_engine,
    init_database,
    get_all_inventory,
    get_cash_balance,
    search_quote_history,
    get_stock_level,
    get_supplier_delivery_date,
    create_transaction,
    generate_financial_report,
)

TEST_DATE = "2025-04-01"
PASS = "PASS"
FAIL = "FAIL"
results = []

def check(label, fn):
    try:
        result = fn()
        print(f"  [{PASS}] {label}")
        print(f"         -> {repr(result)[:120]}")
        results.append((label, True))
        return result
    except Exception as e:
        print(f"  [{FAIL}] {label}")
        print(f"         -> {e}")
        results.append((label, False))
        return None

print("\n=== Local Environment Smoke Test ===\n")

print("1. init_database()")
check("init_database", lambda: init_database(db_engine))

print("\n2. get_all_inventory()")
inv = check("get_all_inventory", lambda: get_all_inventory(TEST_DATE))

print("\n3. get_cash_balance()")
check("get_cash_balance", lambda: get_cash_balance(TEST_DATE))

print("\n4. search_quote_history()")
check("search_quote_history", lambda: search_quote_history("paper", limit=3))

print("\n5. get_stock_level()")
# Pick an item we know exists from the inventory
item = list(inv.keys())[0] if inv else "A4 paper"
check("get_stock_level", lambda: get_stock_level(item, TEST_DATE))

print("\n6. get_supplier_delivery_date()")
check("get_supplier_delivery_date — small (10 units)",  lambda: get_supplier_delivery_date(TEST_DATE, 10))
check("get_supplier_delivery_date — medium (100 units)", lambda: get_supplier_delivery_date(TEST_DATE, 100))
check("get_supplier_delivery_date — large (1000 units)", lambda: get_supplier_delivery_date(TEST_DATE, 1000))

print("\n7. create_transaction()")
tx_id = check(
    "create_transaction (sales)",
    lambda: create_transaction(item, "sales", 5, 2.50, TEST_DATE)
)

print("\n8. generate_financial_report()")
check("generate_financial_report", lambda: generate_financial_report(TEST_DATE))

print("\n=== Results ===")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("\n  All helpers confirmed working!")
else:
    print("\n  Some checks FAILED — review errors above before proceeding.")
