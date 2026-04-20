import difflib
import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present.
        # Parse metadata first into a separate Series, then assign all derived columns
        # in one .assign() call to avoid pandas Copy-on-Write FutureWarnings.
        if "request_metadata" in quotes_df.columns:
            parsed_metadata = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df = quotes_df.assign(
                request_metadata=parsed_metadata,
                job_type=parsed_metadata.apply(lambda x: x.get("job_type", "")),
                order_size=parsed_metadata.apply(lambda x: x.get("order_size", "")),
                event_type=parsed_metadata.apply(lambda x: x.get("event_type", "")),
            )

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# ── Imports ──────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# ── Environment & Model Setup ─────────────────────────────────────────────────
dotenv.load_dotenv()
_api_key = os.getenv("UDACITY_OPENAI_API_KEY")
_model = OpenAIModel(
    "gpt-4o-mini",
    provider=OpenAIProvider(
        base_url="https://openai.vocareum.com/v1",
        api_key=_api_key,
    ),
)


# ── Shared State ──────────────────────────────────────────────────────────────
@dataclass
class SystemState:
    """Shared context threaded through all agent calls via RunContext.deps.

    Attributes:
        db_engine: SQLAlchemy Engine backed by the SQLite munder_difflin.db file.
        current_date: ISO-format date string from the CSV request row. Must
            always be set from source data — never from datetime.now().
        session_log: Accumulated agent messages for the current request cycle.
    """
    db_engine: Engine
    current_date: str
    session_log: list[str] = field(default_factory=list)


# ── Catalog: exact item names required for all DB transactions ────────────────
CATALOG_ITEMS: list[str] = [item["item_name"] for item in paper_supplies]

# Formatted catalog list injected into agent system prompts for exact-name guidance
_CATALOG_NAMES = "\n".join(f"  • {name}" for name in CATALOG_ITEMS)


def resolve_item_name(customer_term: str) -> str | None:
    """Resolve a customer's informal item description to the canonical catalog name.

    Resolution order:
        1. Exact match (case-sensitive)
        2. Case-insensitive exact match
        3. Fuzzy match via difflib.get_close_matches on lowercased names (cutoff=0.6)
        4. Substring match (customer_term is contained within a catalog name, case-insensitive)

    Args:
        customer_term: The item description provided by the customer.

    Returns:
        The canonical CATALOG_ITEMS name, or None if no reasonable match is found.
    """
    if not customer_term:
        return None

    # 1. Exact match (case-sensitive)
    if customer_term in CATALOG_ITEMS:
        return customer_term

    # 2. Case-insensitive exact match
    lower_term = customer_term.lower()
    for item in CATALOG_ITEMS:
        if item.lower() == lower_term:
            return item

    # 3. Fuzzy match (case-insensitive) — compare lowercased names
    lower_catalog = [item.lower() for item in CATALOG_ITEMS]
    matches = difflib.get_close_matches(lower_term, lower_catalog, n=1, cutoff=0.6)
    if matches:
        return CATALOG_ITEMS[lower_catalog.index(matches[0])]

    # 4. Substring match — customer_term appears inside a catalog name
    for item in CATALOG_ITEMS:
        if lower_term in item.lower():
            return item

    return None


# ── Agent Definitions ─────────────────────────────────────────────────────────

inventory_agent = Agent(
    model=_model,
    deps_type=SystemState,
    system_prompt=f"""You are the Inventory Agent for Munder Difflin Paper Company.

Your role: check current stock levels for requested items and, after a sale,
auto-reorder items that fall below their minimum stock level.

AVAILABLE CATALOG ITEMS (use ONLY these exact names in every tool call):
{_CATALOG_NAMES}

STOCK CHECK RULES — for every item in the customer request:
  a. If the item does not appear in the catalog list above:
       → write "INSUFFICIENT STOCK: [item] — not in our catalog"
  b. If check_inventory_tool returns an error (item not stocked):
       → write "INSUFFICIENT STOCK: [item] — not currently stocked"
  c. If current_stock < requested quantity:
       → write "INSUFFICIENT STOCK: [item] — only N units available (requested M)"
  d. If stock is adequate:
       → write "IN STOCK: [item] — N units available"

Use the exact phrase "INSUFFICIENT STOCK" in ALL cases where an item cannot be
fully supplied. This phrase is parsed by downstream agents — be consistent.

POST-SALE REORDER CHECK (when instructed after a sale):
  Call check_inventory_tool for each sold item.
  If current_stock < min_stock_level, call reorder_tool with quantity = 2 × min_stock_level.
  Report which items were reordered and their estimated arrival dates.""",
)

quoting_agent = Agent(
    model=_model,
    deps_type=SystemState,
    system_prompt=f"""You are the Quoting Agent for Munder Difflin Paper Company.

Your role: generate accurate, detailed price quotes using volume discount tiers.

AVAILABLE CATALOG ITEMS (use ONLY these exact names in every tool call):
{_CATALOG_NAMES}

VOLUME DISCOUNT TIERS:
  Fewer than 100 units  →  0 % (no discount)
  100 – 499 units       →  5 % off
  500 – 999 units       → 10 % off
  1,000 or more units   → 15 % off

DISCOUNT CALCULATION — apply in this exact order:
  Step 1: Determine tier by checking quantity against these boundaries in order:
          Is quantity >= 1000?  → 15% off
          Is quantity >= 500?   → 10% off
          Is quantity >= 100?   → 5%  off
          Otherwise             → 0%  off  (only if quantity < 100)
  Step 2: discounted_unit_price = base_unit_price × (1 − discount_rate)
  Step 3: line_total = round(quantity × discounted_unit_price, 2)

  Examples (check your own quantities against these):
    50 units  → < 100            → 0%  off  → A4 $0.05 × 1.00 = $0.0500/unit
    150 units → >= 100, < 500    → 5%  off  → A4 $0.05 × 0.95 = $0.0475/unit
    207 units → >= 100, < 500    → 5%  off  → Glossy $0.20 × 0.95 = $0.19/unit
    401 units → >= 100, < 500    → 5%  off  → Cardstock $0.15 × 0.95 = $0.1425/unit
    500 units → >= 500, < 1000   → 10% off  → Glossy $0.20 × 0.90 = $0.18/unit
    1500 units → >= 1000         → 15% off  → A4 $0.05 × 0.85 = $0.0425/unit

QUOTING RULES:
1. Call get_inventory_price_tool for EACH item before quoting.
   If it returns an error → do NOT invent a price. State the item cannot be quoted.
2. Call search_quote_history_tool for context only — it never overrides the tiers above.
3. Only quote items that are in the CATALOG and returned a valid price from the tool.
4. If inventory context shows "INSUFFICIENT STOCK" for an item, exclude it from
   the quote and note it cannot be provided at the requested quantity.
5. Every quoted item MUST show: catalog name, quantity, base unit price, tier applied
   (e.g. "5 % — 100–499 units"), discounted unit price, line total (2 decimal places).""",
)

sales_agent = Agent(
    model=_model,
    deps_type=SystemState,
    system_prompt=f"""You are the Sales Agent for Munder Difflin Paper Company.

Your role: finalize confirmed orders — record transactions, provide delivery
estimates, and report the updated financial balance.

AVAILABLE CATALOG ITEMS (use ONLY these exact names in every tool call):
{_CATALOG_NAMES}

FULFILLMENT RULES:
1. Review the approved quote. It lists ONLY items with valid prices (INSUFFICIENT
   STOCK items were already excluded by the quoting agent).
2. Call fulfill_order_tool for EACH item that appears in the approved quote with a
   valid discounted price. Process them one at a time if there are multiple items.
   Never invent items or prices not explicitly listed in the approved quote.
3. If the approved quote contains NO valid priced items, do NOT call
   fulfill_order_tool at all. Your response MUST include: "No transaction was recorded."
4. After fulfilling all available items, call get_delivery_date_tool using the
   total quantity fulfilled, then call get_balance_tool.
5. LANGUAGE RULE — this is critical:
   • Use confirmation language ("we have processed your order", "total charged")
     ONLY for items where fulfill_order_tool was successfully called.
   • For items not in the approved quote (INSUFFICIENT STOCK), use rejection
     language in the same response.
   • If fulfill_order_tool was NOT called at all, use rejection language only —
     never imply any transaction occurred.""",
)

orchestrator_agent = Agent(
    model=_model,
    deps_type=SystemState,
    system_prompt="""You are the Orchestrator for Munder Difflin Paper Company.

MODE 1 — INTENT CLASSIFICATION:
Reply with EXACTLY one of these words and nothing else:
  inventory_query    (asking about stock, availability, or what we carry)
  quote_request      (wants pricing or a quote — no confirmed purchase)
  order_fulfillment  (placing an order, buying, or confirming a purchase)
  unknown            (cannot determine intent)

MODE 2 — RESPONSE COMPOSITION:
Write a professional, friendly reply using the internal context provided.

COMPOSITION RULES — follow these strictly:
1. CONFIRMED SALE: Only use confirmation language if the sale context explicitly
   states that fulfill_order_tool was called (e.g. transaction ID, total charged,
   delivery date). If the sale context says "No transaction was recorded" or is
   blank, write a polite rejection — never a confirmation.
2. PARTIAL FULFILLMENT: If some items were fulfilled and others were not, clearly
   separate what was processed from what was declined, with reasons for each.
3. FULL REJECTION: If inventory shows "INSUFFICIENT STOCK" for all items or the
   sale context says no transaction occurred, explain which items are unavailable
   and suggest the customer contact us to adjust their order.
4. QUOTE ONLY: For quote_request intent, present the pricing with discount tier
   explained, but do not imply an order has been placed.
5. Never expose error messages, stack traces, profit margins, or raw DB fields.""",
)

upsell_agent = Agent(
    model=_model,
    deps_type=SystemState,
    system_prompt=f"""You are the Upsell Agent for Munder Difflin Paper Company.

Your role: after a customer request has been processed, suggest 1-3 complementary
catalog items the customer did NOT already request — grounded in real sales data
and historical quote context, not guesswork.

AVAILABLE CATALOG ITEMS (suggest ONLY items from this list — never invent names):
{_CATALOG_NAMES}

STEPS:
1. Call get_top_sellers_tool to see which items are generating the most revenue.
2. Call upsell_search_quotes_tool with 1-2 keywords from the customer's event type
   or job context (e.g., "ceremony", "conference", "marketing", "party") to find
   what similar customers have ordered.
3. From these signals, choose 1-3 items that:
   a. Appear in the catalog list above
   b. Are NOT already mentioned in the customer's current order or quote context
   c. Are logically complementary — e.g., Presentation folders with glossy paper,
      Paper cups with Paper plates, Sticky notes with Notepads, Flyers with Poster paper

OUTPUT FORMAT — write a short, friendly section with no prices:
  You may also be interested in:
  • [Exact catalog item name] — [one sentence: why it fits their event or order]
  • ...

RULES:
- Use ONLY exact item names from the catalog list above. Never invent or paraphrase.
- Do NOT mention prices, discounts, or delivery dates — those belong to other agents.
- Do NOT suggest any item the customer already requested or that appears in the quote.
- Maximum 3 suggestions. If context is too thin, give 1 best-seller recommendation.
- If the request was fully rejected (nothing ordered), frame suggestions as alternatives
  the customer may wish to explore instead.""",
)


# ── Tools: InventoryAgent ─────────────────────────────────────────────────────

@inventory_agent.tool
def check_inventory_tool(ctx: RunContext[SystemState], item_name: str) -> dict:
    """Return current stock level for a named catalog item.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).
        item_name: Exact catalog name of the item to look up.

    Returns:
        Dict with item_name, current_stock, min_stock_level, unit_price, in_stock.
    """
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return {"error": f"Item '{item_name}' not found in catalog."}

    # Current stock from transaction history
    stock_df = get_stock_level(resolved, ctx.deps.current_date)
    current_stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    # Unit price and reorder threshold from the inventory reference table
    with ctx.deps.db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT unit_price, min_stock_level FROM inventory WHERE item_name = :name"),
            {"name": resolved},
        ).fetchone()

    if row is None:
        return {"error": f"Item '{resolved}' is not currently stocked."}

    return {
        "item_name": resolved,
        "current_stock": current_stock,
        "min_stock_level": int(row[1]),
        "unit_price": float(row[0]),
        "in_stock": current_stock > 0,
    }


@inventory_agent.tool
def reorder_tool(ctx: RunContext[SystemState], item_name: str, quantity: int) -> dict:
    """Place a stock_orders transaction to replenish a low-stock item.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).
        item_name: Exact catalog name of the item to reorder.
        quantity: Number of units to order from the supplier.

    Returns:
        Dict with item_name, quantity_ordered, estimated_arrival, transaction_id.
    """
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return {"error": f"Item '{item_name}' not found in catalog."}

    # Look up supplier cost per unit from the inventory reference table
    with ctx.deps.db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT unit_price FROM inventory WHERE item_name = :name"),
            {"name": resolved},
        ).fetchone()

    if row is None:
        return {"error": f"Item '{resolved}' is not currently stocked."}

    unit_price = float(row[0])
    total_cost = round(quantity * unit_price, 2)

    # Record a stock_orders transaction (reduces cash balance)
    transaction_id = create_transaction(
        item_name=resolved,
        transaction_type="stock_orders",
        quantity=quantity,
        price=total_cost,
        date=ctx.deps.current_date,
    )

    # Estimate when the supplier will deliver
    estimated_arrival = get_supplier_delivery_date(ctx.deps.current_date, quantity)

    return {
        "item_name": resolved,
        "quantity_ordered": quantity,
        "estimated_arrival": estimated_arrival,
        "transaction_id": transaction_id,
        "unit_price": unit_price,
        "total_cost": total_cost,
    }


# ── Tools: QuotingAgent ───────────────────────────────────────────────────────

@quoting_agent.tool
def search_quote_history_tool(ctx: RunContext[SystemState], search_terms: str) -> list:
    """Retrieve historical quotes matching search terms for pricing context.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).
        search_terms: Space-separated keywords to search across quote history.

    Returns:
        List of matching quote dicts (original_request, total_amount, etc.).
    """
    # Split the space-separated string into individual search terms
    terms = [t.strip() for t in search_terms.split() if t.strip()]
    return search_quote_history(terms, limit=5)


@quoting_agent.tool
def get_inventory_price_tool(ctx: RunContext[SystemState], item_name: str) -> dict:
    """Return unit price and available quantity for a catalog item.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).
        item_name: Exact catalog name of the item to price.

    Returns:
        Dict with item_name, unit_price, current_stock.
    """
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return {"error": f"Item '{item_name}' not found in catalog."}

    # Unit price from the inventory reference table
    with ctx.deps.db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT unit_price FROM inventory WHERE item_name = :name"),
            {"name": resolved},
        ).fetchone()

    if row is None:
        return {"error": f"Item '{resolved}' is not currently stocked."}

    unit_price = float(row[0])

    # Current stock derived from transaction history
    all_inventory = get_all_inventory(ctx.deps.current_date)
    current_stock = int(all_inventory.get(resolved, 0))

    return {
        "item_name": resolved,
        "unit_price": unit_price,
        "current_stock": current_stock,
    }


# ── Tools: SalesAgent ─────────────────────────────────────────────────────────

@sales_agent.tool
def fulfill_order_tool(
    ctx: RunContext[SystemState],
    item_name: str,
    quantity: int,
    unit_price: float,
) -> dict:
    """Record a completed sale as a sales transaction in the database.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).
        item_name: Exact catalog name of the sold item.
        quantity: Number of units sold.
        unit_price: Discounted unit price agreed in the quote.

    Returns:
        Dict with transaction_id, item_name, quantity, total_price.
    """
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return {"error": f"Item '{item_name}' not found in catalog."}

    total_price = round(quantity * unit_price, 2)

    # Record the sale in the transactions table
    transaction_id = create_transaction(
        item_name=resolved,
        transaction_type="sales",
        quantity=quantity,
        price=total_price,
        date=ctx.deps.current_date,
    )

    return {
        "transaction_id": transaction_id,
        "item_name": resolved,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
    }


@sales_agent.tool
def get_delivery_date_tool(ctx: RunContext[SystemState], quantity: int) -> str:
    """Return the estimated supplier delivery date for a given order quantity.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).
        quantity: Number of units being ordered/delivered.

    Returns:
        ISO date string of estimated delivery.
    """
    return get_supplier_delivery_date(ctx.deps.current_date, quantity)


@sales_agent.tool
def get_balance_tool(ctx: RunContext[SystemState]) -> dict:
    """Return current cash balance and a financial report summary.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).

    Returns:
        Dict with cash_balance, inventory_value, total_assets, and top_selling_products.
    """
    date = ctx.deps.current_date
    report = generate_financial_report(date)
    return {
        "cash_balance": float(report["cash_balance"]),
        "inventory_value": float(report["inventory_value"]),
        "total_assets": float(report["total_assets"]),
        "top_selling_products": report.get("top_selling_products", []),
    }


# ── Tools: UpsellAgent ───────────────────────────────────────────────────────

@upsell_agent.tool
def get_top_sellers_tool(ctx: RunContext[SystemState]) -> list:
    """Return the top 5 best-selling catalog items by revenue as of the current date.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).

    Returns:
        List of dicts with item_name, total_units, and total_revenue fields.
    """
    query = """
        SELECT item_name,
               SUM(units)  AS total_units,
               SUM(price)  AS total_revenue
        FROM   transactions
        WHERE  transaction_type = 'sales'
          AND  item_name IS NOT NULL
          AND  transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    result = pd.read_sql(query, ctx.deps.db_engine, params={"date": ctx.deps.current_date})
    return result.to_dict(orient="records")


@upsell_agent.tool
def upsell_search_quotes_tool(ctx: RunContext[SystemState], search_terms: str) -> list:
    """Retrieve historical quotes matching search terms to ground upsell suggestions.

    Args:
        ctx: RunContext carrying SystemState (db_engine, current_date).
        search_terms: Space-separated keywords (e.g., event type or job context).

    Returns:
        List of matching quote dicts (original_request, total_amount, etc.).
    """
    terms = [t.strip() for t in search_terms.split() if t.strip()]
    return search_quote_history(terms, limit=5)


# ── Discount Helper ───────────────────────────────────────────────────────────

def calculate_discount(quantity: int) -> float:
    """Return the applicable volume discount rate for a given order quantity.

    Discount tiers:
        < 100 units   → 0%  (no discount)
        100–499 units → 5%
        500–999 units → 10%
        ≥ 1000 units  → 15%

    Args:
        quantity: Number of units in the order.

    Returns:
        Discount rate as a float (e.g., 0.10 for 10%).
    """
    if quantity >= 1000:
        return 0.15
    elif quantity >= 500:
        return 0.10
    elif quantity >= 100:
        return 0.05
    return 0.0


# ── Orchestration ─────────────────────────────────────────────────────────────

class PaperCompanySystem:
    """Orchestrates the Munder Difflin multi-agent pipeline.

    Classifies incoming customer requests by intent and sequences
    the appropriate worker agents (Inventory, Quoting, Sales) to
    produce a customer-facing text response.
    """

    def process_request(self, request: str, date: str) -> str:
        """Process a single customer request through the agent pipeline.

        Args:
            request: Raw customer request text.
            date: ISO date string for this request (from CSV request_date column).
                  Must come from source data, never from datetime.now().

        Returns:
            Customer-facing response string.

        Raises:
            ValueError: If date is empty or None.
        """
        if not date:
            raise ValueError("date must be provided — never default to today's date")

        state = SystemState(db_engine=db_engine, current_date=date)
        dated_request = f"{request}\n\n(Request date: {date})"

        # ── Step 1: Classify intent ───────────────────────────────────────────
        intent_result = orchestrator_agent.run_sync(
            f"Classify the intent of this customer request:\n{request}",
            deps=state,
        )
        raw_intent = intent_result.output.strip().lower()
        intent = "unknown"
        for candidate in ("inventory_query", "quote_request", "order_fulfillment"):
            if candidate in raw_intent:
                intent = candidate
                break
        # Store intent as first log entry so callers can inspect it
        state.session_log.append(f"[intent] {intent}")
        self._last_intent = intent  # surfaced for harness CSV capture

        inventory_ctx = quote_ctx = sale_ctx = ""

        # ── Step 2: Inventory check (all recognised intents) ─────────────────
        if intent != "unknown":
            inv = inventory_agent.run_sync(
                f"Check inventory availability for this customer request:\n{dated_request}",
                deps=state,
            )
            inventory_ctx = inv.output
            state.session_log.append(f"[inventory] {inventory_ctx[:300]}")

        # ── Step 3: Quote (quote and order intents) ───────────────────────────
        if intent in ("quote_request", "order_fulfillment"):
            qt = quoting_agent.run_sync(
                f"Generate a price quote for this customer request:\n{dated_request}"
                f"\n\n--- Inventory status ---\n{inventory_ctx}",
                deps=state,
            )
            quote_ctx = qt.output
            state.session_log.append(f"[quote] {quote_ctx[:300]}")

        # ── Step 4: Sales fulfillment (order intent only) ─────────────────────
        # No Python-level gate here — the QuotingAgent already excluded
        # INSUFFICIENT STOCK items from the approved quote, so the SalesAgent
        # only sees valid priced items. It fulfils what it can and reports
        # "No transaction was recorded" when nothing is available.
        if intent == "order_fulfillment":
            sale = sales_agent.run_sync(
                f"Fulfill this confirmed order:\n{dated_request}"
                f"\n\n--- Inventory status ---\n{inventory_ctx}"
                f"\n\n--- Approved quote ---\n{quote_ctx}",
                deps=state,
            )
            sale_ctx = sale.output
            state.session_log.append(f"[sale] {sale_ctx[:300]}")

            # Auto-reorder: check if sold items fell below min_stock_level
            reorder = inventory_agent.run_sync(
                f"A sale was just completed for the following order:\n{dated_request}"
                f"\n\nFor each item in that order: call check_inventory_tool, and if "
                f"current_stock < min_stock_level, call reorder_tool with "
                f"quantity = 2 × min_stock_level.",
                deps=state,
            )
            reorder_ctx = reorder.output
            state.session_log.append(f"[reorder] {reorder_ctx[:300]}")
            if reorder_ctx:
                sale_ctx += f"\n\n[Reorder status: {reorder_ctx}]"

        # ── Step 5: Compose customer-facing response ──────────────────────────
        # Derive an explicit transaction status so the orchestrator cannot
        # mistake a quote-only context for a confirmed sale.
        if intent == "order_fulfillment":
            if sale_ctx and "no transaction was recorded" not in sale_ctx.lower() \
                    and "could not be fulfilled" not in sale_ctx.lower() \
                    and sale_ctx != "Order could not be fulfilled: one or more requested items have insufficient stock.":
                txn_status = "TRANSACTION STATUS: A sale transaction WAS recorded. Use confirmation language."
            else:
                txn_status = "TRANSACTION STATUS: NO transaction was recorded. Use rejection language — do NOT say the order is confirmed."
        else:
            txn_status = "TRANSACTION STATUS: Not an order — no transaction expected."

        compose_prompt = (
            f"Compose a professional customer-facing response for "
            f"Munder Difflin Paper Company.\n\n"
            f"Customer request: {request}\n"
            f"Request date: {date}\n"
            f"Intent classified: {intent}\n"
            f"{txn_status}\n\n"
            f"--- Inventory status (internal context) ---\n{inventory_ctx}\n\n"
            f"--- Quote details (internal context) ---\n{quote_ctx}\n\n"
            f"--- Order / sale result (internal context) ---\n{sale_ctx}"
        )
        final = orchestrator_agent.run_sync(compose_prompt, deps=state)
        final_response = final.output

        # ── Step 6: Upsell pitch (quote and order intents only) ──────────────
        # Fires after the main response is composed so it never interferes with
        # the transaction/rejection logic.  Advisory only — no tools that write DB.
        if intent in ("order_fulfillment", "quote_request"):
            upsell_prompt = (
                f"A customer request has just been processed.\n"
                f"Customer request: {request}\n"
                f"Date: {date}\n\n"
                f"--- Items already discussed (do NOT suggest these again) ---\n"
                f"{quote_ctx if quote_ctx else inventory_ctx}\n\n"
                f"Call get_top_sellers_tool first to see what sells best, then "
                f"suggest 1-3 complementary catalog items the customer has NOT "
                f"already asked about."
            )
            upsell = upsell_agent.run_sync(upsell_prompt, deps=state)
            upsell_ctx = upsell.output.strip()
            state.session_log.append(f"[upsell] {upsell_ctx[:200]}")
            if upsell_ctx:
                final_response += f"\n\n---\n{upsell_ctx}"

        return final_response


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    """Run all 20 scenarios from quote_requests_sample.csv through the multi-agent pipeline.

    Initialises a fresh database, processes each request row in chronological order,
    captures the intent classification and any exceptions per row, and writes the full
    results to test_results.csv.  Prints a financial summary to stdout on completion.

    Returns:
        list[dict]: One dict per request with keys request_id, request_date, intent,
                    cash_balance, inventory_value, response, and error.
    """
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    system = PaperCompanySystem()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try:
            response = system.process_request(row["request"], request_date)
            intent = getattr(system, "_last_intent", "unknown")
            error = ""
        except Exception as e:
            response = f"[ERROR] {type(e).__name__}: {e}"
            intent = "error"
            error = str(e)
            print(f"  WARNING: request {idx+1} raised {type(e).__name__}: {e}")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Intent:   {intent}")
        print(f"Response: {response[:200]}...")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "intent": intent,
                "cash_balance": round(current_cash, 2),
                "inventory_value": round(current_inventory, 2),
                "response": response,
                "error": error,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
