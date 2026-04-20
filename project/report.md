# Munder Difflin Paper Company — Multi-Agent System Reflection Report

> **Submission files**: `project_starter.py`, `diagram.svg`, `test_results.csv`, `report.md`  
> **Framework**: pydantic-ai v1.81.0  
> **Model**: gpt-4o-mini (via Vocareum OpenAI-compatible proxy)

---

## Section 1 — System Architecture

### Agent Workflow Overview

The system is built around four specialized agents orchestrated by a central coordinator (see `diagram.svg`). All agents are implemented using the **pydantic-ai** framework, which was selected for its clean type-safe tool definitions, native `RunContext` dependency injection, and straightforward `agent.run_sync()` invocation pattern. Each agent holds a shared `SystemState` dataclass through the `RunContext`, giving every tool access to the SQLite database engine and the current request date without requiring global state.

```
Customer Request → OrchestratorAgent → InventoryAgent
                                      → QuotingAgent (quote/order intents)
                                      → SalesAgent   (order intent only)
                                      ← all agents return structured context
                 → OrchestratorAgent composes final reply → Customer Response
```

### Agent Roles

**OrchestratorAgent** is the entry and exit point for every request. It runs twice per request: first in MODE 1 to classify the customer's intent into one of `inventory_query`, `quote_request`, `order_fulfillment`, or `unknown`; and then in MODE 2 to compose the final customer-facing reply from the structured context returned by the worker agents. The orchestrator holds no tools of its own — it is a pure coordinator.

**InventoryAgent** answers stock-level queries and enforces the auto-reorder policy. It exposes two tools: `check_inventory_tool` (wraps `get_stock_level` and `get_all_inventory`) and `reorder_tool` (wraps `create_transaction` and `get_supplier_delivery_date`). After every fulfilled order, the orchestrator invokes the InventoryAgent a second time to check whether sold items fell below their minimum stock level and to trigger replenishment at 2× the minimum quantity if so.

**QuotingAgent** generates accurate, itemized price quotes using the volume discount tier table. It exposes two tools: `get_inventory_price_tool` (retrieves unit price and current stock) and `search_quote_history_tool` (retrieves similar historical quotes for pricing context). The agent is instructed to apply discount tiers in a specific top-down order (≥1000 → ≥500 → ≥100 → otherwise) to prevent boundary errors.

**SalesAgent** finalizes confirmed orders by calling `fulfill_order_tool` (wraps `create_transaction('sales')`), then `get_delivery_date_tool`, and finally `get_balance_tool` for a post-sale financial audit. It is instructed to process only items that appear in the approved quote (items already excluded for insufficient stock are never passed for fulfillment), enabling correct partial-order handling within a single response.

### Key Architectural Decision — `txn_status` Signal

A significant challenge encountered during evaluation was the orchestrator composing confirmation-language responses even when no sale transaction had been recorded. The root cause was that the approved quote context naturally contains dollar amounts, and the LLM treated those figures as evidence of a completed purchase. The fix was to derive an explicit **machine-generated transaction status** in Python after Step 4 and inject it into the Step 5 compose prompt:

- `"TRANSACTION STATUS: A sale transaction WAS recorded. Use confirmation language."` — when `sale_ctx` does not contain rejection indicators
- `"TRANSACTION STATUS: NO transaction was recorded. Use rejection language."` — otherwise

This deterministic Python signal takes the ambiguity out of the LLM's hands and reduced false confirmations to zero across all 20 test requests.

### Pipeline Steps (per request)

1. **Classify intent** — OrchestratorAgent (MODE 1)
2. **Check inventory** — InventoryAgent (all recognized intents)
3. **Generate quote** — QuotingAgent (`quote_request` and `order_fulfillment` intents)
4. **Fulfill order** — SalesAgent (`order_fulfillment` only); followed by auto-reorder pass
5. **Compose reply** — OrchestratorAgent (MODE 2) with injected `txn_status`

### Date Integrity

Every tool call uses `ctx.deps.current_date`, which is populated from the CSV `request_date` column at the start of `process_request()`. The method raises `ValueError` immediately if `date` is empty or `None`, preventing any possibility of defaulting to the system clock. This satisfies the project's critical constraint that dates must always flow explicitly from source data through all agent calls.

---

## Section 2 — Evaluation Results

### Summary

The system was evaluated against all 20 requests in `quote_requests_sample.csv` (see `test_results.csv`). Results across the full batch:

| Outcome | Count | Requests |
|---------|-------|----------|
| Confirmed sale (full or partial) | 8 | R5, R6, R7, R8, R9, R13, R16, R18 |
| Correctly rejected (out of stock / not in catalog) | 11 | R2, R3, R4, R10, R11, R12, R14, R15, R17, R19, R20 |
| Quote only (no order confirmed) | 1 | R1 |

Cash balance changed in **14 of 20 requests** (initial balance: $45,059.70; final balance: $45,299.75), well above the rubric minimum of 3. Zero false confirmations were recorded — every response that used confirmation language had a corresponding `sales` transaction in the database.

### Strengths

**1. Reliable rejection/confirmation separation.** The `txn_status` injection pattern ensured the orchestrator never used confirmation language without a recorded transaction. Responses for rejected requests consistently named the specific reason — insufficient stock level or item not in catalog — with the actual available quantity cited (e.g., R3: *"we currently have only 272 units available against your request for 10,000"*).

**2. Correct partial-order handling.** For requests containing a mix of available and unavailable items (R5, R7, R8, R16, R18), the system correctly processed and confirmed what it could while rejecting the rest in the same response. This required the QuotingAgent to exclude INSUFFICIENT STOCK items from the approved quote before passing it to the SalesAgent, which then processed only the valid line items.

**3. Consistent signal phrase discipline.** The `"INSUFFICIENT STOCK"` phrase was emitted reliably by the InventoryAgent across all failure modes (not in catalog, not stocked, quantity exceeds available stock), enabling downstream agents to parse availability status without ambiguity.

**4. Accurate volume discount application.** For the 8 fulfilled orders, discount tiers were applied correctly in the majority of cases — for example, 500 units of glossy paper at 10% off ($0.18/unit), 200 units of A4 at 5% off ($0.0475/unit), and 500 units of cardstock at 10% off ($0.135/unit).

### Areas for Improvement

**1. LLM discount tier errors at boundary quantities.** Three responses (R4, R6, R18) applied an incorrect discount rate to quantities in the 100–499 unit range, quoting 0% or 10% instead of the correct 5%. This is a known LLM arithmetic weakness near tier boundaries. The Python `calculate_discount()` function is correct and verified by unit tests — the issue is purely in the quoting agent's calculation. Additional worked examples in the system prompt (e.g., "207 units → ≥100, <500 → 5% off") mitigated but did not eliminate the problem.

**2. Auto-reorder triggering on unfulfilled orders.** The reorder pass runs after every `order_fulfillment` intent regardless of whether a sale was actually completed. When previously depleted items were coincidentally below their minimum stock level, the reorder agent restocked them even though the current request's order failed. This caused cash balance decreases on requests R3, R4, R10, R11, and R15 — requests that did not result in any sale revenue. The behavior is not incorrect (low stock genuinely needs replenishing) but is misleading in the context of a failed order.

---

## Section 3 — Improvement Suggestions

### Suggestion 1 — Python-Enforced Price Calculator Tool

Replace the LLM's discount arithmetic with a dedicated `calculate_quote_tool` that calls the Python `calculate_discount()` function directly and returns the computed unit prices as structured data. The QuotingAgent would then format the already-computed figures rather than performing the calculation itself. This eliminates the boundary-case tier errors observed in R4, R6, and R18, making pricing deterministic and auditable regardless of LLM behavior. The tradeoff is a slightly more constrained agent interaction — the LLM loses the ability to reason about pricing exceptions — but this is an acceptable loss given that the discount table is a business rule, not a judgment call.

### Suggestion 2 — Upsell / Sales Pitch Agent (5th Agent)

Add a dedicated `upsell_agent` that runs after the orchestrator composes the customer reply (Step 6). Given the order or quote context and the customer's event type, the agent suggests 1–3 complementary catalog items the customer did not request — for example, suggesting Presentation folders and Flyers to a customer who ordered glossy paper for a marketing event. The agent would use `search_quote_history_tool` to anchor suggestions in what similar events historically ordered, and a `get_top_sellers_tool` backed by the transactions table to surface high-revenue items. This approach adds genuine commercial value with minimal implementation cost (one new agent, two tools, one additional `run_sync` call) and no transaction risk — the upsell agent is advisory only.

### Suggestion 3 — Reorder Guard (Transaction-ID Check)

Add a Python-level guard before the auto-reorder pass that checks whether `sale_ctx` contains evidence of a completed transaction (e.g., a `transaction_id` field). If no transaction was recorded, skip the reorder pass entirely. This prevents cash balance changes on fully rejected orders, making the financial log cleaner and easier to audit. The fix is a small Python string check — no LLM changes required.

### Suggestion 4 — Vector Similarity Search on Quote History

Replace the keyword `LIKE` matching in `search_quote_history()` with a TF-IDF or embedding similarity search over the `quotes` table. The current keyword approach misses semantically relevant historical quotes when the customer uses different vocabulary (e.g., "reams" vs. "sheets", "printing paper" vs. "A4 paper"). A vector index, following the course Lesson 7 RAG pattern, would return genuinely similar precedents and improve quoting context quality — particularly useful for novel or multi-item requests.
