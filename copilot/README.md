# Pathoconnect Copilot

Chat assistant in the CRM. It answers order questions from live data (read-only),
explains where things are in the CRM, and (phase 2) books home-visit orders
through a confirm card. It cannot edit, cancel, reschedule or assign orders yet.

```
Angular widget (secure layout)  ──JWT──►  copilot_app.py (this folder)
                                          ├─ auth.py      verify CRM JWT, same blocks as CustomerAccessGuard
                                          ├─ settings.py  LLM config from copilot_llm_settings (60s cache)
                                          ├─ agent.py     tool loop (max 8 rounds)
                                          ├─ providers.py Groq / OpenAI / any OpenAI-compatible / Anthropic
                                          ├─ tools_read.py order_summary, get_order, find_orders,
                                          │               attention_needed, phlebo_workload, help_search
                                          ├─ tools_order.py search_tests, available_slots, prepare_order (draft
                                          │               + confirm card only), draft claim/finish/cancel
                                          ├─ orders_api.py  POST {ORDERS_API_URL}/create-order with the user's JWT
                                          ├─ scope.py     same visit visibility rules as the CRM order list
                                          └─ store.py     conversations, messages, tool audit, daily limit
HSC console /hsc-console-a0ada463/copilot ──► customers service AdminSetupCopilot* (provider, model, key, on/off)
```

## Setup

1. **SQL**: run `sql/001_copilot_tables.sql` on the DB (dev first). It adds
   `customers.copilot_feature` (default 0) and the `copilot_*` tables.
   Phase 2: run `sql/002_copilot_order_drafts.sql` (one new table).
   Returning customers: run `sql/003_copilot_customer_lookups.sql` (one new table).
2. **Encryption key**: generate one value and put it in BOTH the customers
   service `.env` and this service's `.env`:
   ```
   python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"
   COPILOT_ENC_KEY=<that value>
   ```
3. **This service's `.env`** (same file app.py uses):
   ```
   DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD   # copilot_* writes
   COPILOT_READ_DB_USER / COPILOT_READ_DB_PASSWORD      # optional SELECT-only user for data reads
   JWT_SECRET                                           # same as orders/customers
   COPILOT_ENC_KEY
   CORS_ORIGINS                                         # CRM origins
   ORDERS_API_URL                                       # orders service as the CRM calls it
                                                        # (environment.OrderApiendpoint), for Confirm
   ```
4. **Console**: HSC console › Copilot → pick provider (Groq), model, paste the
   API key, *Test connection*, *Save*. Turn Copilot on for a customer.
5. **Angular**: set `CopilotApiendpoint` in `environment.prod.ts` to the
   deployed URL. Blank keeps the widget hidden.

## Run

```
pip install -r requirements-copilot.txt
python copilot_app.py                      # http://localhost:5010
```

Render (a new web service from this repo, separate from app.py):
- Build: `pip install -r requirements-copilot.txt`
- Start: `gunicorn copilot_app:app --worker-class gthread --threads 8 --timeout 120`

## Safety rules (keep them)

- customer_id / user_id come only from the verified JWT.
- The model never writes SQL. Each tool has fixed, parameterized queries and
  uses `scope.visit_scope()`.
- Data reads run in a READ ONLY session.
- Patient phones are masked before anything goes to the LLM.
- The API key is stored encrypted in the DB. This repo is public, so keys never go in the repo.
- The model can never create an order. Only the confirm route does, after a user click.

## Returning customers

When the user gives the contact phone, the model calls `customer_history` (`tools_customer.py`). It reads the
lab's past orders for that number, limited to orders this user could open in the CRM (`scope.visit_scope`), and
returns saved addresses (A1–A3), family members with today's age, their last tests at today's price, and any
order already booked for today or later. The widget shows this as the "Returning customer" card. Its
"Use these" button only sends a chat message.

The model never sees a saved address in full, only "city pincode". `prepare_order(address_ref="A1")` reads
the address from `copilot_customer_lookups` and sends `address_id` with an empty `contact_address`, so
create-order reuses that contact and address row, lat/lng included. If the user's messages already give
everything, the model calls prepare_order straight away and the confirm card appears without more questions.

## Order create (phase 2)

1. The model collects the details and calls `search_tests` / `available_slots`.
2. `prepare_order` runs the same checks as create-order (permission
   `orders-create`, test bookable, date within 15 days, slot exists and not
   past, pincode/city serviceable, user city access, area slot capacity) and
   saves a `pending` row in `copilot_order_drafts`. The widget shows it as a card.
3. Only the user's click on **Confirm** (`POST /copilot/drafts/<id>/confirm`)
   creates the order: the draft is claimed atomically (pending → creating, so a
   double click cannot create two orders), then sent to create-order with the
   user's own JWT. The result (order number + link, or the error) is saved on the draft.
4. A card expires after 10 minutes; a new card in the same chat replaces the old one.
5. Home visits only. Device-service customers are refused (they need case ID /
   service type from the Add Order page).

## Adding help topics

Add a `## Title` section with `link:` and `keywords:` lines in `help_docs/*.md`.

## Editing the order card

Every row of a pending card has an Edit link (layout A). Each Save calls `POST /copilot/drafts/<id>/update`
(`draft_edit.update_draft`) with one change and no AI call; the server checks it like the Add Order page:
`payment_type`, `address_ref` (saved addresses of the same phone only), `date` + `slot`, `contact_name`,
`special_instruction`, `patient` {index, name, age, gender}, `test` (swap to another match of the same search),
`add_test` / `remove_test`, `add_patient` (needs one test) / `remove_patient`. Totals, the create-order patient
list and warnings are rebuilt after every change, and the card gets a fresh 10 minutes. The phone cannot change.
`GET /copilot/drafts/<id>/slots?date=` lists the day's slots as ok / passed / full (for the card's area) and
`GET /copilot/tests?q=` feeds the Add test box. Confirm is refused until a payment type is set.
