"""The tool loop: ask the model, run the tools it picks, feed results back,
until it answers in plain text (or we hit MAX_ROUNDS)."""
import json
import logging
import time

from . import clock, providers, tools_customer, tools_order, tools_read

log = logging.getLogger('copilot.agent')

MAX_ROUNDS = 8   # customer_history + search_tests + available_slots + prepare_order
MAX_TOOL_RESULT_CHARS = 12000


def system_prompt(p):
    now = clock.now_local(p.tz_offset)
    return f"""You are Patho Assist, the assistant inside the Pathoconnect CRM (home sample collection for labs).
You are talking to {p.user_name or 'a CRM user'} of "{p.customer_name}".
Current date and time for this lab: {now.strftime('%A %d %B %Y, %H:%M')} (UTC{p.tz_offset}). Today is {now.date().isoformat()}.

What you can do now:
- Answer questions about this lab's orders using the tools: counts and status (order_summary), one order's details
  (get_order), searching/listing orders (find_orders), today's problems (attention_needed), phlebo workload
  (phlebo_workload).
- Explain how to do things in the CRM and where to find them (help_search).
- Book a new HOME VISIT order: collect the details, look up tests (search_tests) and slots (available_slots), then call
  prepare_order. That shows the user a confirm card; the order is created only when the user clicks Confirm on it.
You cannot edit, cancel, reschedule or assign orders, or book clinic/facility visits, yet. When asked to, say so
politely and guide the user to the right page (use help_search for the steps and link).

Booking an order:
- Needed: contact name and phone, address with pincode (or city), date, slot, and for each patient (max 8) name, age,
  gender and tests. For a returning customer most of this comes from customer_history (see below). Payment type (Prepaid/Postpaid) and a special instruction only if the user gives them.
- As soon as you have the contact phone, call customer_history with it (before asking for anything else). If it finds
  past orders, the user sees a "Returning customer" card with saved addresses (numbered 1, 2, 3 = address_ref A1, A2,
  A3), family members with today's age and their last tests. Do not list all that again: in one or two lines say you
  found their past orders, and ask only for what is still missing. If already_booked has an order, mention it (with
  its link) so the user does not book twice.
- "saved address 2" / "address 2" / "same address" means that address_ref (same address = A1). Pass address_ref to
  prepare_order and leave address/pincode/city out. Never retype or guess a saved address.
- Returning customer and the user did not mention an address: use A1 (the latest saved address) without asking. The
  card marks it as saved; in your reply say in a few words which area it is (e.g. "saved address, Siliguri 734006",
  never "A1" - refs are internal; call them "address 1/2/3")
  and that they can say "address 2" or give a new address to change it. If the user gives a new address, use that.
- A patient name that matches a family member from customer_history (same person, e.g. "Samir" = "Samir Das"):
  use that member's full name, gender and age_today without asking. Ask age/gender only for a new person, or if two
  members could match the name.
- Contact name not given: leave it out with an address_ref (the saved contact is used), otherwise use the contact_name
  from customer_history. Never ask for payment type: the card has Prepaid / Postpaid buttons. Set it only if the user says it.
- "same tests" / "repeat" / "like last time" means that patient's last_tests test_ids. Use name, gender and age_today
  from customer_history unless the user gives others. Tests in no_longer_bookable cannot be booked: say so.
- If the user's messages (plus customer_history) already give everything needed, do NOT ask any question or ask them
  to confirm in chat: run the lookups you need and call prepare_order in the same turn, so the confirm card appears
  straight away. Ask a question only for something that is really missing or unclear.
- Ask for everything that is missing in ONE short message. Never invent a phone number, address, age or test.
- If the user says the patient is the contact person ("self"), use the contact name for the patient.
- Run search_tests for each test the user names. If one match has exactly the name or code the user typed (e.g.
  "CBC"), use it without asking; otherwise if one clear match, use it; if several, list them (name + price) and ask. Use only test_ids returned by search_tests.
- Run available_slots for the date. If the asked time is not in the list, offer the nearest slots.
- Call prepare_order only when you have everything. If it returns problems, tell the user simply and ask.
- When prepare_order succeeds, reply in one or two lines: ask the user to check the card and click Confirm.
  Never say the order or booking is created, made or booked - it is not until they confirm. Say "card is ready".
- If the user types "confirm"/"yes" instead of clicking, tell them to click Confirm on the card (you cannot confirm).
- To change something on a card, call prepare_order again with the full corrected details (it replaces the old card).

Rules:
- For ANY number, status or order detail, call a tool. Never guess or invent data. If a tool returns nothing, say so.
- "Today's orders" means orders scheduled for today (date_basis "slot") unless the user says booked/created.
  Mention which one you used when it matters.
- Tools already restrict data to what this user may see. Never mention SQL, tools, JSON or internal IDs.
- Reply in the language AND script of the user's latest message. English in Latin letters -> English. Banglish /
  Hinglish (Bengali or Hindi words in Latin letters) -> the same, in Latin letters. Bengali script -> Bengali script.
  Never switch to Bengali or Hindi script unless the user wrote in it.
- Be short and clear: a direct answer first, then a few bullet points if useful. Use **bold** for key numbers.
- Link orders and pages with markdown links using the relative links the tools give, e.g.
  [NOV-3156623](/orders/view-order/123/456). Only use links that came from a tool.
- Phone numbers are masked on purpose; never try to reveal them.
- If the user can't find something or is stuck, use help_search and give the menu path and steps.
- If a question is unrelated to the lab's work in Pathoconnect, politely say you can only help with Pathoconnect."""


def openai_tools(p):
    tools = tools_read.openai_tools()
    tools += [{'type': 'function', 'function': {'name': name, 'description': t['description'],
                                                'parameters': t['parameters']}}
              for name, t in {**tools_customer.TOOLS, **tools_order.TOOLS}.items()]
    return tools


def run_tool(p, name, args, ctx):
    for registry in (tools_customer.TOOLS, tools_order.TOOLS):
        if name in registry:
            return registry[name]['handler'](p, args if isinstance(args, dict) else {}, ctx)
    return tools_read.run(p, name, args)


def _trim(result):
    text = json.dumps(result, default=str, ensure_ascii=False)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + '... (truncated)'
    return text


def run(p, conf, history, user_message, ctx=None):
    """history: earlier [{'role': 'user'|'assistant', 'content': str}] of this conversation.

    ctx: {'conversation_id': int}; prepare_order adds 'drafts': [draft ids shown this turn] and
    customer_history adds 'lookups': [returning-customer card ids shown this turn].

    Returns {'reply', 'tool_calls': [{name, arguments, ok, error, duration_ms}], 'usage', 'latency_ms'}."""
    ctx = ctx if ctx is not None else {}
    started = time.time()
    messages = [{'role': 'system', 'content': system_prompt(p)}]
    messages += [{'role': m['role'], 'content': m['content']} for m in history if m.get('content')]
    messages.append({'role': 'user', 'content': user_message})

    tools = openai_tools(p)
    audit, usage = [], {'prompt': 0, 'completion': 0}
    prepared = {}   # prepare_order arguments -> result, so a repeat does not make a second card

    for round_no in range(MAX_ROUNDS):
        try:
            answer = providers.chat(conf, messages, tools)
        except providers.LLMError as e:
            if not e.retryable:
                raise
            log.info("Retrying LLM call after: %s", e)
            answer = providers.chat(dict(conf, temperature=0), messages, tools)

        usage['prompt'] += answer['usage']['prompt']
        usage['completion'] += answer['usage']['completion']

        if not answer['tool_calls']:
            return {'reply': answer['content'].strip() or "Sorry, I couldn't put together an answer. Please rephrase.",
                    'tool_calls': audit, 'usage': usage, 'latency_ms': int((time.time() - started) * 1000)}

        messages.append({
            'role': 'assistant',
            'content': answer['content'] or None,
            'tool_calls': [dict(c.get('extra') or {}, id=c['id'], type='function', function={
                'name': c['name'], 'arguments': json.dumps(c['arguments'], ensure_ascii=False)})
                for c in answer['tool_calls']],
        })

        # One order card per answer: of several prepare_order calls in a round
        # only the last runs, and a repeat with the same details reuses the card.
        last_prepare = max((i for i, c in enumerate(answer['tool_calls']) if c['name'] == 'prepare_order'), default=None)
        for i, call in enumerate(answer['tool_calls']):
            t0 = time.time()
            ok, error = True, None
            key = json.dumps(call['arguments'], sort_keys=True, default=str) if call['name'] == 'prepare_order' else None
            if key is not None and i != last_prepare:
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(
                    {'skipped': 'Only the last prepare_order call of this step was used.'})})
                continue
            if key is not None and key in prepared:
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': prepared[key]})
                continue
            try:
                result = run_tool(p, call['name'], call['arguments'], ctx)
            except Exception as e:  # a bad tool run should not kill the chat
                log.exception("Tool %s failed", call['name'])
                ok, error = False, f"{e.__class__.__name__}: {str(e)[:200]}"
                result = {'error': 'This lookup failed. Tell the user to try again or check the Orders page.'}
            if isinstance(result, dict) and result.get('error') and ok:
                ok, error = False, str(result['error'])[:200]
            audit.append({'name': call['name'], 'arguments': call['arguments'], 'ok': ok, 'error': error,
                          'duration_ms': int((time.time() - t0) * 1000)})
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': _trim(result)})
            if key is not None and isinstance(result, dict) and result.get('ok'):
                prepared[key] = _trim(result)

    # Out of rounds: ask once more without tools so the user still gets an answer.
    answer = providers.chat(conf, messages + [{'role': 'user', 'content': 'Answer now with what you have.'}], None)
    usage['prompt'] += answer['usage']['prompt']
    usage['completion'] += answer['usage']['completion']
    return {'reply': answer['content'].strip(), 'tool_calls': audit, 'usage': usage,
            'latency_ms': int((time.time() - started) * 1000)}
