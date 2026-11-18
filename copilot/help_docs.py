"""Tiny keyword search over the CRM help docs in copilot/help_docs/*.md.

Each file holds sections in this shape:

    ## Title of the help topic
    link: /orders/create
    keywords: create order, new booking, add order
    Body text (steps, notes) ...

Good enough for a few dozen topics; swap for embeddings when it grows.
"""
import os
import re
from functools import lru_cache

DOCS_DIR = os.path.join(os.path.dirname(__file__), 'help_docs')
_WORD = re.compile(r'[a-z0-9]+')
_STOP = {'the', 'a', 'an', 'to', 'of', 'in', 'on', 'for', 'and', 'or', 'how', 'do', 'i', 'can', 'is',
         'where', 'what', 'my', 'me', 'it', 'with', 'from', 'set', 'up', 'find', 'see', 'page',
         'add', 'new', 'create', 'want', 'need', 'please'}


def _words(text):
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    # crude singular form so "orders" matches "order"
    return {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in words}


@lru_cache(maxsize=1)
def load():
    topics = []
    for name in sorted(os.listdir(DOCS_DIR)):
        if not name.endswith('.md'):
            continue
        with open(os.path.join(DOCS_DIR, name), encoding='utf-8') as fh:
            text = fh.read()
        for chunk in re.split(r'^## ', text, flags=re.M)[1:]:
            lines = chunk.strip().splitlines()
            title, meta, body = lines[0].strip(), {}, []
            for line in lines[1:]:
                m = re.match(r'^(link|keywords):\s*(.*)$', line.strip())
                if m and not body:
                    meta[m.group(1)] = m.group(2).strip()
                else:
                    body.append(line)
            topics.append({
                'title': title,
                'link': meta.get('link') or None,
                'keywords': meta.get('keywords', ''),
                'body': '\n'.join(body).strip(),
                '_title_words': _words(title),
                '_key_words': _words(meta.get('keywords', '')),
                '_body_words': _words('\n'.join(body)),
            })
    return topics


def search(query, limit=3):
    q = _words(query)
    if not q:
        return {'results': [], 'note': 'Empty query'}
    scored = []
    for t in load():
        score = 3 * len(q & t['_key_words']) + 2 * len(q & t['_title_words']) + len(q & t['_body_words'])
        if score:
            scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    results = [{'title': t['title'], 'link': t['link'], 'steps': t['body']} for _, t in scored[:limit]]
    if not results:
        return {'results': [], 'note': 'No help topic matched. Suggest the closest menu from the main menu list, '
                                        'or ask the user to contact support.',
                'main_menu': [{'title': t['title'], 'link': t['link']} for t in load() if t['link']][:25]}
    return {'results': results}
