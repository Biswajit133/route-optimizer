"""DB connections for the Copilot.

Two connections, on purpose:
- read():  everything the model can ask about (orders, visits, patients ...).
  Uses COPILOT_READ_DB_* when set (a SELECT-only MySQL user is recommended),
  otherwise the normal DB_* user, and is always opened as a READ ONLY session
  so even a bug in a tool cannot write.
- write(): only the copilot_* tables (conversations, messages, audit).
"""
import os
from contextlib import contextmanager

import pymysql
import pymysql.cursors
from dotenv import load_dotenv

load_dotenv()

# Tool queries must stay snappy; a runaway query should fail, not hang a worker.
READ_TIMEOUT_SECONDS = 15


def _conf(prefix):
    def pick(name, default=None):
        value = os.environ.get(prefix + name)
        if value in (None, ''):
            value = os.environ.get('DB_' + name, default)
        return value

    return {
        'host': pick('HOST'),
        'port': int(pick('PORT', 3306)),
        'user': pick('USER'),
        'password': pick('PASSWORD', ''),
        'database': pick('NAME'),
    }


def _connect(conf, autocommit):
    return pymysql.connect(
        host=conf['host'],
        port=conf['port'],
        user=conf['user'],
        password=conf['password'],
        database=conf['database'],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=READ_TIMEOUT_SECONDS,
        write_timeout=READ_TIMEOUT_SECONDS,
        autocommit=autocommit,
    )


@contextmanager
def read():
    conn = _connect(_conf('COPILOT_READ_DB_'), autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION TRANSACTION READ ONLY")
        yield conn
    finally:
        conn.close()


@contextmanager
def write():
    conn = _connect(_conf('DB_'), autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def fetch_one(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(conn, sql, params=()):
    """Runs a write and returns lastrowid."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.lastrowid


def in_list(values):
    """'%s, %s, %s' for an IN (...) clause, plus the params. Never empty."""
    values = list(values) or [None]
    return ', '.join(['%s'] * len(values)), values
