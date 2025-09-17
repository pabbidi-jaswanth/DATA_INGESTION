# make_db.py
import sqlite3, pathlib

DB = pathlib.Path("Data/sql/vit_vellore.db")
DB.parent.mkdir(parents=True, exist_ok=True)

schema = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Core dictionary
CREATE TABLE IF NOT EXISTS blocks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  block_name TEXT NOT NULL,              -- e.g., 'MHS', 'MHR', 'LH-Apt'
  display_name TEXT,                     -- e.g., 'Deluxe (MHS & MHT)'
  gender TEXT CHECK(gender IN ('Male','Female')) NOT NULL,
  level TEXT CHECK(level IN ('First-Year','Senior')) NOT NULL,
  block_type TEXT,                        -- 'Regular' | 'Apartment' | 'Deluxe' | etc.
  notes TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_blocks_name ON blocks(block_name);

-- Contacts per block
CREATE TABLE IF NOT EXISTS contacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  block_id INTEGER NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
  name TEXT,
  role TEXT,
  phone TEXT,
  email TEXT
);
CREATE INDEX IF NOT EXISTS idx_contacts_block ON contacts(block_id);

-- Hostel fee rows: one row per (block, ay, category, occupancy, ac, mess_type)
CREATE TABLE IF NOT EXISTS hostel_fees (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  block_id INTEGER NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
  ay TEXT,                               -- '2025-26'
  category TEXT,                         -- 'Indian' | 'NRI' | 'Foreign'
  occupancy INTEGER,                      -- 1/2/3/4/6 etc.
  ac INTEGER CHECK(ac IN (0,1)),         -- 1=AC, 0=Non-AC
  mess_type TEXT,                         -- 'Veg' | 'Non-Veg' | 'Special'
  room_mess_fee REAL,
  admission_fee REAL,
  caution_deposit REAL,
  other_fee REAL,
  total_fee REAL,
  currency TEXT,                         -- 'INR' | 'USD'
  source_file TEXT,
  source_page INTEGER
);
CREATE INDEX IF NOT EXISTS idx_hostel_fees_lookup
  ON hostel_fees(ay, category, occupancy, ac, mess_type, block_id);

-- Amenities / policies (dhobi, timings, rules snippets you want structured)
CREATE TABLE IF NOT EXISTS amenities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  block_id INTEGER NOT NULL REFERENCES blocks(id),
  key TEXT,                               -- 'dhobi', 'mess_options', 'gate_timing', etc.
  value TEXT
);
CREATE INDEX IF NOT EXISTS idx_amenities_block ON amenities(block_id);

-- Free-text policies you may want verbatim (e.g., refund policy paragraphs)
CREATE TABLE IF NOT EXISTS policies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,           -- 'Hostel Refund Policy'
  ay TEXT,
  body TEXT,
  source_file TEXT
);
"""

with sqlite3.connect(DB) as con:
    con.executescript(schema)
print(f"[OK] Created schema → {DB}")
