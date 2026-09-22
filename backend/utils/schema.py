import pyodbc
from collections import defaultdict

DB_CONFIG = {
    "server": "localhost",
    "database": "SpitalDB",
    "username": "",
    "password": "",
    "driver": "ODBC Driver 17 for SQL Server",
    "trusted_connection": "yes"
}


def get_connection():
    if DB_CONFIG["trusted_connection"].lower() == "yes":
        conn_str = (
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"Trusted_Connection=yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']};"
        )

    return pyodbc.connect(conn_str)


def get_tables(conn):
    query = """
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
      AND TABLE_SCHEMA = 'dbo'
    ORDER BY TABLE_NAME
    """
    cursor = conn.cursor()
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]


def get_columns(conn):
    query = """
    SELECT TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo'
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    cursor = conn.cursor()
    cursor.execute(query)

    columns_by_table = defaultdict(list)
    for table_name, column_name in cursor.fetchall():
        columns_by_table[table_name].append(column_name)

    return dict(columns_by_table)


def get_foreign_keys(conn):
    query = """
    SELECT
        fk.name AS fk_name,
        tp.name AS parent_table,
        cp.name AS parent_column,
        tr.name AS referenced_table,
        cr.name AS referenced_column
    FROM sys.foreign_keys fk
    INNER JOIN sys.foreign_key_columns fkc
        ON fk.object_id = fkc.constraint_object_id
    INNER JOIN sys.tables tp
        ON fkc.parent_object_id = tp.object_id
    INNER JOIN sys.columns cp
        ON fkc.parent_object_id = cp.object_id
       AND fkc.parent_column_id = cp.column_id
    INNER JOIN sys.tables tr
        ON fkc.referenced_object_id = tr.object_id
    INNER JOIN sys.columns cr
        ON fkc.referenced_object_id = cr.object_id
       AND fkc.referenced_column_id = cr.column_id
    ORDER BY tp.name, cp.name
    """

    cursor = conn.cursor()
    cursor.execute(query)

    fks = []
    for row in cursor.fetchall():
        fks.append({
            "fk_name": row[0],
            "parent_table": row[1],
            "parent_column": row[2],
            "referenced_table": row[3],
            "referenced_column": row[4],
        })

    return fks


def build_schema_info(conn):
    tables = get_tables(conn)
    columns_by_table = get_columns(conn)
    foreign_keys = get_foreign_keys(conn)

    manual_synonyms = {
        "pacienti": ["pacient", "pacienți", "bolnav", "bolnavi", "persoana", "persoane"],
        "medici": ["medic", "medici", "doctor", "doctori", "specialist", "specialiști"],
        "consultatii": ["consultatie", "consultații", "vizita", "vizite", "programare", "programări"],
    }

    schema_info = {
        "tabele": {},
        "relatii": [],
        "specialitati": ["cardiologie", "neurologie", "ortopedie", "pediatrie"],
    }

    for table in tables:
        schema_info["tabele"][table] = {
            "coloane": columns_by_table.get(table, []),
            "sinonime": manual_synonyms.get(table, [])
        }

    for fk in foreign_keys:
        schema_info["relatii"].append({
            "from_table": fk["parent_table"],
            "from_column": fk["parent_column"],
            "to_table": fk["referenced_table"],
            "to_column": fk["referenced_column"]
        })

    return schema_info


SCHEMA_INFO = {
    "tabele": {
        "pacienti": {
            "coloane": ["id", "nume", "prenume", "varsta", "sex", "oras", "telefon"],
            "sinonime": ["pacient", "pacienți", "bolnav", "bolnavi", "persoana", "persoane"],
        },
        "medici": {
            "coloane": ["id", "nume", "prenume", "specialitate", "experienta_ani", "oras"],
            "sinonime": ["medic", "medici", "doctor", "doctori", "specialist", "specialiști"],
        },
        "consultatii": {
            "coloane": ["id", "pacient_id", "medic_id", "data", "diagnostic", "cost", "durata_minute"],
            "sinonime": ["consultatie", "consultații", "vizita", "vizite", "programare", "programări"],
        },
    },
    "specialitati": ["cardiologie", "neurologie", "ortopedie", "pediatrie"],
}