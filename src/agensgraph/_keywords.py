"""Words the server's grammar keeps for itself, on any supported release.

Every keyword that ``src/include/parser/kwlist.h`` lists as reserved, a column name or a type or
function name in AgensGraph 2.14.1, 2.17.0 or 2.18.6.1 -- the releases whose grammars bracket the
supported range. An unreserved keyword works as a bare name and is left out, so names as common as
``name``, ``type`` and ``value`` stay bare. Measured, every word that broke a statement as a bare
label or variable is on this list: ``offset`` and ``array`` on every release, and ``let``,
``finish`` and ``current_graph`` on 2.18.

To add a release, take the union again from its ``kwlist.h``.
"""

from __future__ import annotations

GRAMMAR_KEYWORDS = frozenset(
    {
        "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric",
        "authorization", "between", "bigint", "binary", "bit", "boolean", "both", "case",
        "cast", "char", "character", "check", "coalesce", "collate", "collation", "column",
        "concurrently", "constraint", "create", "cross", "current_catalog", "current_date",
        "current_graph", "current_property_graph", "current_role", "current_schema",
        "current_time", "current_timestamp", "current_user", "dec", "decimal", "default",
        "deferrable", "desc", "distinct", "do", "else", "end", "except", "exists", "extract",
        "false", "fetch", "finish", "float", "for", "foreign", "freeze", "from", "full",
        "grant", "greatest", "group", "grouping", "having", "ilike", "in", "initially", "inner",
        "inout", "insert", "int", "integer", "intersect", "interval", "into", "is", "isnull",
        "join", "json", "json_array", "json_arrayagg", "json_exists", "json_object",
        "json_objectagg", "json_query", "json_scalar", "json_serialize", "json_table",
        "json_value", "lateral", "leading", "least", "left", "let", "like", "limit",
        "localtime", "localtimestamp", "match", "merge", "merge_action", "national", "natural",
        "nchar", "none", "normalize", "not", "notnull", "null", "nullif", "numeric", "offset",
        "on", "only", "or", "order", "out", "outer", "overlaps", "overlay", "placing",
        "position", "precision", "primary", "real", "references", "return", "returning",
        "right", "row", "select", "session_user", "setof", "similar", "smallint", "some",
        "substring", "symmetric", "system_user", "table", "tablesample", "then", "time",
        "timestamp", "to", "trailing", "treat", "trim", "true", "union", "unique", "unwind",
        "user", "using", "values", "varchar", "variadic", "verbose", "when", "where", "window",
        "with", "xmlattributes", "xmlconcat", "xmlelement", "xmlexists", "xmlforest",
        "xmlnamespaces", "xmlparse", "xmlpi", "xmlroot", "xmlserialize", "xmltable", "xor",
    }
)  # fmt: skip
