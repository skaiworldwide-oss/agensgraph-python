from __future__ import annotations

import pytest

from agensgraph._protocol import composite, decode

from . import wire


def resolve(labid: int) -> str:
    return {1: "ag_vertex", 2: "ag_edge", 5: "person", 6: "knows", 7: "n"}.get(
        labid, f"label{labid}"
    )


class TestRecord:
    def test_columns_carry_their_type_and_payload(self):
        fields = composite.decode_record(wire.vertex(5, 1, b'{"n": 1}'))
        assert [f.oid for f in fields] == [7002, 3802, 27]
        assert fields[0].data == wire.graphid(5, 1)

    def test_tuple_id_is_present_in_this_rendering(self):
        """It is absent from the text form, so a reader must consume and ignore it."""
        fields = composite.decode_record(wire.vertex(5, 1, b"{}"))
        assert len(fields) == 3
        assert fields[2].oid == composite.TID_OID

    def test_null_column(self):
        buf = wire.record([(7002, None), (3802, wire.jsonb(b"{}"))])
        fields = composite.decode_record(buf)
        assert fields[0].data is None

    @pytest.mark.parametrize(
        "buf",
        [
            b"",
            b"\x00\x00",
            b"\xff\xff\xff\xff",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x01" + b"\x00\x00\x1bZ",
            b"\x00\x00\x00\x01" + b"\x00\x00\x1bZ" + b"\x7f\xff\xff\xff",
            b"\x00\x00\x00\x01" + b"\x00\x00\x1bZ" + b"\xff\xff\xff\xfe",
        ],
    )
    def test_malformed_framing_is_rejected(self, buf):
        """A length claiming more than exists must fail rather than over-read."""
        with pytest.raises(ValueError):
            composite.decode_record(buf)


class TestJsonb:
    def test_version_byte_then_json_text(self):
        assert composite.decode_jsonb(wire.jsonb(b'{"a": 1}')) == {"a": 1}

    def test_unknown_version_is_rejected(self):
        with pytest.raises(ValueError):
            composite.decode_jsonb(b"\x02{}")

    def test_empty_payload_is_rejected(self):
        with pytest.raises(ValueError):
            composite.decode_jsonb(b"")


class TestArray:
    def test_elements(self):
        buf = wire.array(7012, [b"a", b"bb", None])
        oid, payloads = composite.decode_array(buf)
        assert oid == 7012
        assert payloads == [b"a", b"bb", None]

    def test_empty_array_reports_no_dimensions(self):
        oid, payloads = composite.decode_array(wire.array(7012, []))
        assert oid == 7012
        assert payloads == []

    def test_truncated_element_is_rejected(self):
        with pytest.raises(ValueError):
            composite.decode_array(b"\x00\x00\x00\x01" * 3)


class TestValues:
    def test_vertex(self):
        v = decode.vertex_from_binary(wire.vertex(5, 1, b'{"name": "a"}'), resolve)
        assert v.label == "person"
        assert (v.id.labid, v.id.locid) == (5, 1)
        assert v.properties == {"name": "a"}

    def test_label_comes_from_the_resolver_not_the_value(self):
        """The binary rendering does not carry the name, only the id."""
        v = decode.vertex_from_binary(wire.vertex(99, 1, b"{}"), resolve)
        assert v.label == "label99"

    def test_edge(self):
        e = decode.edge_from_binary(wire.edge(6, 1, (5, 1), (5, 2), b'{"w": 2}'), resolve)
        assert e.label == "knows"
        assert (e.start.labid, e.start.locid) == (5, 1)
        assert (e.end.labid, e.end.locid) == (5, 2)
        assert e.properties == {"w": 2}

    def test_path(self):
        buf = wire.path(
            [wire.vertex(5, 1, b"{}"), wire.vertex(5, 2, b"{}")],
            [wire.edge(6, 1, (5, 1), (5, 2), b"{}")],
        )
        p = decode.path_from_binary(buf, resolve)
        assert len(p.vertices) == 2
        assert p.length == 1

    def test_empty_path(self):
        p = decode.path_from_binary(wire.path([], []), resolve)
        assert len(p) == 0

    def test_a_null_element_in_a_path_is_rejected(self):
        buf = wire.record(
            [
                (wire.VERTEX_ARRAY_OID, wire.array(7012, [wire.vertex(5, 1, b"{}"), None])),
                (
                    wire.EDGE_ARRAY_OID,
                    wire.array(7022, [wire.edge(6, 1, (5, 1), (5, 2), b"{}")]),
                ),
            ]
        )
        with pytest.raises(ValueError):
            decode.path_from_binary(buf, resolve)

    def test_a_wrong_type_in_the_property_column_is_rejected(self):
        buf = wire.record([(7002, wire.graphid(5, 1)), (25, b"text not jsonb")])
        with pytest.raises(ValueError):
            decode.vertex_from_binary(buf, resolve)
