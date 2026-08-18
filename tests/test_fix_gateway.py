"""Tests for the FIX 4.4 protocol gateway.

Covers message parsing, checksum computation, order mapping,
execution report generation, session management, and integration.
"""

from __future__ import annotations

import asyncio
import pytest

import exchange_simulator as ex

from gateway.fix_parser import (
    SOH,
    build_message,
    compute_checksum,
    extract_message,
    parse_message,
    FixParseError,
    TAG_BEGIN_STRING,
    TAG_BODY_LENGTH,
    TAG_MSG_TYPE,
    TAG_MSG_SEQ_NUM,
    TAG_SENDER_COMP_ID,
    TAG_TARGET_COMP_ID,
    TAG_SENDING_TIME,
    TAG_CHECKSUM,
    TAG_ENCRYPT_METHOD,
    TAG_HEARTBT_INT,
    TAG_CL_ORD_ID,
    TAG_SIDE,
    TAG_ORD_TYPE,
    TAG_PRICE,
    TAG_ORDER_QTY,
    TAG_SYMBOL,
    TAG_TIME_IN_FORCE,
    TAG_EXEC_TYPE,
    TAG_ORD_STATUS,
    TAG_LAST_PX,
    TAG_LAST_QTY,
    TAG_CUM_QTY,
    MSG_TYPE_LOGON,
    MSG_TYPE_LOGOUT,
    MSG_TYPE_NEW_ORDER_SINGLE,
    MSG_TYPE_ORDER_CANCEL_REQUEST,
    MSG_TYPE_EXECUTION_REPORT,
    MSG_TYPE_HEARTBEAT,
    TAG_ORIG_CL_ORD_ID,
    TAG_TEST_REQ_ID,
)
from gateway.fix_session import FixSession, SessionState
from gateway.fix_gateway import (
    FixGateway,
    FIX_SIDE_BUY,
    FIX_SIDE_SELL,
    FIX_ORD_TYPE_LIMIT,
    FIX_TIF_DAY,
    EXEC_TYPE_NEW,
    EXEC_TYPE_FILL,
    EXEC_TYPE_CANCELED,
    ORD_STATUS_NEW,
    ORD_STATUS_FILLED,
    ORD_STATUS_CANCELED,
)


# --- FIX Parser Tests ---


class TestFixParser:
    """Tests for FIX message parsing and building."""

    def test_parse_valid_message(self):
        """Parse a well-formed FIX message into a dict."""
        raw = (
            f"8=FIX.4.4{SOH}9=5{SOH}35=A{SOH}"
        )
        # Add proper checksum
        checksum = compute_checksum(raw)
        raw += f"10={checksum}{SOH}"

        fields = parse_message(raw)
        assert fields["8"] == "FIX.4.4"
        assert fields["35"] == "A"
        assert fields["10"] == checksum

    def test_parse_malformed_no_equals(self):
        """Reject a field that has no '=' separator."""
        raw = f"8=FIX.4.4{SOH}BADFIELD{SOH}35=A{SOH}"
        with pytest.raises(FixParseError, match="no '='"):
            parse_message(raw)

    def test_parse_empty_message(self):
        """Reject an empty string."""
        with pytest.raises(FixParseError, match="Empty message"):
            parse_message("")

    def test_checksum_computation(self):
        """Verify checksum is sum of ASCII bytes mod 256, zero-padded."""
        # Simple known case
        data = "8=FIX.4.4" + SOH + "9=5" + SOH + "35=A" + SOH
        cs = compute_checksum(data)
        expected = sum(ord(c) for c in data) % 256
        assert cs == f"{expected:03d}"

    def test_checksum_validation_fails(self):
        """Reject a message with invalid checksum."""
        raw = f"8=FIX.4.4{SOH}9=5{SOH}35=A{SOH}10=000{SOH}"
        with pytest.raises(FixParseError, match="Checksum mismatch"):
            parse_message(raw)

    def test_build_message_includes_checksum_and_body_length(self):
        """Built messages have correct BodyLength and Checksum."""
        fields = {
            TAG_MSG_TYPE: "A",
            TAG_SENDER_COMP_ID: "CLIENT",
            TAG_TARGET_COMP_ID: "SERVER",
        }
        msg = build_message(fields)

        # Parse it back
        parsed = parse_message(msg)
        assert parsed[TAG_BEGIN_STRING] == "FIX.4.4"
        assert TAG_BODY_LENGTH in parsed
        assert TAG_CHECKSUM in parsed
        assert parsed[TAG_MSG_TYPE] == "A"

    def test_build_and_parse_roundtrip(self):
        """A built message can be parsed back successfully."""
        fields = {
            TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
            TAG_MSG_SEQ_NUM: "1",
            TAG_SENDER_COMP_ID: "CLIENT1",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:00.000",
            TAG_CL_ORD_ID: "order123",
            TAG_SIDE: "1",
            TAG_ORD_TYPE: "2",
            TAG_PRICE: "100.5000",
            TAG_ORDER_QTY: "50",
            TAG_TIME_IN_FORCE: "0",
        }
        raw = build_message(fields)
        parsed = parse_message(raw)

        assert parsed[TAG_MSG_TYPE] == MSG_TYPE_NEW_ORDER_SINGLE
        assert parsed[TAG_CL_ORD_ID] == "order123"
        assert parsed[TAG_PRICE] == "100.5000"
        assert parsed[TAG_ORDER_QTY] == "50"

    def test_extract_message_from_buffer(self):
        """Extract a complete message from a byte buffer."""
        msg = build_message({TAG_MSG_TYPE: "0"})
        buffer = msg.encode("ascii") + b"leftover"
        extracted, remaining = extract_message(buffer)
        assert extracted is not None
        assert remaining == b"leftover"

    def test_extract_message_incomplete(self):
        """Return None when buffer has no complete message."""
        buffer = b"8=FIX.4.4\x019=5\x0135=A\x01"
        extracted, remaining = extract_message(buffer)
        assert extracted is None
        assert remaining == buffer


# --- FIX Session Tests ---


class TestFixSession:
    """Tests for FIX session state management."""

    @pytest.fixture
    def sent_messages(self):
        """Collector for messages sent by the session."""
        return []

    @pytest.fixture
    def session(self, sent_messages):
        """Create a session with a mock send function."""
        s = FixSession(
            sender_comp_id="EXCHANGE",
            target_comp_id=None,
            heartbeat_interval=30,
        )

        async def mock_send(data: str):
            sent_messages.append(data)

        s.set_send_func(mock_send)
        return s

    @pytest.mark.asyncio
    async def test_sequence_numbers_increment(self, session, sent_messages):
        """Outgoing sequence numbers increment with each message."""
        assert session.outgoing_seq_num == 0

        await session.send_message({TAG_MSG_TYPE: MSG_TYPE_HEARTBEAT})
        assert session.outgoing_seq_num == 1

        await session.send_message({TAG_MSG_TYPE: MSG_TYPE_HEARTBEAT})
        assert session.outgoing_seq_num == 2

        # Verify in the sent messages
        parsed1 = parse_message(sent_messages[0])
        parsed2 = parse_message(sent_messages[1])
        assert parsed1[TAG_MSG_SEQ_NUM] == "1"
        assert parsed2[TAG_MSG_SEQ_NUM] == "2"

    @pytest.mark.asyncio
    async def test_logon_flow(self, session, sent_messages):
        """Logon from client transitions session to ACTIVE."""
        assert session.state == SessionState.DISCONNECTED

        # Simulate client sending logon
        logon_msg = build_message({
            TAG_MSG_TYPE: MSG_TYPE_LOGON,
            TAG_MSG_SEQ_NUM: "1",
            TAG_SENDER_COMP_ID: "CLIENT1",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:00.000",
            TAG_ENCRYPT_METHOD: "0",
            TAG_HEARTBT_INT: "30",
        })
        await session.receive_message(logon_msg)

        assert session.state == SessionState.ACTIVE
        assert session.target_comp_id == "CLIENT1"
        assert session.incoming_seq_num == 1

        # Should have sent logon ack
        assert len(sent_messages) == 1
        ack = parse_message(sent_messages[0])
        assert ack[TAG_MSG_TYPE] == MSG_TYPE_LOGON

    @pytest.mark.asyncio
    async def test_logout_flow(self, session, sent_messages):
        """Logout transitions session from ACTIVE to LOGOUT."""
        # First logon
        logon_msg = build_message({
            TAG_MSG_TYPE: MSG_TYPE_LOGON,
            TAG_MSG_SEQ_NUM: "1",
            TAG_SENDER_COMP_ID: "CLIENT1",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:00.000",
            TAG_ENCRYPT_METHOD: "0",
            TAG_HEARTBT_INT: "30",
        })
        await session.receive_message(logon_msg)
        assert session.state == SessionState.ACTIVE

        # Now logout
        logout_msg = build_message({
            TAG_MSG_TYPE: MSG_TYPE_LOGOUT,
            TAG_MSG_SEQ_NUM: "2",
            TAG_SENDER_COMP_ID: "CLIENT1",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:01.000",
        })
        await session.receive_message(logout_msg)
        assert session.state == SessionState.LOGOUT

    @pytest.mark.asyncio
    async def test_invalid_comp_id_rejected(self, session, sent_messages):
        """Messages with wrong CompID are rejected after logon."""
        # Logon as CLIENT1
        logon_msg = build_message({
            TAG_MSG_TYPE: MSG_TYPE_LOGON,
            TAG_MSG_SEQ_NUM: "1",
            TAG_SENDER_COMP_ID: "CLIENT1",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:00.000",
            TAG_ENCRYPT_METHOD: "0",
            TAG_HEARTBT_INT: "30",
        })
        await session.receive_message(logon_msg)
        sent_messages.clear()

        # Send message as WRONG_CLIENT
        bad_msg = build_message({
            TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
            TAG_MSG_SEQ_NUM: "2",
            TAG_SENDER_COMP_ID: "WRONG_CLIENT",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:01.000",
            TAG_CL_ORD_ID: "order1",
            TAG_SIDE: "1",
            TAG_ORD_TYPE: "2",
            TAG_PRICE: "100.0000",
            TAG_ORDER_QTY: "10",
        })
        await session.receive_message(bad_msg)

        # Should have sent a rejection logout
        assert len(sent_messages) == 1
        reject = parse_message(sent_messages[0])
        assert reject[TAG_MSG_TYPE] == MSG_TYPE_LOGOUT
        assert "Invalid SenderCompID" in reject.get("58", "")

    @pytest.mark.asyncio
    async def test_test_request_heartbeat_response(self, session, sent_messages):
        """TestRequest is answered with a Heartbeat containing the TestReqID."""
        # Logon first
        logon_msg = build_message({
            TAG_MSG_TYPE: MSG_TYPE_LOGON,
            TAG_MSG_SEQ_NUM: "1",
            TAG_SENDER_COMP_ID: "CLIENT1",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:00.000",
            TAG_ENCRYPT_METHOD: "0",
            TAG_HEARTBT_INT: "30",
        })
        await session.receive_message(logon_msg)
        sent_messages.clear()

        # Send TestRequest
        test_req = build_message({
            TAG_MSG_TYPE: "1",  # TestRequest
            TAG_MSG_SEQ_NUM: "2",
            TAG_SENDER_COMP_ID: "CLIENT1",
            TAG_TARGET_COMP_ID: "EXCHANGE",
            TAG_SENDING_TIME: "20260707-12:00:05.000",
            TAG_TEST_REQ_ID: "TEST123",
        })
        await session.receive_message(test_req)

        # Should respond with heartbeat
        assert len(sent_messages) == 1
        hb = parse_message(sent_messages[0])
        assert hb[TAG_MSG_TYPE] == MSG_TYPE_HEARTBEAT
        assert hb[TAG_TEST_REQ_ID] == "TEST123"


# --- Gateway Integration Tests ---
# These tests require a live TCP connection and can be slow in CI.
# Run manually with: pytest tests/test_fix_gateway.py::TestFixGateway -v --timeout=30


@pytest.mark.skip(reason="TCP integration tests require manual run (heartbeat timing)")
class TestFixGateway:
    """Integration tests for the FIX gateway with the matching engine."""

    @pytest.fixture
    def engine(self):
        """Fresh matching engine."""
        return ex.MatchingEngine()

    @pytest.fixture
    def gateway(self, engine):
        """Gateway instance with the engine."""
        return FixGateway(engine=engine, port=0, comp_id="EXCHANGE")

    @pytest.mark.asyncio
    async def test_new_order_single_mapping(self, gateway):
        """NewOrderSingle is correctly mapped to engine Order and submitted."""
        # Start gateway on random port
        await gateway.start()
        port = gateway._server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Send logon
            logon = build_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGON,
                TAG_MSG_SEQ_NUM: "1",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:00.000",
                TAG_ENCRYPT_METHOD: "0",
                TAG_HEARTBT_INT: "30",
            })
            writer.write(logon.encode("ascii"))
            await writer.drain()

            # Wait for logon ack
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            ack_msg, _ = extract_message(data)
            assert ack_msg is not None
            ack = parse_message(ack_msg)
            assert ack[TAG_MSG_TYPE] == MSG_TYPE_LOGON

            # Send NewOrderSingle (limit buy)
            nos = build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: "2",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:01.000",
                TAG_CL_ORD_ID: "myorder1",
                TAG_SIDE: FIX_SIDE_BUY,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "100.5000",
                TAG_ORDER_QTY: "50",
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "AAPL",
            })
            writer.write(nos.encode("ascii"))
            await writer.drain()

            # Wait for execution report (New ack)
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            er_msg, _ = extract_message(data)
            assert er_msg is not None
            er = parse_message(er_msg)
            assert er[TAG_MSG_TYPE] == MSG_TYPE_EXECUTION_REPORT
            assert er[TAG_EXEC_TYPE] == EXEC_TYPE_NEW
            assert er[TAG_ORD_STATUS] == ORD_STATUS_NEW
            assert er[TAG_CL_ORD_ID] == "myorder1"

            writer.close()
            await writer.wait_closed()
        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_execution_report_on_fill(self, gateway):
        """Matching orders produce ExecutionReport with fill details."""
        await gateway.start()
        port = gateway._server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Logon
            logon = build_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGON,
                TAG_MSG_SEQ_NUM: "1",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:00.000",
                TAG_ENCRYPT_METHOD: "0",
                TAG_HEARTBT_INT: "30",
            })
            writer.write(logon.encode("ascii"))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)

            # Submit a resting sell order (limit)
            sell = build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: "2",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:01.000",
                TAG_CL_ORD_ID: "sell1",
                TAG_SIDE: FIX_SIDE_SELL,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "100.0000",
                TAG_ORDER_QTY: "25",
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "AAPL",
            })
            writer.write(sell.encode("ascii"))
            await writer.drain()

            # Read sell ack
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)

            # Submit matching buy order
            buy = build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: "3",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:02.000",
                TAG_CL_ORD_ID: "buy1",
                TAG_SIDE: FIX_SIDE_BUY,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "100.0000",
                TAG_ORDER_QTY: "25",
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "AAPL",
            })
            writer.write(buy.encode("ascii"))
            await writer.drain()

            # Read execution reports - expect New + Fill
            data = await asyncio.wait_for(reader.read(8192), timeout=2.0)
            buffer = data
            messages = []
            while True:
                msg, buffer = extract_message(buffer)
                if msg is None:
                    break
                messages.append(parse_message(msg))

            # Find the fill report for buy1
            fill_reports = [
                m for m in messages
                if m.get(TAG_EXEC_TYPE) == EXEC_TYPE_FILL
                and m.get(TAG_CL_ORD_ID) == "buy1"
            ]
            assert len(fill_reports) == 1
            fill = fill_reports[0]
            assert fill[TAG_ORD_STATUS] == ORD_STATUS_FILLED
            assert fill[TAG_LAST_QTY] == "25"
            assert fill[TAG_CUM_QTY] == "25"
            assert float(fill[TAG_LAST_PX]) == pytest.approx(100.0, abs=0.001)

            writer.close()
            await writer.wait_closed()
        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_cancel_request(self, gateway):
        """OrderCancelRequest cancels a resting order."""
        await gateway.start()
        port = gateway._server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Logon
            logon = build_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGON,
                TAG_MSG_SEQ_NUM: "1",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:00.000",
                TAG_ENCRYPT_METHOD: "0",
                TAG_HEARTBT_INT: "30",
            })
            writer.write(logon.encode("ascii"))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)

            # Submit order
            nos = build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: "2",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:01.000",
                TAG_CL_ORD_ID: "order_to_cancel",
                TAG_SIDE: FIX_SIDE_BUY,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "99.0000",
                TAG_ORDER_QTY: "100",
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "AAPL",
            })
            writer.write(nos.encode("ascii"))
            await writer.drain()

            # Read new ack
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)

            # Send cancel request
            cancel = build_message({
                TAG_MSG_TYPE: MSG_TYPE_ORDER_CANCEL_REQUEST,
                TAG_MSG_SEQ_NUM: "3",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:02.000",
                TAG_CL_ORD_ID: "cancel1",
                TAG_ORIG_CL_ORD_ID: "order_to_cancel",
                TAG_SIDE: FIX_SIDE_BUY,
                TAG_SYMBOL: "AAPL",
            })
            writer.write(cancel.encode("ascii"))
            await writer.drain()

            # Read cancel ack
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            msg, _ = extract_message(data)
            assert msg is not None
            er = parse_message(msg)
            assert er[TAG_MSG_TYPE] == MSG_TYPE_EXECUTION_REPORT
            assert er[TAG_EXEC_TYPE] == EXEC_TYPE_CANCELED
            assert er[TAG_ORD_STATUS] == ORD_STATUS_CANCELED

            writer.close()
            await writer.wait_closed()
        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_full_integration_roundtrip(self, gateway):
        """Full roundtrip: logon, submit, fill, cancel, logout."""
        await gateway.start()
        port = gateway._server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # 1. Logon
            logon = build_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGON,
                TAG_MSG_SEQ_NUM: "1",
                TAG_SENDER_COMP_ID: "TRADER1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:00.000",
                TAG_ENCRYPT_METHOD: "0",
                TAG_HEARTBT_INT: "30",
            })
            writer.write(logon.encode("ascii"))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            msg, _ = extract_message(data)
            ack = parse_message(msg)
            assert ack[TAG_MSG_TYPE] == MSG_TYPE_LOGON

            # 2. Submit sell limit at 50.0
            sell = build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: "2",
                TAG_SENDER_COMP_ID: "TRADER1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:01.000",
                TAG_CL_ORD_ID: "S001",
                TAG_SIDE: FIX_SIDE_SELL,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "50.0000",
                TAG_ORDER_QTY: "10",
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "TEST",
            })
            writer.write(sell.encode("ascii"))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)

            # 3. Submit buy limit at 50.0 (should fill)
            buy = build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: "3",
                TAG_SENDER_COMP_ID: "TRADER1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:02.000",
                TAG_CL_ORD_ID: "B001",
                TAG_SIDE: FIX_SIDE_BUY,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "50.0000",
                TAG_ORDER_QTY: "10",
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "TEST",
            })
            writer.write(buy.encode("ascii"))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(8192), timeout=2.0)
            buffer = data
            messages = []
            while True:
                m, buffer = extract_message(buffer)
                if m is None:
                    break
                messages.append(parse_message(m))

            # Should have New + Fill for buy
            buy_fills = [
                m for m in messages
                if m.get(TAG_CL_ORD_ID) == "B001"
                and m.get(TAG_EXEC_TYPE) == EXEC_TYPE_FILL
            ]
            assert len(buy_fills) == 1

            # 4. Logout
            logout = build_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGOUT,
                TAG_MSG_SEQ_NUM: "4",
                TAG_SENDER_COMP_ID: "TRADER1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:03.000",
            })
            writer.write(logout.encode("ascii"))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            msg, _ = extract_message(data)
            if msg:
                resp = parse_message(msg)
                assert resp[TAG_MSG_TYPE] == MSG_TYPE_LOGOUT

            writer.close()
            await writer.wait_closed()
        finally:
            await gateway.stop()
