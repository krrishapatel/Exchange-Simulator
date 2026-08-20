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
    TAG_LEAVES_QTY,
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
    EXEC_TYPE_PARTIAL_FILL,
    EXEC_TYPE_CANCELED,
    ORD_STATUS_NEW,
    ORD_STATUS_FILLED,
    ORD_STATUS_PARTIALLY_FILLED,
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
# These run against a live TCP connection on an ephemeral port, so they need no
# fixed port and no fixed timing.


async def read_messages(reader, count=1, timeout=2.0, buffer=b""):
    """Read from the socket until `count` complete FIX messages have arrived.

    TCP preserves no message boundaries, so one read() can return half a
    message, one message, or several. Waiting on a message count instead of a
    read count is what makes these tests deterministic: an order that produces
    both a New ack and a Fill used to look like a missing fill purely because
    the two reports landed in different segments.

    Returns (messages, leftover) and the leftover must be threaded into the next
    call, or a message that arrived early is silently dropped.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    messages = []

    while True:
        while True:
            raw, buffer = extract_message(buffer)
            if raw is None:
                break
            messages.append(parse_message(raw))
        if len(messages) >= count:
            return messages, buffer

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(
                f"expected {count} FIX message(s), got {len(messages)}: {messages}"
            )
        data = await asyncio.wait_for(reader.read(4096), timeout=remaining)
        if not data:
            raise AssertionError(
                f"gateway closed the connection after {len(messages)} message(s)"
            )
        buffer += data


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
            (ack,), buf = await read_messages(reader)
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
            (er,), buf = await read_messages(reader, buffer=buf)
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
            _, buf = await read_messages(reader)

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
            _, buf = await read_messages(reader, buffer=buf)

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

            # A trade has two sides, so expect three reports: the New ack for the
            # incoming buy, its fill, and a fill for the resting sell it hit.
            messages, buf = await read_messages(reader, count=3, buffer=buf)

            for cl_ord_id in ("buy1", "sell1"):
                fill_reports = [
                    m for m in messages
                    if m.get(TAG_EXEC_TYPE) == EXEC_TYPE_FILL
                    and m.get(TAG_CL_ORD_ID) == cl_ord_id
                ]
                assert len(fill_reports) == 1, cl_ord_id
                fill = fill_reports[0]
                assert fill[TAG_ORD_STATUS] == ORD_STATUS_FILLED
                assert fill[TAG_LAST_QTY] == "25"
                assert fill[TAG_CUM_QTY] == "25"
                assert float(fill[TAG_LAST_PX]) == pytest.approx(100.0, abs=0.001)

            # Each side keeps its own side tag, not the incoming order's.
            by_id = {m[TAG_CL_ORD_ID]: m for m in messages
                     if m.get(TAG_EXEC_TYPE) == EXEC_TYPE_FILL}
            assert by_id["buy1"][TAG_SIDE] == FIX_SIDE_BUY
            assert by_id["sell1"][TAG_SIDE] == FIX_SIDE_SELL

            writer.close()
            await writer.wait_closed()
        finally:
            await gateway.stop()

    @pytest.mark.asyncio
    async def test_fill_reaches_the_resting_order_on_another_session(self, gateway):
        """The resting side of a trade is told about it, on its own connection.

        The order books are shared, so the two sides of a fill are usually two
        different clients. This is the case a per-connection order map cannot
        serve: the incoming order's session has never heard of the resting order.
        """
        await gateway.start()
        port = gateway._server.sockets[0].getsockname()[1]

        async def logon(comp_id):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(build_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGON,
                TAG_MSG_SEQ_NUM: "1",
                TAG_SENDER_COMP_ID: comp_id,
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:00.000",
                TAG_ENCRYPT_METHOD: "0",
                TAG_HEARTBT_INT: "30",
            }).encode("ascii"))
            await writer.drain()
            _, buf = await read_messages(reader)
            return reader, writer, buf

        def order(comp_id, seq, cl_ord_id, side, qty):
            return build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: str(seq),
                TAG_SENDER_COMP_ID: comp_id,
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:01.000",
                TAG_CL_ORD_ID: cl_ord_id,
                TAG_SIDE: side,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "100.0000",
                TAG_ORDER_QTY: str(qty),
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "AAPL",
            }).encode("ascii")

        try:
            maker_r, maker_w, maker_buf = await logon("MAKER")
            taker_r, taker_w, taker_buf = await logon("TAKER")

            # MAKER rests a sell, and reads only its New ack for now.
            maker_w.write(order("MAKER", 2, "rest1", FIX_SIDE_SELL, 40))
            await maker_w.drain()
            (ack,), maker_buf = await read_messages(maker_r, buffer=maker_buf)
            assert ack[TAG_EXEC_TYPE] == EXEC_TYPE_NEW
            assert ack[TAG_CL_ORD_ID] == "rest1"

            # TAKER hits part of it from a different session.
            taker_w.write(order("TAKER", 2, "hit1", FIX_SIDE_BUY, 15))
            await taker_w.drain()
            taker_msgs, taker_buf = await read_messages(
                taker_r, count=2, buffer=taker_buf
            )
            taker_fills = [
                m for m in taker_msgs if m.get(TAG_EXEC_TYPE) == EXEC_TYPE_FILL
            ]
            assert len(taker_fills) == 1
            assert taker_fills[0][TAG_CL_ORD_ID] == "hit1"
            assert taker_fills[0][TAG_ORD_STATUS] == ORD_STATUS_FILLED
            assert taker_fills[0][TAG_LAST_QTY] == "15"

            # MAKER hears about it without having sent anything, as a partial:
            # 15 of its 40 traded, so 25 are still working.
            (maker_fill,), maker_buf = await read_messages(
                maker_r, buffer=maker_buf
            )
            assert maker_fill[TAG_CL_ORD_ID] == "rest1"
            assert maker_fill[TAG_EXEC_TYPE] == EXEC_TYPE_PARTIAL_FILL
            assert maker_fill[TAG_ORD_STATUS] == ORD_STATUS_PARTIALLY_FILLED
            assert maker_fill[TAG_LAST_QTY] == "15"
            assert maker_fill[TAG_CUM_QTY] == "15"
            assert maker_fill[TAG_LEAVES_QTY] == "25"
            # Its own terms, not the incoming order's.
            assert maker_fill[TAG_SIDE] == FIX_SIDE_SELL
            assert maker_fill[TAG_ORDER_QTY] == "40"

            for w in (maker_w, taker_w):
                w.close()
                await w.wait_closed()
        finally:
            await gateway.stop()

        # Both sessions are gone, so nothing is left pointing at a dead socket.
        assert gateway._order_owner == {}

    @pytest.mark.asyncio
    async def test_a_disconnect_deregisters_that_session_s_orders(self, gateway):
        """One client leaving takes its orders out of the owner map.

        The gateway stays up here, so this is the case `stop()` does not cover:
        without it the map grows for the life of the process and a later fill
        tries to report down a socket that is already gone.
        """
        await gateway.start()
        port = gateway._server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(build_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGON,
                TAG_MSG_SEQ_NUM: "1",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:00.000",
                TAG_ENCRYPT_METHOD: "0",
                TAG_HEARTBT_INT: "30",
            }).encode("ascii"))
            await writer.drain()
            _, buf = await read_messages(reader)

            writer.write(build_message({
                TAG_MSG_TYPE: MSG_TYPE_NEW_ORDER_SINGLE,
                TAG_MSG_SEQ_NUM: "2",
                TAG_SENDER_COMP_ID: "CLIENT1",
                TAG_TARGET_COMP_ID: "EXCHANGE",
                TAG_SENDING_TIME: "20260707-12:00:01.000",
                TAG_CL_ORD_ID: "leaky",
                TAG_SIDE: FIX_SIDE_SELL,
                TAG_ORD_TYPE: FIX_ORD_TYPE_LIMIT,
                TAG_PRICE: "100.0000",
                TAG_ORDER_QTY: "10",
                TAG_TIME_IN_FORCE: FIX_TIF_DAY,
                TAG_SYMBOL: "AAPL",
            }).encode("ascii"))
            await writer.drain()
            _, buf = await read_messages(reader, buffer=buf)
            assert gateway._order_owner, "order was never registered"

            writer.close()
            await writer.wait_closed()

            # The handler's exit is not ordered against this coroutine, so poll
            # rather than sleep a guessed interval.
            for _ in range(200):
                if not gateway._order_owner:
                    break
                await asyncio.sleep(0.01)
            assert gateway._order_owner == {}
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
            _, buf = await read_messages(reader)

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
            _, buf = await read_messages(reader, buffer=buf)

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
            (er,), buf = await read_messages(reader, buffer=buf)
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
            (ack,), buf = await read_messages(reader)
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
            _, buf = await read_messages(reader, buffer=buf)

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
            # New ack for the buy, its fill, and the resting sell's fill.
            messages, buf = await read_messages(reader, count=3, buffer=buf)

            for cl_ord_id in ("B001", "S001"):
                fills = [
                    m for m in messages
                    if m.get(TAG_CL_ORD_ID) == cl_ord_id
                    and m.get(TAG_EXEC_TYPE) == EXEC_TYPE_FILL
                ]
                assert len(fills) == 1, cl_ord_id
                assert fills[0][TAG_ORD_STATUS] == ORD_STATUS_FILLED

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
            (resp,), buf = await read_messages(reader, buffer=buf)
            assert resp[TAG_MSG_TYPE] == MSG_TYPE_LOGOUT

            writer.close()
            await writer.wait_closed()
        finally:
            await gateway.stop()
