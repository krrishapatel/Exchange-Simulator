"""FIX 4.4 protocol gateway server.

Maps FIX protocol messages to the C++ matching engine, allowing external
trading systems to submit and cancel orders via standard FIX connectivity.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import exchange_simulator as ex

from gateway.fix_parser import (
    MSG_TYPE_NEW_ORDER_SINGLE,
    MSG_TYPE_ORDER_CANCEL_REQUEST,
    MSG_TYPE_EXECUTION_REPORT,
    TAG_MSG_TYPE,
    TAG_CL_ORD_ID,
    TAG_ORIG_CL_ORD_ID,
    TAG_ORDER_ID,
    TAG_EXEC_ID,
    TAG_EXEC_TYPE,
    TAG_ORD_STATUS,
    TAG_SIDE,
    TAG_ORD_TYPE,
    TAG_PRICE,
    TAG_ORDER_QTY,
    TAG_TIME_IN_FORCE,
    TAG_LAST_PX,
    TAG_LAST_QTY,
    TAG_CUM_QTY,
    TAG_LEAVES_QTY,
    TAG_AVG_PX,
    TAG_SYMBOL,
    TAG_TRANSACT_TIME,
    extract_message,
    FixParseError,
)
from gateway.fix_session import FixSession

logger = logging.getLogger(__name__)

# FIX Side values
FIX_SIDE_BUY = "1"
FIX_SIDE_SELL = "2"

# FIX OrdType values
FIX_ORD_TYPE_MARKET = "1"
FIX_ORD_TYPE_LIMIT = "2"

# FIX TimeInForce values
FIX_TIF_DAY = "0"
FIX_TIF_GTC = "1"
FIX_TIF_IOC = "3"
FIX_TIF_FOK = "4"

# FIX ExecType values
EXEC_TYPE_NEW = "0"
EXEC_TYPE_PARTIAL_FILL = "1"
EXEC_TYPE_FILL = "2"
EXEC_TYPE_CANCELED = "4"
EXEC_TYPE_REJECTED = "8"

# FIX OrdStatus values
ORD_STATUS_NEW = "0"
ORD_STATUS_PARTIALLY_FILLED = "1"
ORD_STATUS_FILLED = "2"
ORD_STATUS_CANCELED = "4"
ORD_STATUS_REJECTED = "8"

# Price scale factor (fixed-point conversion: FIX price * 10000 -> engine integer)
PRICE_SCALE = 10000


class ClientConnection:
    """Represents a single FIX client connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        session: FixSession,
        gateway: "FixGateway",
    ):
        self.reader = reader
        self.writer = writer
        self.session = session
        self.gateway = gateway
        self._buffer = b""
        # Maps ClOrdID -> internal order ID
        self._cl_ord_to_id: dict[str, int] = {}
        # Maps internal order ID -> ClOrdID
        self._id_to_cl_ord: dict[int, str] = {}
        # Track cumulative fills per order
        self._cum_qty: dict[str, int] = {}
        self._cum_value: dict[str, float] = {}
        # ClOrdID -> the order's own side, quantity, price and symbol. A fill
        # against a resting order has to be reported with that order's terms,
        # not with the terms of the incoming order that hit it.
        self._orders: dict[str, dict] = {}

    async def send_raw(self, data: str) -> None:
        """Send raw FIX message bytes to the client."""
        self.writer.write(data.encode("ascii"))
        await self.writer.drain()

    async def handle(self) -> None:
        """Main read loop for this client connection."""
        self.session.set_send_func(self.send_raw)

        try:
            while True:
                data = await self.reader.read(4096)
                if not data:
                    break

                self._buffer += data
                while True:
                    msg, self._buffer = extract_message(self._buffer)
                    if msg is None:
                        break
                    try:
                        await self.session.receive_message(msg)
                    except FixParseError as e:
                        logger.warning("Parse error from client: %s", e)
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self.session.disconnect()
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass

    async def on_application_message(self, fields: dict[str, str]) -> None:
        """Handle application-level FIX messages (orders, cancels)."""
        msg_type = fields.get(TAG_MSG_TYPE, "")

        if msg_type == MSG_TYPE_NEW_ORDER_SINGLE:
            await self._handle_new_order(fields)
        elif msg_type == MSG_TYPE_ORDER_CANCEL_REQUEST:
            await self._handle_cancel(fields)
        else:
            logger.warning("Unsupported message type: %s", msg_type)

    async def _handle_new_order(self, fields: dict[str, str]) -> None:
        """Map FIX NewOrderSingle to engine.submit()."""
        cl_ord_id = fields.get(TAG_CL_ORD_ID, "")
        if not cl_ord_id:
            await self._send_reject(cl_ord_id, "Missing ClOrdID")
            return

        # Map FIX side to engine Side
        fix_side = fields.get(TAG_SIDE, "")
        if fix_side == FIX_SIDE_BUY:
            side = ex.Side.Buy
        elif fix_side == FIX_SIDE_SELL:
            side = ex.Side.Sell
        else:
            await self._send_reject(cl_ord_id, f"Invalid Side: {fix_side}")
            return

        # Map FIX OrdType to engine OrderType
        fix_ord_type = fields.get(TAG_ORD_TYPE, "")
        if fix_ord_type == FIX_ORD_TYPE_MARKET:
            order_type = ex.OrderType.Market
        elif fix_ord_type == FIX_ORD_TYPE_LIMIT:
            order_type = ex.OrderType.Limit
        else:
            await self._send_reject(cl_ord_id, f"Invalid OrdType: {fix_ord_type}")
            return

        # Map FIX TimeInForce
        fix_tif = fields.get(TAG_TIME_IN_FORCE, "0")
        if fix_tif in (FIX_TIF_DAY, FIX_TIF_GTC):
            tif = ex.TimeInForce.GTC
        elif fix_tif == FIX_TIF_IOC:
            tif = ex.TimeInForce.IOC
        elif fix_tif == FIX_TIF_FOK:
            tif = ex.TimeInForce.FOK
        else:
            await self._send_reject(cl_ord_id, f"Invalid TimeInForce: {fix_tif}")
            return

        # Parse quantity
        try:
            quantity = int(fields.get(TAG_ORDER_QTY, "0"))
        except ValueError:
            await self._send_reject(cl_ord_id, "Invalid OrderQty")
            return

        if quantity <= 0:
            await self._send_reject(cl_ord_id, "OrderQty must be positive")
            return

        # Parse price (convert from decimal to fixed-point integer)
        price = 0
        if order_type == ex.OrderType.Limit:
            try:
                price_float = float(fields.get(TAG_PRICE, "0"))
                price = int(price_float * PRICE_SCALE)
            except ValueError:
                await self._send_reject(cl_ord_id, "Invalid Price")
                return

        # Generate internal order ID
        internal_id = self.gateway.next_order_id()
        self._cl_ord_to_id[cl_ord_id] = internal_id
        self._id_to_cl_ord[internal_id] = cl_ord_id
        self._cum_qty[cl_ord_id] = 0
        self._cum_value[cl_ord_id] = 0.0
        self._orders[cl_ord_id] = {
            "order_id": internal_id,
            "side": fix_side,
            "quantity": quantity,
            "price": price,
            "symbol": fields.get(TAG_SYMBOL, "N/A"),
        }
        self.gateway._order_owner[internal_id] = self

        # Build engine Order
        order = ex.Order()
        order.id = internal_id
        order.side = side
        order.type = order_type
        order.tif = tif
        order.quantity = quantity
        order.price = price

        # Submit to engine
        fills = self.gateway.engine.submit(order)

        # Send New acknowledgment
        await self._send_execution_report(
            cl_ord_id=cl_ord_id,
            order_id=str(internal_id),
            exec_type=EXEC_TYPE_NEW,
            ord_status=ORD_STATUS_NEW,
            side=fix_side,
            quantity=quantity,
            price=price,
            cum_qty=0,
            leaves_qty=quantity,
            last_px=0,
            last_qty=0,
            symbol=fields.get(TAG_SYMBOL, "N/A"),
        )

        # Process fills. Every fill has two sides, and both of them are somebody's
        # order: the incoming one, and the resting one it traded against. Both
        # owners get a report.
        for fill in fills:
            await self._report_fill(cl_ord_id, fill)

            maker_conn = self.gateway._order_owner.get(fill.maker_order_id)
            maker_cl_ord_id = (
                maker_conn._id_to_cl_ord.get(fill.maker_order_id)
                if maker_conn is not None
                else None
            )
            if maker_cl_ord_id is not None:
                await maker_conn._report_fill(maker_cl_ord_id, fill)
            else:
                # An order the gateway never submitted, so there is no session to
                # report to. Seeded books and direct engine use both land here.
                logger.debug(
                    "No FIX owner for maker order %d, fill not reported",
                    fill.maker_order_id,
                )

    async def _report_fill(self, cl_ord_id: str, fill) -> None:
        """Send one fill's ExecutionReport for the order named by `cl_ord_id`.

        Used for both sides of a trade, which is why it reads the order's terms
        out of `_orders` rather than taking them from whatever message is being
        handled. Sending a resting order's report with the incoming order's
        quantity is the bug this shape avoids.
        """
        order = self._orders.get(cl_ord_id)
        if order is None:
            logger.warning("Fill for untracked order %s, dropped", cl_ord_id)
            return

        self._cum_qty[cl_ord_id] = self._cum_qty.get(cl_ord_id, 0) + fill.quantity
        self._cum_value[cl_ord_id] = self._cum_value.get(cl_ord_id, 0.0) + (
            fill.quantity * (fill.price / PRICE_SCALE)
        )

        cum_qty = self._cum_qty[cl_ord_id]
        leaves_qty = order["quantity"] - cum_qty

        if leaves_qty <= 0:
            exec_type = EXEC_TYPE_FILL
            ord_status = ORD_STATUS_FILLED
        else:
            exec_type = EXEC_TYPE_PARTIAL_FILL
            ord_status = ORD_STATUS_PARTIALLY_FILLED

        await self._send_execution_report(
            cl_ord_id=cl_ord_id,
            order_id=str(order["order_id"]),
            exec_type=exec_type,
            ord_status=ord_status,
            side=order["side"],
            quantity=order["quantity"],
            price=order["price"],
            cum_qty=cum_qty,
            leaves_qty=max(0, leaves_qty),
            last_px=fill.price,
            last_qty=fill.quantity,
            symbol=order["symbol"],
        )

    async def _handle_cancel(self, fields: dict[str, str]) -> None:
        """Map FIX OrderCancelRequest to engine.cancel()."""
        cl_ord_id = fields.get(TAG_CL_ORD_ID, "")
        orig_cl_ord_id = fields.get(TAG_ORIG_CL_ORD_ID, "")

        if not orig_cl_ord_id:
            orig_cl_ord_id = cl_ord_id

        # Look up internal ID
        internal_id = self._cl_ord_to_id.get(orig_cl_ord_id)
        if internal_id is None:
            await self._send_cancel_reject(cl_ord_id, orig_cl_ord_id, "Unknown order")
            return

        # Cancel in engine
        result = self.gateway.engine.cancel(internal_id)

        if result == ex.CancelResult.Success:
            fix_side = fields.get(TAG_SIDE, FIX_SIDE_BUY)
            cum_qty = self._cum_qty.get(orig_cl_ord_id, 0)

            await self._send_execution_report(
                cl_ord_id=cl_ord_id,
                order_id=str(internal_id),
                exec_type=EXEC_TYPE_CANCELED,
                ord_status=ORD_STATUS_CANCELED,
                side=fix_side,
                quantity=0,
                price=0,
                cum_qty=cum_qty,
                leaves_qty=0,
                last_px=0,
                last_qty=0,
                symbol=fields.get(TAG_SYMBOL, "N/A"),
            )
        else:
            await self._send_cancel_reject(
                cl_ord_id, orig_cl_ord_id,
                f"Cancel failed: {result.name}"
            )

    async def _send_execution_report(
        self,
        cl_ord_id: str,
        order_id: str,
        exec_type: str,
        ord_status: str,
        side: str,
        quantity: int,
        price: int,
        cum_qty: int,
        leaves_qty: int,
        last_px: int,
        last_qty: int,
        symbol: str,
    ) -> None:
        """Send an ExecutionReport to the client."""
        exec_id = str(self.gateway.next_exec_id())

        avg_px = 0.0
        if cum_qty > 0 and cl_ord_id in self._cum_value:
            avg_px = self._cum_value[cl_ord_id] / cum_qty

        fields = {
            TAG_MSG_TYPE: MSG_TYPE_EXECUTION_REPORT,
            TAG_ORDER_ID: order_id,
            TAG_EXEC_ID: exec_id,
            TAG_EXEC_TYPE: exec_type,
            TAG_ORD_STATUS: ord_status,
            TAG_SYMBOL: symbol,
            TAG_SIDE: side,
            TAG_ORDER_QTY: str(quantity),
            TAG_LAST_QTY: str(last_qty),
            TAG_LAST_PX: f"{last_px / PRICE_SCALE:.4f}" if last_px else "0.0000",
            TAG_CUM_QTY: str(cum_qty),
            TAG_LEAVES_QTY: str(leaves_qty),
            TAG_AVG_PX: f"{avg_px:.4f}",
            TAG_CL_ORD_ID: cl_ord_id,
            TAG_TRANSACT_TIME: datetime.now(timezone.utc).strftime(
                "%Y%m%d-%H:%M:%S.%f"
            )[:-3],
        }
        if price:
            fields[TAG_PRICE] = f"{price / PRICE_SCALE:.4f}"

        await self.session.send_message(fields)

    async def _send_reject(self, cl_ord_id: str, reason: str) -> None:
        """Send a rejected ExecutionReport."""
        exec_id = str(self.gateway.next_exec_id())
        fields = {
            TAG_MSG_TYPE: MSG_TYPE_EXECUTION_REPORT,
            TAG_ORDER_ID: "NONE",
            TAG_EXEC_ID: exec_id,
            TAG_EXEC_TYPE: EXEC_TYPE_REJECTED,
            TAG_ORD_STATUS: ORD_STATUS_REJECTED,
            TAG_CL_ORD_ID: cl_ord_id,
            TAG_SYMBOL: "N/A",
            TAG_SIDE: FIX_SIDE_BUY,
            TAG_ORDER_QTY: "0",
            TAG_LAST_QTY: "0",
            TAG_LAST_PX: "0.0000",
            TAG_CUM_QTY: "0",
            TAG_LEAVES_QTY: "0",
            TAG_AVG_PX: "0.0000",
            "58": reason,
            TAG_TRANSACT_TIME: datetime.now(timezone.utc).strftime(
                "%Y%m%d-%H:%M:%S.%f"
            )[:-3],
        }
        await self.session.send_message(fields)

    async def _send_cancel_reject(
        self, cl_ord_id: str, orig_cl_ord_id: str, reason: str
    ) -> None:
        """Send a cancel-rejected ExecutionReport."""
        exec_id = str(self.gateway.next_exec_id())
        fields = {
            TAG_MSG_TYPE: MSG_TYPE_EXECUTION_REPORT,
            TAG_ORDER_ID: "NONE",
            TAG_EXEC_ID: exec_id,
            TAG_EXEC_TYPE: EXEC_TYPE_REJECTED,
            TAG_ORD_STATUS: ORD_STATUS_REJECTED,
            TAG_CL_ORD_ID: cl_ord_id,
            TAG_ORIG_CL_ORD_ID: orig_cl_ord_id,
            TAG_SYMBOL: "N/A",
            TAG_SIDE: FIX_SIDE_BUY,
            TAG_ORDER_QTY: "0",
            TAG_LAST_QTY: "0",
            TAG_LAST_PX: "0.0000",
            TAG_CUM_QTY: "0",
            TAG_LEAVES_QTY: "0",
            TAG_AVG_PX: "0.0000",
            "58": reason,
            TAG_TRANSACT_TIME: datetime.now(timezone.utc).strftime(
                "%Y%m%d-%H:%M:%S.%f"
            )[:-3],
        }
        await self.session.send_message(fields)


class FixGateway:
    """FIX 4.4 protocol gateway server.

    Accepts TCP connections from FIX clients, manages sessions,
    and routes orders to the matching engine.

    Usage:
        engine = exchange_simulator.MatchingEngine()
        gateway = FixGateway(engine, port=9876, comp_id="EXCHANGE")
        await gateway.start()
        # ... gateway is now accepting connections ...
        await gateway.stop()
    """

    def __init__(
        self,
        engine: Any = None,
        port: int = 9876,
        host: str = "0.0.0.0",
        comp_id: str = "EXCHANGE",
    ):
        """Initialize the FIX gateway.

        Args:
            engine: MatchingEngine instance. If None, creates a new one.
            port: TCP port to listen on.
            host: Host/IP to bind to.
            comp_id: CompID for this gateway (SenderCompID in outgoing messages).
        """
        self.engine = engine if engine is not None else ex.MatchingEngine()
        self.port = port
        self.host = host
        self.comp_id = comp_id
        self._server: asyncio.Server | None = None
        self._connections: list[ClientConnection] = []
        # Internal order ID -> the connection that submitted it. Order books are
        # shared across sessions, so the resting side of a fill usually belongs
        # to a different client than the incoming order, and this is what makes
        # its execution report reachable.
        self._order_owner: dict[int, ClientConnection] = {}
        self._order_id_counter = 0
        self._exec_id_counter = 0

    def next_order_id(self) -> int:
        """Generate the next unique order ID."""
        self._order_id_counter += 1
        return self._order_id_counter

    def next_exec_id(self) -> int:
        """Generate the next unique execution ID."""
        self._exec_id_counter += 1
        return self._exec_id_counter

    async def start(self) -> None:
        """Start the FIX gateway server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )
        logger.info("FIX gateway listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the FIX gateway server and close all connections.

        Order matters. Since Python 3.12, Server.wait_closed() waits on the
        connection handlers, and a handler sits in a read that only returns once
        its client goes away, so awaiting the server while a client is still
        attached hangs for as long as that client stays. Closing the client
        connections first lets the handlers reach their exit.

        It is not a barrier in the other direction, though: wait_closed() can
        return before every handler has actually run its cleanup, which is why
        anything this method needs to be true afterwards is done here rather than
        left to the handlers.
        """
        # list() because a handler removes itself from _connections as it exits.
        for conn in list(self._connections):
            conn.writer.close()
            try:
                await conn.writer.wait_closed()
            except Exception:
                pass
        self._connections.clear()

        # Every session is going away, so no order has anywhere to be reported
        # to. Cleared here rather than left to the per-connection cleanup, per the
        # ordering caveat above.
        self._order_owner.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a new incoming TCP connection."""
        peer = writer.get_extra_info("peername")
        logger.info("New connection from %s", peer)

        session = FixSession(
            sender_comp_id=self.comp_id,
            target_comp_id=None,  # Will be set on logon
            heartbeat_interval=30,
        )

        conn = ClientConnection(reader, writer, session, self)
        session._on_message = conn.on_application_message
        self._connections.append(conn)

        try:
            await conn.handle()
        finally:
            if conn in self._connections:
                self._connections.remove(conn)
            # Drop this connection's orders from the owner map. Without this the
            # map grows for the life of the process and a later fill tries to
            # report down a socket that is already gone.
            for order_id in list(conn._id_to_cl_ord):
                if self._order_owner.get(order_id) is conn:
                    del self._order_owner[order_id]
            logger.info("Connection closed from %s", peer)

    async def serve_forever(self) -> None:
        """Start and serve until cancelled."""
        await self.start()
        if self._server:
            await self._server.serve_forever()
