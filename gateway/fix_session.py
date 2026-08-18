"""FIX 4.4 session management.

Handles sequence numbers, heartbeats, session state machine,
and CompID validation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Callable, Awaitable

from gateway.fix_parser import (
    MSG_TYPE_HEARTBEAT,
    MSG_TYPE_TEST_REQUEST,
    MSG_TYPE_LOGON,
    MSG_TYPE_LOGOUT,
    TAG_BEGIN_STRING,
    TAG_MSG_TYPE,
    TAG_MSG_SEQ_NUM,
    TAG_SENDER_COMP_ID,
    TAG_TARGET_COMP_ID,
    TAG_SENDING_TIME,
    TAG_ENCRYPT_METHOD,
    TAG_HEARTBT_INT,
    TAG_TEST_REQ_ID,
    TAG_TEXT,
    build_message,
    parse_message,
)


class SessionState(Enum):
    """FIX session state machine states."""
    DISCONNECTED = auto()
    LOGON_RECEIVED = auto()
    ACTIVE = auto()
    LOGOUT = auto()


class FixSession:
    """Manages a single FIX session with sequence numbers and heartbeats.

    Attributes:
        sender_comp_id: This side's CompID (the exchange/gateway).
        target_comp_id: The counterparty's CompID (the client).
        state: Current session state.
        heartbeat_interval: Heartbeat interval in seconds.
    """

    def __init__(
        self,
        sender_comp_id: str,
        target_comp_id: str | None = None,
        heartbeat_interval: int = 30,
        on_message: Callable[[dict[str, str]], Awaitable[None]] | None = None,
    ):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.heartbeat_interval = heartbeat_interval
        self.state = SessionState.DISCONNECTED
        self._outgoing_seq_num = 0
        self._incoming_seq_num = 0
        self._on_message = on_message
        self._last_sent_time: float = 0.0
        self._last_recv_time: float = 0.0
        self._heartbeat_task: asyncio.Task | None = None
        self._send_func: Callable[[str], Awaitable[None]] | None = None

    @property
    def outgoing_seq_num(self) -> int:
        """Current outgoing sequence number (last used)."""
        return self._outgoing_seq_num

    @property
    def incoming_seq_num(self) -> int:
        """Current incoming sequence number (last received)."""
        return self._incoming_seq_num

    def _next_seq_num(self) -> int:
        """Get and increment the outgoing sequence number."""
        self._outgoing_seq_num += 1
        return self._outgoing_seq_num

    def _sending_time(self) -> str:
        """Get current UTC time in FIX format."""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]

    def set_send_func(self, func: Callable[[str], Awaitable[None]]) -> None:
        """Set the function used to send raw FIX messages to the wire."""
        self._send_func = func

    async def send_message(self, fields: dict[str, str]) -> None:
        """Build and send a FIX message with proper session-level fields.

        Automatically adds BeginString, MsgSeqNum, SenderCompID,
        TargetCompID, and SendingTime.
        """
        fields[TAG_BEGIN_STRING] = "FIX.4.4"
        fields[TAG_MSG_SEQ_NUM] = str(self._next_seq_num())
        fields[TAG_SENDER_COMP_ID] = self.sender_comp_id
        if self.target_comp_id:
            fields[TAG_TARGET_COMP_ID] = self.target_comp_id
        fields[TAG_SENDING_TIME] = self._sending_time()

        raw = build_message(fields)
        self._last_sent_time = time.time()

        if self._send_func:
            await self._send_func(raw)

    async def receive_message(self, raw: str) -> None:
        """Process an incoming FIX message.

        Validates session-level fields, updates sequence numbers,
        handles heartbeat/test-request, and dispatches application messages.

        Args:
            raw: Raw FIX message string.

        Raises:
            FixParseError: If message is malformed.
        """
        fields = parse_message(raw)
        self._last_recv_time = time.time()

        # Validate and update sequence number
        if TAG_MSG_SEQ_NUM in fields:
            seq = int(fields[TAG_MSG_SEQ_NUM])
            self._incoming_seq_num = seq

        msg_type = fields.get(TAG_MSG_TYPE, "")

        # CompID validation (after logon establishes target)
        if self.state == SessionState.ACTIVE:
            if self.target_comp_id:
                sender = fields.get(TAG_SENDER_COMP_ID, "")
                if sender != self.target_comp_id:
                    await self._send_reject(
                        f"Invalid SenderCompID: expected {self.target_comp_id}, got {sender}"
                    )
                    return

        # Handle session-level messages
        if msg_type == MSG_TYPE_LOGON:
            await self._handle_logon(fields)
        elif msg_type == MSG_TYPE_LOGOUT:
            await self._handle_logout(fields)
        elif msg_type == MSG_TYPE_HEARTBEAT:
            pass  # Just update last_recv_time (done above)
        elif msg_type == MSG_TYPE_TEST_REQUEST:
            await self._handle_test_request(fields)
        else:
            # Application-level message
            if self.state != SessionState.ACTIVE:
                await self._send_reject("Session not active")
                return
            if self._on_message:
                await self._on_message(fields)

    async def _handle_logon(self, fields: dict[str, str]) -> None:
        """Handle incoming Logon message."""
        # Extract client CompID
        client_comp_id = fields.get(TAG_SENDER_COMP_ID, "")

        # If we have a target set, validate it
        if self.target_comp_id and client_comp_id != self.target_comp_id:
            await self._send_reject(
                f"CompID mismatch: expected {self.target_comp_id}, got {client_comp_id}"
            )
            await self._send_logout("Invalid CompID")
            return

        # Accept the logon - set target if not already set
        if not self.target_comp_id:
            self.target_comp_id = client_comp_id

        # Get heartbeat interval from client
        if TAG_HEARTBT_INT in fields:
            self.heartbeat_interval = int(fields[TAG_HEARTBT_INT])

        self.state = SessionState.ACTIVE

        # Send logon acknowledgment
        await self.send_message({
            TAG_MSG_TYPE: MSG_TYPE_LOGON,
            TAG_ENCRYPT_METHOD: "0",
            TAG_HEARTBT_INT: str(self.heartbeat_interval),
        })

        # Start heartbeat monitoring
        self._start_heartbeat()

    async def _handle_logout(self, fields: dict[str, str]) -> None:
        """Handle incoming Logout message."""
        if self.state == SessionState.ACTIVE:
            # Send logout acknowledgment
            await self.send_message({
                TAG_MSG_TYPE: MSG_TYPE_LOGOUT,
            })

        self.state = SessionState.LOGOUT
        self._stop_heartbeat()

    async def _handle_test_request(self, fields: dict[str, str]) -> None:
        """Handle incoming TestRequest - respond with Heartbeat."""
        test_req_id = fields.get(TAG_TEST_REQ_ID, "")
        heartbeat_fields: dict[str, str] = {TAG_MSG_TYPE: MSG_TYPE_HEARTBEAT}
        if test_req_id:
            heartbeat_fields[TAG_TEST_REQ_ID] = test_req_id
        await self.send_message(heartbeat_fields)

    async def _send_reject(self, reason: str) -> None:
        """Send a Logout with rejection reason."""
        await self.send_message({
            TAG_MSG_TYPE: MSG_TYPE_LOGOUT,
            TAG_TEXT: reason,
        })

    async def _send_logout(self, text: str = "") -> None:
        """Send a Logout message."""
        fields: dict[str, str] = {TAG_MSG_TYPE: MSG_TYPE_LOGOUT}
        if text:
            fields[TAG_TEXT] = text
        await self.send_message(fields)
        self.state = SessionState.LOGOUT
        self._stop_heartbeat()

    def _start_heartbeat(self) -> None:
        """Start the heartbeat monitoring task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat monitoring task."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeats if no other messages sent."""
        try:
            while self.state == SessionState.ACTIVE:
                await asyncio.sleep(self.heartbeat_interval)
                if self.state != SessionState.ACTIVE:
                    break
                # Send heartbeat if we haven't sent anything recently
                elapsed = time.time() - self._last_sent_time
                if elapsed >= self.heartbeat_interval:
                    await self.send_message({
                        TAG_MSG_TYPE: MSG_TYPE_HEARTBEAT,
                    })
        except asyncio.CancelledError:
            pass

    def disconnect(self) -> None:
        """Mark session as disconnected and clean up."""
        self.state = SessionState.DISCONNECTED
        self._stop_heartbeat()
