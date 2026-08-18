"""FIX 4.4 message parser and builder.

Handles parsing raw FIX messages (tag=value delimited by SOH/0x01),
building FIX messages from dicts, and checksum/body-length computation.
"""

from __future__ import annotations

SOH = "\x01"

# Standard FIX message types
MSG_TYPE_HEARTBEAT = "0"
MSG_TYPE_TEST_REQUEST = "1"
MSG_TYPE_LOGON = "A"
MSG_TYPE_LOGOUT = "5"
MSG_TYPE_NEW_ORDER_SINGLE = "D"
MSG_TYPE_ORDER_CANCEL_REQUEST = "F"
MSG_TYPE_EXECUTION_REPORT = "8"

# Key tag numbers
TAG_BEGIN_STRING = "8"
TAG_BODY_LENGTH = "9"
TAG_MSG_TYPE = "35"
TAG_MSG_SEQ_NUM = "34"
TAG_SENDER_COMP_ID = "49"
TAG_TARGET_COMP_ID = "56"
TAG_SENDING_TIME = "52"
TAG_CHECKSUM = "10"
TAG_ENCRYPT_METHOD = "98"
TAG_HEARTBT_INT = "108"
TAG_TEST_REQ_ID = "112"
TAG_TEXT = "58"

# Order-related tags
TAG_CL_ORD_ID = "11"
TAG_ORIG_CL_ORD_ID = "41"
TAG_ORDER_ID = "37"
TAG_EXEC_ID = "17"
TAG_EXEC_TYPE = "150"
TAG_ORD_STATUS = "39"
TAG_SIDE = "54"
TAG_ORD_TYPE = "40"
TAG_PRICE = "44"
TAG_ORDER_QTY = "38"
TAG_TIME_IN_FORCE = "59"
TAG_LAST_PX = "31"
TAG_LAST_QTY = "32"
TAG_CUM_QTY = "14"
TAG_LEAVES_QTY = "151"
TAG_AVG_PX = "6"
TAG_SYMBOL = "55"
TAG_TRANSACT_TIME = "60"


class FixParseError(Exception):
    """Raised when a FIX message cannot be parsed."""
    pass


def compute_checksum(data: str) -> str:
    """Compute FIX checksum (sum of bytes mod 256, zero-padded to 3 digits)."""
    total = sum(ord(c) for c in data) % 256
    return f"{total:03d}"


def parse_message(raw: str) -> dict[str, str]:
    """Parse a raw FIX message string into a dict of tag -> value.

    The raw message uses SOH (0x01) as delimiter between tag=value pairs.
    Validates checksum if present.

    Args:
        raw: Raw FIX message string with SOH delimiters.

    Returns:
        Dictionary mapping tag numbers (as strings) to their values.

    Raises:
        FixParseError: If the message is malformed or checksum is invalid.
    """
    if not raw:
        raise FixParseError("Empty message")

    # Strip trailing SOH if present
    if raw.endswith(SOH):
        raw = raw[:-1]

    pairs = raw.split(SOH)
    fields: dict[str, str] = {}

    for pair in pairs:
        if "=" not in pair:
            raise FixParseError(f"Invalid field (no '='): {pair!r}")
        tag, value = pair.split("=", 1)
        if not tag:
            raise FixParseError(f"Empty tag in field: {pair!r}")
        fields[tag] = value

    # Validate checksum if present
    if TAG_CHECKSUM in fields:
        # The checksum covers everything up to but not including the 10= field.
        idx = (raw + SOH).rfind(f"{SOH}{TAG_CHECKSUM}=")
        # A checksum as the first field is malformed. Leave that to the
        # comparison below rather than guessing at a body here.
        body_for_checksum = "" if idx == -1 else raw[:idx] + SOH

        expected = compute_checksum(body_for_checksum)
        if fields[TAG_CHECKSUM] != expected:
            raise FixParseError(
                f"Checksum mismatch: got {fields[TAG_CHECKSUM]}, expected {expected}"
            )

    # Validate BeginString
    if TAG_BEGIN_STRING in fields and fields[TAG_BEGIN_STRING] != "FIX.4.4":
        raise FixParseError(
            f"Unsupported FIX version: {fields[TAG_BEGIN_STRING]}"
        )

    return fields


def build_message(fields: dict[str, str]) -> str:
    """Build a FIX message string from a dict of fields.

    Automatically computes BodyLength (tag 9) and Checksum (tag 10).
    BeginString (tag 8) must be provided.

    The field ordering follows FIX protocol requirements:
    - Tag 8 (BeginString) first
    - Tag 9 (BodyLength) second
    - Tag 35 (MsgType) third
    - Remaining fields in input order
    - Tag 10 (Checksum) last

    Args:
        fields: Dictionary of tag -> value. Tags 9 and 10 are computed automatically.

    Returns:
        Complete FIX message string with SOH delimiters.
    """
    # Ensure BeginString
    begin_string = fields.get(TAG_BEGIN_STRING, "FIX.4.4")

    # Build body (everything between BodyLength and Checksum)
    body_parts: list[str] = []

    # MsgType first in body
    if TAG_MSG_TYPE in fields:
        body_parts.append(f"{TAG_MSG_TYPE}={fields[TAG_MSG_TYPE]}")

    # Then remaining fields in order (excluding 8, 9, 10, 35)
    skip_tags = {TAG_BEGIN_STRING, TAG_BODY_LENGTH, TAG_CHECKSUM, TAG_MSG_TYPE}
    for tag, value in fields.items():
        if tag not in skip_tags:
            body_parts.append(f"{tag}={value}")

    body = SOH.join(body_parts) + SOH

    # Compute body length
    body_length = len(body)

    # Build header
    header = f"{TAG_BEGIN_STRING}={begin_string}{SOH}{TAG_BODY_LENGTH}={body_length}{SOH}"

    # Compute checksum over header + body
    message_without_checksum = header + body
    checksum = compute_checksum(message_without_checksum)

    return message_without_checksum + f"{TAG_CHECKSUM}={checksum}{SOH}"


def extract_message(buffer: bytes) -> tuple[str | None, bytes]:
    """Extract a complete FIX message from a byte buffer.

    Looks for a complete message (terminated by 10=XXX|SOH|).
    Returns the extracted message string and remaining buffer.

    Args:
        buffer: Byte buffer that may contain partial or complete messages.

    Returns:
        Tuple of (extracted_message_or_None, remaining_buffer).
    """
    text = buffer.decode("ascii", errors="replace")

    # Look for checksum field pattern: SOH10=XXXSOH
    # The checksum is always the last field
    import re
    pattern = re.compile(r"10=\d{3}" + SOH)
    match = pattern.search(text)

    if match:
        end = match.end()
        message = text[:end]
        remaining = buffer[end:]
        return message, remaining

    return None, buffer
