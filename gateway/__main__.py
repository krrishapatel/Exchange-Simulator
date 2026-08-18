"""Entry point for running the FIX gateway via `python -m gateway`."""

import argparse
import asyncio
import logging

import exchange_simulator as ex

from gateway.fix_gateway import FixGateway


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIX 4.4 protocol gateway for the exchange simulator"
    )
    parser.add_argument(
        "--port", type=int, default=9876, help="TCP port to listen on (default: 9876)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host/IP to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--comp-id",
        type=str,
        default="EXCHANGE",
        help="CompID for the gateway (default: EXCHANGE)",
    )
    parser.add_argument(
        "--heartbeat",
        type=int,
        default=30,
        help="Heartbeat interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = ex.MatchingEngine()
    gateway = FixGateway(
        engine=engine,
        port=args.port,
        host=args.host,
        comp_id=args.comp_id,
    )

    print(f"Starting FIX 4.4 gateway on {args.host}:{args.port}")
    print(f"CompID: {args.comp_id}, Heartbeat: {args.heartbeat}s")
    print("Press Ctrl+C to stop")

    try:
        asyncio.run(gateway.serve_forever())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
