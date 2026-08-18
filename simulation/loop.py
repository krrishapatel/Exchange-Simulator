"""Simulation loop that drives agents against the matching engine."""



class SimulationLoop:
    """Drives N agents against the matching engine for T steps.

    Each step:
      1. Calls each agent's on_market_data() to get orders.
      2. Submits each order to the engine.
      3. Routes fills back to the relevant agents.
    """

    def __init__(self, agents: list, num_steps: int = 10000):
        """Initialize the simulation loop.

        Args:
            agents: List of BaseAgent instances to participate.
            num_steps: Number of simulation steps to run.
        """
        self.agents = agents
        self.num_steps = num_steps
        self.engine = None
        self.total_fills: list = []

    def run(self) -> dict:
        """Run the simulation and return results.

        Returns:
            Dictionary with:
                - 'pnl': dict mapping agent name to final PnL
                - 'inventory': dict mapping agent name to final inventory
                - 'fills': total number of fills across all agents
                - 'steps': number of steps completed
        """
        import exchange_simulator as ex

        self.engine = ex.MatchingEngine()
        self.total_fills = []

        # Build a map from order_id -> agent for fill routing
        order_to_agent: dict[int, object] = {}

        for step in range(self.num_steps):
            timestamp = step * 1_000_000  # 1ms per step in nanoseconds

            for agent in self.agents:
                orders = agent.on_market_data(self.engine, timestamp)

                for order in orders:
                    # Track which agent owns this order
                    order_to_agent[order.id] = agent

                    # Submit and collect fills
                    fills = self.engine.submit(order)

                    for fill in fills:
                        self.total_fills.append(fill)

                        # Route fill to taker
                        taker_agent = order_to_agent.get(fill.taker_order_id)
                        if taker_agent is not None:
                            taker_agent.on_fill(fill)

                        # Route fill to maker
                        maker_agent = order_to_agent.get(fill.maker_order_id)
                        if maker_agent is not None:
                            maker_agent.on_fill(fill)

        return {
            "pnl": {agent.name: agent.pnl for agent in self.agents},
            "inventory": {agent.name: agent.inventory for agent in self.agents},
            "fills": len(self.total_fills),
            "steps": self.num_steps,
        }
