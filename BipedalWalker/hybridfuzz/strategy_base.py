class FuzzingStrategy:
    name: str = "base"

    def initialize(self, config):
        """Initialize any required models, structures, or state for the strategy."""
        pass

    def mutate_or_generate(self, seed):
        """
        Given a seed from the shared pool, generate a new candidate testcase.
        seed can be None if the pool is empty or if this strategy generates from scratch.
        Returns the raw testcase (e.g. numpy array of states).
        """
        raise NotImplementedError

    def update(self, candidate, result, features):
        """
        Update the internal state of the strategy based on execution feedback.
        """
        pass

    def compute_feedback(self, candidate, result, features):
        """
        Compute scores (like novelty, uncertainty) that this strategy is responsible for.
        Returns a dict of scores, e.g., {'novelty_score': 0.5, 'diversity_score': 0.2}
        """
        return {}

    def get_status(self):
        """Return lightweight diagnostics for experiment summaries."""
        return {}
