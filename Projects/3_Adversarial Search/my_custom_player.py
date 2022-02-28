from sample_players import DataPlayer, BasePlayer
import math
import random
from isolation import Isolation


class Stats:
    def __init__(self, state: Isolation, player_id: int, visit_count=0, win_score=0):
        self.state = state
        self.player_id = player_id
        self.visit_count = visit_count
        self.win_score = win_score

    def actions(self):
        return self.state.actions()

    def random(self):
        return random.choice(self.actions())


class Node:
    def __init__(self, stats: Stats, parent=None, action=None):
        self.stats = stats
        self.action = action
        self.parent = parent
        self.children = []

    def best_child(self):
        return max(self.children, key=lambda c: c.stats.visit_count)


class UCT:

    @staticmethod
    def uct(total_visit: int, win_score: float, node_visit: int):
        if node_visit == 0:
            return float('inf')
        return win_score / node_visit + 0.5 * 1.41 * math.sqrt(math.log(total_visit) / node_visit)

    @staticmethod
    def best_node(node: Node) -> Node:
        parent_visit = node.stats.visit_count
        return max(node.children, key=lambda c: UCT.uct(parent_visit, c.stats.win_score, c.stats.visit_count))


class MCTS:

    def next_move(self, board: Isolation, player_id: int):
        stats = Stats(board, player_id)
        root = Node(stats)
        last = None
        while True:
            promising = self.promising_action(root)
            if not MCTS.is_terminal(promising):
                MCTS.expand_node(promising)
            exploration_node = promising
            if len(promising.children) > 0:
                exploration_node = random.choice(promising.children)
            result = MCTS.random_playout(exploration_node)
            MCTS.back_prop(exploration_node, result)
            action_to_yield = root.best_child().action
            # print("returned action", action_to_yield)
            last = action_to_yield
            yield action_to_yield

    @staticmethod
    def back_prop(node: Node, score: int):
        temp_node = node
        while temp_node is not None:
            temp_node.stats.visit_count += 1
            temp_node.stats.win_score += score
            temp_node = temp_node.parent

    @staticmethod
    def random_playout(node: Node):
        temp_node = node
        if temp_node.stats.state.terminal_test():
            score = 1 if temp_node.stats.state.utility(temp_node.stats.player_id) > 0 else -1
            temp_node.stats.win_score += score
            return score
        temp_state = node.stats.state
        while not temp_state.terminal_test():
            temp_state = temp_state.result(random.choice(temp_state.actions()))
        return 1 if temp_state.utility(temp_node.stats.player_id) > 0 else -1

    @staticmethod
    def is_terminal(node: Node) -> bool:
        return node.stats.state.terminal_test()

    @staticmethod
    def promising_action(node: Node) -> Node:
        while len(node.children) != 0:
            node = UCT.best_node(node)
        return node

    @staticmethod
    def expand_node(node: Node):
        actions = node.stats.actions()
        for action in actions:
            stats = Stats(node.stats.state.result(action), node.stats.player_id)
            next_node = Node(stats, node, action)
            node.children.append(next_node)


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state: Isolation):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        # AlphaBetaPlayer
        # random_action = random.choice(state.actions())
        # self.queue.put(random_action)
        # if state.ply_count < 2:
        #     return random_action
        # if not state.terminal_test():
        #     for action in AlphaBetaPlayer(self.player_id).alpha_beta_iter(state):
        #         self.queue.put(action)

        # Monte Carlo
        random_choice = random.choice(state.actions())
        self.queue.put(random_choice)
        if state.ply_count < 2:
            return random_choice
        # map = {}
        if not state.terminal_test():
            for action in MCTS().next_move(state, self.player_id):
                # if map.get(action) is not None:
                #     map[action] += 1
                # else:
                #     map[action] =
                self.queue.put(action)


class AlphaBetaPlayer(BasePlayer):
    """ Implement an agent using any combination of techniques discussed
    in lecture (or that you find online on your own) that can beat
    sample_players.GreedyPlayer in >80% of "fair" matches (see tournament.py
    or readme for definition of fair matches).

    Implementing get_action() is the only required method, but you can add any
    other methods you want to perform minimax/alpha-beta/monte-carlo tree search,
    etc.

    **********************************************************************
    NOTE: The test cases will NOT be run on a machine with GPU access, or
          be suitable for using any other machine learning techniques.
    **********************************************************************
    """

    def get_action(self, state):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move at a fixed search depth of 3 plies
        random_action = random.choice(state.actions())
        self.queue.put(random_action)
        if state.ply_count < 2:
            return random_action
        else:
            for action in self.alpha_beta_iter(state):
                self.queue.put(action)

    def alpha_beta_iter(self, state):
        depth = 1
        while True:
            depth += 1
            yield self.alpha_beta(state, depth)

    def alpha_beta(self, state, depth):

        def min_value(m_state, m_depth, m_alpha, m_beta):
            if m_state.terminal_test():
                return m_state.utility(self.player_id)
            if m_depth <= 0:
                return self.score(m_state)
            value = float("inf")
            for action in m_state.actions():
                value = min(value, max_value(m_state.result(action), m_depth - 1, m_alpha, m_beta))
                if value <= m_alpha:
                    return value
                m_beta = min(m_beta, value)
            return value

        def max_value(m_state, m_depth, m_alpha, m_beta):
            if m_state.terminal_test():
                return m_state.utility(self.player_id)
            if m_depth <= 0:
                return self.score(m_state)
            value = float("-inf")
            for action in m_state.actions():
                value = max(value, min_value(m_state.result(action), m_depth - 1, m_alpha, m_beta))
                if value >= m_beta:
                    return value
                m_alpha = max(m_alpha, value)
            return value

        alpha = float('-inf')
        beta = float('inf')
        best_score = float('-inf')
        best_move = None
        for a in state.actions():
            v = min_value(state.result(a), depth - 1, alpha, beta)
            if v > best_score:
                best_score = v
                best_move = a
            alpha = max(alpha, v)
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
