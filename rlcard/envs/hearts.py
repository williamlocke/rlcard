import json
import os
import numpy as np

import rlcard
from rlcard.envs.env import Env
from rlcard.games.hearts.game import HeartsGame
from rlcard.games.hearts.game import HeartsMiniGame

ACTION_DUMP_SQ = 0
ACTION_DUMP_HIGHEST_HEART = 1
ACTION_DUMP_HIGHEST_DIAMOND = 2
ACTION_DUMP_HIGHEST_CLUB = 3
ACTION_DUMP_HIGHEST_SPADE = 4
ACTION_HIGHEST_LOSER = 5
ACTION_PLAY_LOWEST_CARD = 6
ACTION_RANDOM = 7
ACTION_PLAY_HIGHEST_CARD = 8
ACTION_HEURISTIC = 9


def get_rank(card):
    letter_ranks = {'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    if card[1] in letter_ranks.keys():
        return letter_ranks[card[1]]
    return int(card[1])


class HeartsEnv(Env):
    ''' Hearts Environment
    '''

    def __init__(self, allow_step_back=False):
        ''' Initialize the Hearts environment
        '''
        if not hasattr(self, 'game_class'):
            self.game_class = HeartsGame

        super().__init__(self.game_class(allow_step_back), allow_step_back)
        suits = []
        ranks = []
        for card in self.game.dealer.deck:
            suits.append(card.suit)
            ranks.append(card.rank)
        valid_suit = list(set(suits))
        valid_rank = list(set(ranks))
        # valid_suit = ['S', 'H', 'D', 'C']
        # valid_rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.actions = [x + y for x in valid_suit for y in valid_rank]

        deck_size = self.game.deck_size
        #self.state_shape = [deck_size * 3 + self.custom_features_length()]

        self.state_shape = [self.custom_features_length()]

        with open(os.path.join(rlcard.__path__[0], 'games/hearts/card2index.json'), 'r') as file:
            #self.card2index = json.load(file)
            card2index = {}
            for i, card in enumerate(self.game.dealer.deck):
                card2index[card.get_index()] = i
            self.card2index = card2index

    def get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def custom_features_length(self):
        return 13

    def current_winner(self, trick):
        target = None
        if len(trick) > 0:
            target = trick[0]
        current_winner = target
        for c in trick:
            if get_rank(c) > get_rank(current_winner) and c[0] == current_winner[0]:
                current_winner = c
        return current_winner


    def cards_left_by_suit(self):
        cards_left = {'H': [], 'D': [], 'C': [], 'S': []}
        for player in self.game.players:
            for c in player.hand:
                card = c.get_index()
                cards_left[card[0]].append(card)
        return cards_left

    def player_hand_by_suit(self, player_hand):
        hand_by_suit = {'H': [], 'D': [], 'C': [], 'S': []}
        for card in player_hand:
            hand_by_suit[card[0]].append(card)
        return hand_by_suit

    def winners_or_losers_left(self, suit, hand_by_suit, cards_left, winners=True):
        cards_left = cards_left[suit]
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        if winners == False:
            ranks.reverse()
        if len(hand_by_suit[suit]) == len(cards_left):
            return len(cards_left)
        winners_left = 0
        for rank in ranks:
            card = suit + rank
            if card in cards_left:
                if card in hand_by_suit[suit]:
                    winners_left += 1
                else:
                    break
        return winners_left


    def custom_features(self, state):
        trick = state['played_cards']
        target = None
        if len(trick) > 0:
            target = trick[0]
        player_hand = state['hand']

        if target is not None:
            has_target = any([c[0] == target[0] for c in player_hand])
        else:
            has_target = False
        has_QS = ('SQ' in player_hand)
        has_hearts = any([c[0] == 'H' for c in player_hand])
        features = [0] * self.custom_features_length()

        suits = {'H': 1, 'D': 2, 'C': 3, 'S': 4}
        cards_left = self.cards_left_by_suit()
        hand_by_suit = self.player_hand_by_suit(player_hand)

        current_winner = self.current_winner(trick)
        # CURRENT WINNER SUIT AND RANK
        # features[0] = 0 if current_winner is None else suits[current_winner[0]]
        # features[1] = 0 if current_winner is None else get_rank(current_winner)

        features[2] = int(has_QS)
        features[3] = int(has_target)
        features[4] = int(has_hearts)


        #
        # for i, suit in enumerate(suits.keys()):
        #     features[5 + i] = self.winners_or_losers_left(suit, hand_by_suit, cards_left)
        #     features[9 + i] = self.winners_or_losers_left(suit, hand_by_suit, cards_left, winners=False)

        return features

    def get_legal_action_ids(self, state):
        action_ids = []
        legal_actions = state['readable_legal_actions']
        trick = state['played_cards']
        player_hand = state['hand']
        target = None
        if len(trick) > 0:
            target = trick[0]
        # if target is not None:
        #     has_target = any([c[0] == target[0] for c in player_hand])
        # else:
        #     has_target = False
        #
        # current_winner = target
        # for c in trick:
        #     if get_rank(c) > get_rank(current_winner) and c[0] == current_winner[0]:
        #         current_winner = c
        #
        # can_dump_sq = 'SQ' in legal_actions and (not target == 'S') and target is not None
        #
        # if can_dump_sq:
        #     action_ids.append(ACTION_DUMP_SQ)
        #
        # action_dict = {'H': ACTION_DUMP_HIGHEST_HEART, 'D': ACTION_DUMP_HIGHEST_DIAMOND,
        #                'C': ACTION_DUMP_HIGHEST_CLUB, 'S': ACTION_DUMP_HIGHEST_SPADE}
        #
        # if target is not None:
        #     for suit in ['H', 'D', 'C', 'S']:
        #         if any([c[0] == suit for c in legal_actions]) and (not target == suit):
        #             action_ids.append(action_dict[suit])
        #
        # if has_target:
        #     if any([get_rank(c) < get_rank(current_winner) for c in legal_actions]):
        #         action_ids.append(ACTION_HIGHEST_LOSER)
        #

        action_dict = {'H': ACTION_DUMP_HIGHEST_HEART, 'D': ACTION_DUMP_HIGHEST_DIAMOND,
                       'C': ACTION_DUMP_HIGHEST_CLUB, 'S': ACTION_DUMP_HIGHEST_SPADE}
        if target is not None:
            for suit in ['H', 'D', 'C', 'S']:
                if any([c[0] == suit for c in legal_actions]) and (not target == suit):
                    action_ids.append(action_dict[suit])
        if 'SQ' in legal_actions:
            action_ids.append(ACTION_DUMP_SQ)

        action_ids.append(ACTION_PLAY_LOWEST_CARD)
        action_ids.append(ACTION_PLAY_HIGHEST_CARD)
        #action_ids.append(ACTION_RANDOM)

        action_ids.append(ACTION_HEURISTIC)

        return action_ids


    def extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        # TODO for project group: Improve this
        processed_state = {}

        processed_state['readable_legal_actions'] = state['legal_actions']

        for key, value in state.items():
            if key == 'legal_actions':
                # legal_actions = [self.actions.index(a) for a in state['legal_actions']]
                # processed_state['legal_actions'] = legal_actions
                processed_state['legal_actions'] = self.get_legal_action_ids(processed_state)
            else:
                processed_state[key] = value # Current hand (list of IDs), played cards (list of IDs), target suit (String)


        obs = np.zeros(self.state_shape)

        # # add card in players own hand to the observed state
        # idx = [self.card2index[card] for card in state['hand']]
        # obs[idx] = 1
        #
        # deck_size = self.game.__class__.get_action_num()
        #
        # # add cards played in round to the observed state.
        # # TODO: create separate representation for history of played cards and
        # #  cards played in the current round thus far
        # current_winner = self.current_winner(state['played_cards'])
        # idx = [self.card2index[card] + deck_size for card in [current_winner]]
        # obs[idx] = 1
        #
        # # add cards played in game to the observed state.
        # idx = [self.card2index[card] + deck_size * 2 for card in state['collected'] + state['played_cards']]
        # obs[idx] = 1

        feats = self.custom_features(state)
        obs[self.state_shape[0] - self.custom_features_length():] = feats
        processed_state['obs'] = obs

        return processed_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def heuristic_action_card(self):
        state = self.game.get_state(self.game.round.current_player)
        readable_legal_actions = state['legal_actions']
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13,
                  'A': 14}
        is_last = (len(state['played_cards']) == 3)
        is_first = (len(state['played_cards']) == 0)
        has_target = (readable_legal_actions[0][0] == state['target'])
        has_QS = ('SQ' in readable_legal_actions)
        smallest_card_idx = np.argmin([values[card[1]] for card in readable_legal_actions])
        largest_card_idx = np.argmax([values[card[1]] for card in readable_legal_actions])
        largest_heart_idx = np.argmax(
            [values[card[1]] if card[0] == 'H' else 0 for card in readable_legal_actions])
        if is_first:
            return readable_legal_actions[smallest_card_idx]
        else:
            if not has_target:
                if has_QS:
                    return 'SQ'
                if not largest_heart_idx == 0 or readable_legal_actions[0][0] == 'H':
                    return readable_legal_actions[largest_heart_idx]
                else:
                    return readable_legal_actions[largest_card_idx]
            elif is_last:
                return readable_legal_actions[largest_card_idx]
        return readable_legal_actions[smallest_card_idx]


    def decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        if action_id == ACTION_HEURISTIC:
            return self.heuristic_action_card()

        legal_actions = self.game.get_legal_actions()

        state = self.game.get_state(self.game.round.current_player)
        trick = state['played_cards']
        target = None
        if len(trick) > 0:
            target = trick[0]

        current_winner = target
        for c in trick:
            if get_rank(c) > get_rank(current_winner) and c[0] == current_winner[0]:
                current_winner = c


        if action_id == ACTION_DUMP_SQ:
            return "SQ"

        action_dict = {ACTION_DUMP_HIGHEST_HEART: 'H', ACTION_DUMP_HIGHEST_DIAMOND: 'D',
                       ACTION_DUMP_HIGHEST_CLUB: 'C', ACTION_DUMP_HIGHEST_SPADE: 'S'}
        if action_id in [ACTION_DUMP_HIGHEST_HEART, ACTION_DUMP_HIGHEST_DIAMOND,
                        ACTION_DUMP_HIGHEST_CLUB, ACTION_DUMP_HIGHEST_SPADE]:
            suit = action_dict[action_id]
            max_card = None
            max_rank = 0
            for c in legal_actions:
                if c[0] == suit and get_rank(c) > max_rank:
                    max_card = c
            return max_card

        if action_id == ACTION_HIGHEST_LOSER:
            card = None
            for c in legal_actions:
                if get_rank(c) < get_rank(current_winner):
                    if card is None or get_rank(c) > get_rank(card):
                        card = c
            return card

        if action_id == ACTION_PLAY_LOWEST_CARD:
            card = legal_actions[0]
            for c in legal_actions[1:]:
                if get_rank(c) < get_rank(card):
                    card = c
            return card
        if action_id == ACTION_PLAY_HIGHEST_CARD:
            card = legal_actions[0]
            for c in legal_actions[1:]:
                if get_rank(c) > get_rank(card):
                    card = c
            return card

        if not action_id == ACTION_RANDOM:
            print("WARNING: action not found: %d" % action_id)

        return np.random.choice(legal_actions)


class HeartsMiniEnv(HeartsEnv):
    def __init__(self, allow_step_back=False):
        self.game_class = HeartsMiniGame
        super(HeartsMiniEnv, self).__init__(allow_step_back)
