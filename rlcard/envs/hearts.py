import json
import os
import numpy as np

import rlcard
from rlcard.envs.env import Env
from rlcard.games.hearts.game import HeartsGame
from rlcard.games.hearts.game import HeartsMiniGame

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
        self.state_shape = [deck_size * 3 + self.custom_features_length()]

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
        legal_actions = self.game.get_legal_actions()
        legal_actions += [len(self.actions)]
        legal_actions = list(set(legal_actions))
        return legal_actions

        #return self.game.get_legal_actions()


    def custom_features_length(self):
        return 3

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

        if not has_target and not target is None:
            if has_QS and len(player_hand) > 1:
                features[0] = 1
            if has_hearts and len(player_hand) > 1 and not all([c[0] == 'H' for c in player_hand]) and \
                        any([c[0] == 'H' and c[1] in ['A', 'K', 'Q', 'J'] for c in player_hand]):
                features[1] = 1
            if not has_hearts and len(player_hand) > 1 and any([c[1] in ['A', 'K', 'Q', 'J'] for c in player_hand]):
                features[2] = 1
        return features


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

        for key,value in state.items():
            if key == 'legal_actions':
                legal_actions = [self.actions.index(a) for a in state['legal_actions']]
                legal_actions += [len(self.actions)]
                processed_state['legal_actions'] = legal_actions
            else:
                processed_state[key] = value # Current hand (list of IDs), played cards (list of IDs), target suit (String)

        obs = np.zeros(self.state_shape)

        # add card in players own hand to the observed state
        idx = [self.card2index[card] for card in state['hand']]
        obs[idx] = 1


        deck_size = self.game.__class__.get_action_num()

        # add cards played in round to the observed state.
        # TODO: create separate representation for history of played cards and
        #  cards played in the current round thus far
        idx = [self.card2index[card] + deck_size for card in state['played_cards']]
        obs[idx] = 1

        # add cards played in game to the observed state.
        idx = [self.card2index[card] + deck_size * 2 for card in state['collected']]
        obs[idx] = 1

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
        if action_id >= len(self.actions):
            custom_action_id = len(self.actions) - action_id
            if custom_action_id == 0:
                return self.heuristic_action_card()

        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            raise "Selected action is illegal!"
        return self.actions[action_id]


class HeartsMiniEnv(HeartsEnv):
    def __init__(self, allow_step_back=False):
        self.game_class = HeartsMiniGame
        super(HeartsMiniEnv, self).__init__(allow_step_back)

