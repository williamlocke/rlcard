from rlcard.games.hearts.player import HeartsPlayer as Player

def get_first_player(players):
    for idx in range(len(players)):
        player = players[idx]
        if player.has_2_of_spades():
            return idx
    raise 'There is a problem with the dealer - somebody did not get the 2 of spades'
