from pente import Pente
import numpy as np
import time
import pente
from keras import models
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape, Input, Conv2DTranspose
from keras.optimizers.legacy import Adam

# Write augment function that takes in a game and a 19x19 matrix of probabilities
# Rotate the board and probabilities 3 times and add to the list
# Also multiply each board and rotated board and prob and rotated prob by -1 and add to the list
    
def augment(game, probs):

    probs = probs.squeeze(axis=0)

    games = []
    prob_list = []

    board = np.array(game.get_board())

    for i in range(4):

        games.append(np.expand_dims(np.rot90(board, i), axis=0))
        prob_list.append(np.expand_dims(np.rot90(probs, i), axis=0))

        games.append(np.expand_dims(np.rot90(-1*board, i), axis=0))
        prob_list.append(np.expand_dims(np.rot90(-1*probs, i), axis=0))

    return games, prob_list

# Define a neural network that takes a 19x19 board as input and outputs a 19x19 matrix of values
# Each value is the probability of playing on that space

input = Input(shape=(19, 19, 1))
x = Conv2D(256, (5, 5), activation='relu')(input)
x = Conv2D(128, (5, 5), activation='relu')(x)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
# x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
# x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
# x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
x = Conv2DTranspose(256, (5, 5), activation='relu')(x)
x = Conv2DTranspose(1, (5, 5))(x)
x = Reshape((19, 19, 1))(x)

nn = models.Model(inputs=input, outputs=x)

nn.summary()

nn.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mae'])

NUM_BATCHES = 10000
NUM_GAMES = 10000

# states = []

for batch in range(NUM_BATCHES):
    start = time.time()
    batch_games = []
    batch_probs = []
    for i in range(NUM_GAMES):
        done = False
        game = Pente()

        states = game.play_random_game()

        current_game = states[-2][0]
        current_player = current_game.current_player

        values = pente.get_values(game=current_game, rollouts_per_action=100)
        # print(np.array(states[min([100, len(states) - 3])][0].get_board()).sum())
        values = np.array(values)

        probs = 100*((values) + 0.001 * np.random.uniform(low=-1, high=1, size=values.shape))

        probs = np.expand_dims(probs, axis=0)
        if i % 10 == 0:
            print(f"Batch {batch} Game {i}")
        # print(probs)
        # print(len(states))
            
        curr_games, curr_probs = augment(current_game, probs)

        batch_games += curr_games
        batch_probs += curr_probs

        # print(len(batch_games))



    batch_games = np.vstack(batch_games)
    batch_probs = np.vstack(batch_probs)

    # fit network on single state
    nn.fit(batch_games, batch_probs, epochs=100, batch_size=64, shuffle=True, verbose=1)

    end = time.time()

    print(f"Time per game: {(end - start) / NUM_GAMES} seconds")
    print(f"Total time: {end - start} seconds")

