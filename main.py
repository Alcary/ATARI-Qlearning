import itertools
import numpy as np
import sys
from collections import defaultdict

# Supported Games: https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/docs/games.md
from ale_py.roms import Phoenix
from ale_py import ALEInterface, SDL_SUPPORT

ale = ALEInterface()

print(SDL_SUPPORT)
np.set_printoptions(threshold=sys.maxsize, linewidth=1000)

# Get & Set the desired settings
ale.setInt("random_seed", 123)
# The default is already 0.25, this is just an example
ale.setFloat("repeat_action_probability", 0.25)

# Check if we can display the screen
# For the first set of training better let it without UI/sound
if SDL_SUPPORT:
    ale.setBool("sound", True)
    ale.setBool("display_screen", True)

# Load our game
ale.loadROM(Phoenix)

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
	"""
	Creates an epsilon-greedy policy based
	on a given Q-function and epsilon.
	
	Returns a function that takes the state
	as an input and returns the probabilities
	for each action in the form of a numpy array
	of length of the action space(set of possible actions).
	"""
	def policyFunction(state):

		Action_probabilities = np.ones(num_actions,
				dtype = float) * epsilon / num_actions
				
		best_action = np.argmax(Q[state])
		Action_probabilities[best_action] += (1.0 - epsilon)
		return Action_probabilities

	return policyFunction

def fromArrayToHash(x):
    return hash(x.tostring())

# Function to get the player's coordinates
def get_player_position(s):
    m,n = s.shape
    player_x = None
    player_y = None
    for i in range(0,m):
        for j in range (0,n):
            if s[i][j] == 56 and s[i+4][j] == 56:
                player_x = i
                player_y = j
                return player_x,player_y
    return player_x,player_y


""" Function to get the closest enemy's coordinates
    based on the player's position
"""
def get_enemy_position(s,player_x,player_y):
    m,n = s.shape
    enemy_x = None
    enemy_y = None
    for i in range(player_x, m):
        for j in range(player_y, n):
            if s[i][j] == 100:
                enemy_x = i
                enemy_y = j
                return enemy_x, enemy_y
    for i in range(player_x, 0):
        for j in range(player_y, n):
            if s[i][j] == 100:
                enemy_x = i
                enemy_y = j
                return enemy_x, enemy_y
    return enemy_x,enemy_y    


def qLearning(env, num_episodes, discount_factor = 1.0,
							alpha = 0.6, epsilon = 0.1):

	"""
	Q-Learning algorithm: Off-policy TD control.
	Finds the optimal greedy policy while improving
	following an epsilon-greedy policy"""

	# Action value function
	# A nested dictionary that maps
	# state -> (action -> action-value).

	all_actions = env.getLegalActionSet()
	legal_actions = all_actions[0:1] + all_actions[3:4]
	num_actions = len(legal_actions)
	print(num_actions)
	Q = defaultdict(lambda: np.zeros(num_actions))	

	# Create an epsilon greedy policy function
	# appropriately for environment action space
	policy = createEpsilonGreedyPolicy(Q, epsilon, num_actions)
	
	# For every episode
	for _ in range(num_episodes):
		
		# Reset the environment and pick the first action
		env.reset_game()
		
		screen = env.getScreen()
		x, y = get_player_position(screen)
		if(x != None and y != None):
				e_x, e_y = get_enemy_position(screen, x, y)
				state = (x, y, e_x, e_y) 
		else: action = (0)		

		for t in itertools.count():
			
			# get probabilities of all actions from current state
			action_probabilities = policy(state)

			# choose action according to
			# the probability distribution
			action = np.random.choice(np.arange(
					len(action_probabilities)),
					p = action_probabilities)

			# take action and get reward, transit to next state
			reward = env.act(action)
			done = env.game_over()
			
			screen = env.getScreen()
			x, y = get_player_position(screen)
			if(x != None and y != None):
				e_x, e_y = get_enemy_position(screen, x, y)
				next_state = (x, y, e_x, e_y) 
			else: action = (0)
			
			# TD Update
			best_next_action = np.argmax(Q[next_state])	
			td_target = reward + discount_factor * Q[next_state][best_next_action]
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta
			
			# done is True if episode terminated
			if done:
				break
			state = next_state
			


	return Q

avail_modes = ale.getAvailableModes()
avail_diff = ale.getAvailableDifficulties()

print(f"Number of available modes: {len(avail_modes)}")
print(f"Number of available difficulties: {len(avail_diff)}")

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

ale.setDifficulty(avail_diff[0])
ale.setMode(avail_modes[0])
ale.reset_game()
state=qLearning(ale, 1000)