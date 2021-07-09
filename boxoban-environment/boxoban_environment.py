import numpy as np

class BoxobanEnvironment:
    height = 10
    width = 10
    total_boxes = 4
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    room = None
    topology = None
    player_location = None
    boxes_on_target = 0

    steps = 0
    max_steps = 300

    reward_per_step = -0.1
    reward_box_on_target = 1
    reward_box_off_target = -1
    reward_finished = 10

    def __init__(self, room, topology):
        self.room = room
        self.topology = topology
        self.player_location = np.argwhere(room == 5)[0]

    def step(self, action):
        self.steps += 1
        reward = self.reward_per_step

        if action <= 3:
            reward += self.push(action)
        else:
            reward += self.move(action)

        finished = (self.boxes_on_target == self.total_boxes)
        max_steps = (self.steps >= self.max_steps)
        done = finished or max_steps

        if finished == True:
            reward += self.reward_finished

        info = {
            "finished": finished,
            "max_steps": max_steps
        }

        return self.room,reward,done,info
    
    def push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = self.moves[action % 4]

        player_location = self.player_location.copy()
        next_player_location = player_location + change
        next_box_location = next_player_location + change

        if next_box_location[0] >= self.height or next_box_location[1] >= self.width:
            return

        next_player_location_state = self.room[next_player_location[0], next_player_location[1]]
        next_box_location_state = self.room[next_box_location[0], next_box_location[1]]

        if not (
            next_player_location_state in [3, 4] and
            next_box_location_state in [1, 2]
        ):
            return self.move(action)

        reward = 0

        # Move Player
        self.player_location = next_player_location
        self.room[next_player_location[0], next_player_location[1]] = 6 if next_player_location_state == 4 else 5
        self.room[player_location[0], player_location[1]] = self.topology[player_location[0], player_location[1]]

        if next_player_location_state == 4:
            self.boxes_on_target -= 1
            reward += self.reward_box_off_target

        # Move Box
        if next_box_location_state == 2:
            self.room[next_box_location[0],next_box_location[1]] = 4
            self.boxes_on_target += 1
            reward += self.boxes_on_target
        else:
            self.room[next_box_location[0], next_box_location[1]] = 3
        
        return reward

    def move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = self.moves[action% 4]

        player_location = self.player_location.copy()
        next_player_location = player_location + change
        
        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        next_player_location_state = self.room[next_player_location[0], next_player_location[1]] 
        if not (next_player_location_state in [1, 2]):
            return 0
        
        self.player_location = next_player_location
        self.room[next_player_location[0], next_player_location[1]] = 6 if next_player_location_state == 2 else 5
        self.room[player_location[0], player_location[1]] = self.topology[player_location[0], player_location[1]]

        return 0

    def set_max_steps(self, steps):
        self.max_steps = steps

