import numpy as np

class BoxobanEnvironment:
    """
    wall = 0
    empty = 1
    target = 2
    box = 3
    box on target = 4
    player = 5
    player on target = 6
    """
    """
    push up = 0
    push down = 1
    push left = 2
    push right = 3
    move up = 4   
    move down = 5
    move left = 6
    move right = 7
    """

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    height = 10
    width = 10
    total_boxes = 4

    room = None
    topology = None
    player_location = None
    boxes_on_target = 0

    steps = 0
    max_steps = 128

    reward_per_step = -0.01
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

        return self.observation,reward,done,info
    
    def push(self, action):
        change = self.moves[action % 4]

        player_location = self.player_location.copy()
        next_player_location = player_location + change
        next_box_location = next_player_location + change

        if next_box_location[0] >= self.height or next_box_location[1] >= self.width:
            return 0

        next_player_location_state = self.room[next_player_location[0], next_player_location[1]]
        next_box_location_state = self.room[next_box_location[0], next_box_location[1]]

        if not (
            next_player_location_state in [3, 4] and
            next_box_location_state in [1, 2]
        ):
            return self.move(action)

        reward = 0

        #Move Player
        self.player_location = next_player_location
        self.room[next_player_location[0], next_player_location[1]] = 6 if next_player_location_state == 4 else 5
        self.room[player_location[0], player_location[1]] = self.topology[player_location[0], player_location[1]]

        if next_player_location_state == 4:
            self.boxes_on_target -= 1
            reward += self.reward_box_off_target

        #Move Box
        if next_box_location_state == 2:
            self.room[next_box_location[0],next_box_location[1]] = 4
            self.boxes_on_target += 1
            reward += self.reward_box_on_target
        else:
            self.room[next_box_location[0], next_box_location[1]] = 3
        
        return reward

    def move(self, action):
        change = self.moves[action% 4]

        player_location = self.player_location.copy()
        next_player_location = player_location + change
        
        next_player_location_state = self.room[next_player_location[0], next_player_location[1]] 
        if not (next_player_location_state in [1, 2]):
            return 0
        
        self.player_location = next_player_location
        self.room[next_player_location[0], next_player_location[1]] = 6 if next_player_location_state == 2 else 5
        self.room[player_location[0], player_location[1]] = self.topology[player_location[0], player_location[1]]

        return 0

    def set_max_steps(self, steps):
        self.max_steps = steps

    @property
    def observation(self):
        wall = np.zeros((10, 10), dtype=np.float32)
        empty = np.zeros((10, 10), dtype=np.float32)
        target = np.zeros((10, 10), dtype=np.float32)
        box = np.zeros((10, 10), dtype=np.float32)
        box_on_target = np.zeros((10, 10), dtype=np.float32)
        player = np.zeros((10, 10), dtype=np.float32)
        player_on_target = np.zeros((10, 10), dtype=np.float32)

        for i, row in enumerate(self.room):
            for j, state in enumerate(row):
                if state == 0:
                    wall[i, j] = 1
                elif state == 1:
                    empty[i, j] = 1
                elif state == 2:
                    target[i, j] = 1
                elif state == 3:
                    box[i, j] = 1
                elif state == 4:
                    box_on_target[i, j] = 1
                elif state == 5:
                    player[i, j] = 1
                elif state == 6:
                    player_on_target[i, j] = 1
        
        return np.stack((
            wall,
            empty,
            target,
            box,
            box_on_target,
            player,
            player_on_target,
        ), axis=0)
