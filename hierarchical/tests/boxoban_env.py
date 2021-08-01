from gym_sokoban.envs.sokoban_env import SokobanEnv
from gym_sokoban.envs.render_utils import room_to_rgb
import numpy as np

class BoxobanEnv(SokobanEnv):
    num_boxes = 4
    dim_room=(10, 10)

    def __init__(self,
             max_steps=120,
             difficulty='unfiltered', split='train'):
        self.difficulty = difficulty
        self.split = split
        self.verbose = False
        super(BoxobanEnv, self).__init__(self.dim_room, max_steps, self.num_boxes, None, False)
        

    def reset(self, room, topology):
        room[room == 3] = 4
        self.player_position = np.argwhere(room == 5)[0]
        self.room_fixed, self.room_state, self.box_mapping = topology, room, {}

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = room_to_rgb(self.room_state, self.room_fixed)

        return starting_observation