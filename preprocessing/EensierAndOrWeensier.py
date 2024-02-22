import numpy as np
import cv2
from gym.spaces import Box
from gym import ObservationWrapper

class MakeEensierAndOrWeensier(ObservationWrapper):
    def __init__(self, env, cut_in_half=False, scale=1):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:3]
        #print(obs_shape)
        self.scale=scale
        self.observation_space = Box(
                low=0, high=255, shape=(int(obs_shape[0]*scale), int(scale*obs_shape[1]//(1+cut_in_half)), obs_shape[2]), dtype=np.uint8
            )
        #print(self.observation_space.shape)
        self.cutoff_point = obs_shape[1]//(1+cut_in_half)

    def observation(self, obs):
        
        return cv2.resize((obs[:,:self.cutoff_point,:]), (0, 0), fx = self.scale, fy = self.scale, interpolation=cv2.INTER_AREA)[:,:,None]