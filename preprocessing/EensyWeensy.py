import numpy as np
import cv2
from gym.spaces import Box
from gym import ObservationWrapper

class MakeEensyWeensy(ObservationWrapper):
    def __init__(self, env, cut_in_half=False, scale=1):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:3]

        if cut_in_half:
            self.x1 = 109
            self.x2 = 290
            self.y1 = 136
            self.y2 = 498
        else:
            self.x1 = 0
            self.x2 = obs_shape[1]
            self.y1 = 0
            self.y2 = obs_shape[2]
        
        #print(obs_shape)
        self.scale=scale
        self.observation_space = Box(
                low=0, high=255, shape=(round((self.y2-self.y1)*scale), round(scale*(self.x2-self.x1)), obs_shape[2]), dtype=np.uint8
            )
        #print(self.observation_space.shape)

    def observation(self, obs):
        return cv2.resize((obs[self.y1:self.y2,self.x1:self.x2,:]), (0, 0), fx = self.scale, fy = self.scale, interpolation=cv2.INTER_AREA)[:,:,None]