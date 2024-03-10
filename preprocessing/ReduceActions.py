from gym import ActionWrapper
from gym.spaces import Discrete
class ReduceActions(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        ### Action Space Mappings: ###
        # For shifting left and right #
        # using clockwise rotation (3) #
        # 6 left shifts at most #
        # 4 right shifts at most #
        self.ret_string = '0'
        self.action_space = Discrete(44)

    def shift(self, code):
        # recenter the code to the center of the piece
        code -= 6

        # append to the action sequence
        for i in range(abs(code)):
            if code > 0:
                self.ret_string += '5'
            elif code < 0:
                self.ret_string += '6'
            else:
                pass
        return
    
    def rotate(self, code):
        match code:
            case 0: # keep current rotation
                pass
            case 1: # 1 cw
                self.ret_string += '03'
            case 2: # 2 cw
                self.ret_string += '0303'
            case 3: # 1 ccw
                self.ret_string += '04'
            case _:
                raise NotImplementedError('Out of action space bounds')
        return
    
    def action(self, act):
        # reset the return string
        self.ret_string = '0'

        # rotation code is the 11s place
        rotate_code = act // 11
        # shift code is the 1s place
        shift_code = act % 11

        self.rotate(rotate_code)
        self.shift(shift_code)

        # append the force drop to the sequence
        self.ret_string += '02'

        # print(self.ret_string)
        return self.ret_string
