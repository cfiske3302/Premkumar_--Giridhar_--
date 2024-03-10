from gym import ActionWrapper
from gym.spaces import Discrete
class ReduceActions(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        ### Action Space Mappings: ###
        # This version only accepts ZSJLT blocks #
        # By Possible Rotations #
        # 4L 3R # No Rotate #
        # 4L 3R # 2 cw #

        # 5L 3R # 1 cw #
        # 4L 4R # 1 ccw #

        # Action space of 34 is partitioned into two unequal halves #
        self.ret_string = '0'
        self.action_space = Discrete(34)

    def shift(self, center, code):
        # recenter to max number of left shifts
        code -= center

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
            case 1: # vertical flip
                self.ret_string += '0303'
            case 2: # 1 cw
                self.ret_string += '03'
            case 3: # 1 ccw
                self.ret_string += '04'
            case _:
                raise NotImplementedError('Rotate code fail')
        return
    
    def action(self, act):
        # reset the return string
        self.ret_string = '0'

        # First half, no rotate and vertical flip #
        if act < 16:
            rotate_code = act // 8
            shift_code = act % 8

            self.rotate(rotate_code)
            self.shift(4, shift_code)
        elif 16 <= act < 34:
            new_act = act - 16
            rotate_code = new_act // 9 + 2
            shift_code = new_act % 9

            self.rotate(rotate_code)
            self.shift(5 if rotate_code==2 else 4, shift_code)
        else:
            raise NotImplementedError('Out of action space bounds')

        # append the force drop to the sequence
        self.ret_string += '02'

        return self.ret_string