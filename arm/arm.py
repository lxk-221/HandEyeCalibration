# virtual class for arm that attach with camera or calibration board

class Arm:
    def __init__(self) -> None:
        pass
    
    def move2joints(self, joint):
        '''
        move arm to the given joint positions
        '''
        raise NotImplementedError

    @property
    def joints(self):
        '''
        return current joint positions
        '''
        raise NotImplementedError
    
    @property
    def T_ee2base(self):
        '''
        follow opencv formulation https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
        T_gripper2base means a homogeneous transform matrix that can transform a point from gripper frame to base frame
        p_base = T_gripper2base @ p_gripper
        '''
        raise NotImplementedError