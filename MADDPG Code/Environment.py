from dm_control.locomotion import soccer

class Environment:
    def __init__(self):
        self.env = soccer.load(team_size=1,
                      time_limit=10.0,
                      disable_walker_contacts=False,
                      enable_field_box=True,
                      terminate_on_goal=False,
                      walker_type=soccer.WalkerType.BOXHEAD)

    def reset(self):
        return self.env.reset()