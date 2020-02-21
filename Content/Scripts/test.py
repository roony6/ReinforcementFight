import unreal_engine as ue
import numpy as np

ue.log('Hello i am a Python module')


class Hero:

    # this is called on game start
    def begin_play(self):
        ue.log('Begin Play on Hero class')
        d = np.random.rand()
        ue.print_string(d)

    # this is called at every 'tick'
    def tick(self, delta_time):
        # get current location
        rotation = self.uobject.get_actor_rotation()
        # increase Z honouring delta_time
        rotation.yaw += 100 * delta_time
        # set new location
        self.uobject.set_actor_rotation(rotation)

    # this new method will be available to blueprints
    def FunnyNewMethod(self, a_word: str):
        ue.print_string('This is a word from blueprint: ' + a_word)
        location = self.uobject.get_actor_location()
        return str(location)
    # FunnyNewMethod.event = True

    def goUp(self, steps: str):
        ue.print_string('This is a word from blueprint: ' + steps)
        # get current location
        location = self.uobject.get_actor_location()
        steps = int(steps)
        # increase Z honouring delta_time
        location.z += steps# * self.uobject.get_world_delta_seconds()
        # set new location
        self.uobject.set_actor_location(location)
        return f"Location {location} : Steps {steps} "
    # goUp.event = True
