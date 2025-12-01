import pyspacemouse
import time
import asyncio

class State:
    def __init__(self, device_name=None):
        self.state = None
        self.device_name = device_name
    
    def is_zero(self):
        if self.state:
            x = self.state.x == 0
            y = self.state.y == 0
            z = self.state.z == 0
            roll = self.state.roll == 0
            pitch = self.state.pitch == 0
            yaw = self.state.yaw == 0
            buttons = all([b==0 for b in self.state.buttons])
            return all([x, y, z, roll, pitch, yaw, buttons])
        else:
            return 1

    def pose(self):
        return [self.state.x, self.state.y, self.state.z, self.state.roll, self.state.pitch, self.state.yaw]

    def button(self):
        return self.state.buttons

async def read_state(device, state):
    device = pyspacemouse.open(device=device)
    if device:
        while 1:
            state.state = device.read()
            await asyncio.sleep(0.005)

class SpaceMouseReader:
    def __init__(self):
        self.devices = pyspacemouse.list_devices()
        self.devices = list(set(self.devices)) # remove duplicate foundings
        self.devices.sort(reverse=1)
        print("-"*12, "Found Connected Devices:", self.devices, "-"*12)
        
        self.states = [State(device_name=d) for d in self.devices]
        
    async def start_reading(self, freq=1, callback=None):
        print("-"*12, "Sequentially Initialize Target Device.", "-"*12)
        tasks = []
        for i, device in enumerate(self.devices):
            tasks.append(asyncio.create_task(read_state(device, self.states[i])))

        while 1:
            # state callback, trigger callback to process the current state
            if callback:
                for i, s in enumerate(self.states):
                    callback(s, i)
            await asyncio.sleep(1/freq)
        # asyncio.gather(tasks)
        
    def read(self, freq=1, callback=lambda x,y: print(x.state)):
        asyncio.run(self.start_reading(freq=freq, callback=callback))

    
if __name__ == "__main__":
    reader = SpaceMouseReader()
    reader.read()