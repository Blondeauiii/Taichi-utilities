# Methods for updating a field 
class DoubleBuffers:
    def __init__(self, resolution, n_channel):
        if n_channel == 1:
            self.current = ti.field(ti.f32, shape=resolution)
            self.next = ti.field(ti.f32, shape=resolution)
        else:
            self.current = ti.Vector.field(n_channel, ti.f32, shape=resolution)
            self.next = ti.Vector.field(n_channel, ti.f32, shape=resolution)

    def swap(self):
        self.current, self.next = self.next, self.current

    def reset(self):
        self.current.fill(0)
        self.next.fill(0)