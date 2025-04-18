@ti.data_oriented
class Psystem_2D:

	def __init__(self, n, m):

		self.nparticles = n
		self.mass = m

		self.pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.nparticles))
		self.forces = ti.Vector.field(2, dtype=ti.f32, shape=(self.nparticles))

	@ti.kernel
	def reinit_forces(self):
		for k in range(self.nparticles):
			self.forces[k] = [0.0 for _ in range(2)]

# - We can define the basics of a particle system.
# - This is an independan model regarding its dynamics and forcesystem, other objects can be added after
