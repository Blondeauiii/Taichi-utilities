# - Here the pre-allocated version using shift

@ti.data_oriented
class Cortex2D:

	def __init__(self, nfil, lfil, l0, m):

		self.lfil = lfil                         	# Length of initial filaments
		self.nfil = nfil							# Number of initial filaments
		self.l0 = l0								# Length of one segment
		self.nparticles = self.nfil*self.lfil		# Number of polymerised particles
		self.nmax = nmax							# Number of available particles
		self.mass = m								# Mass of the one particle

		self.pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.nmax))
		self.forces = ti.Vector.field(2, dtype=ti.f32, shape=(self.nmax))

		self.nseg = self.nfil * (self.lfil-1)

		#self.lenreg = ti.field(dtype=ti.i32, shape=(self.nfil))		# supposed to evolve slowly
		self.len_start = ti.field(dtype=ti.i32, shape=(self.nfil))
		self.len_stop = ti.field(dtype=ti.i32, shape=(self.nfil))
		# - Lenreg is cumulative like for 10*10 we have [0, 10, 20, 30...] to have a fast access

		self.link0 = ti.field(dtype=ti.i32, shape=(self.nmax))		# to the max diectly
		self.link1 = ti.field(dtype=ti.i32, shape=(self.nmax))		# to the max diectly

		# - Relative to the dynamics of polymerisation

		self.polym = 0.0		# - polymerisation rate
		self.unpolym = 0.0		# - depolymerisation rate
		self.create = 0.0		# - emergence rate 

		self.shift = ti.Vector.field(2, dtype=ti.f32, shape=(self.nmax)) 	# - This is the tool of copy to permite the shifting
		self.lenshift = ti.field(dtype=ti.i32, shape=(2*self.nfil))			# - Same thing for the register

		#for i in range(1,self.nfil): self.lenreg[k] = self.lenreg[k-1] + self.lfil
	
	@ti.kernel
	def reinit_forces(self):
		for k in range(self.nparticles):
			self.forces[k] = [0.0 for _ in range(2)]

	@ti.kernel
	def polym(self):

		self.lenshift.fill(0)

		for k in range(self.nfil):

			if ti.random() < self.polym: 	# Polymerisation at the starting point of the filament
				
				self.lenshift[2*k] += 1
				self.nparticles += 1
				self.nseg += 1

			if ti.random() < self.polym: 	# Polymerisation at the stopping point of the filament
				
				self.lenshift[2*k+1] += 1
				self.nparticles += 1
				self.nseg += 1

			if ti.random() < self.unpolym: 	# Depolymerisation at the starting point of the filament

				self.lenshift[2*k] -= 1
				self.nparticles -= 1
				self.nseg += 1

			if ti.random() < self.unpolym: 	# Depolymerisation at the stopping point of the filament

				self.lenshift[2*k+1] -= 1
				self.nparticles -= 1
				self.nseg += 1

		if ti.random() < self.create: 		# Polymerisation of a new filament
			
			self.lenshift.append(0)
			self.lenshift.append(0)
			self.nfil += 1
			self.nparticles += 1


	@ti.kernel
	def shift(self):

		for k in range(self.nfil): 			# For each filament independantly

			shift = 0						# For the total shifting

			for i in range(2*k):          	# Do the sum of the shifting

				shift += self.lenshift[i] 

			shift += self.lenshift[2*k]  	# Take account of the shift of the beginning of the filament

			for i in range(self.len_start[k], self.len_stop[k]): 

				self.shift[k+shift] = self.pos[i] 		# Pickup the filament and apply the shift

			self.len_start[k] = self.len_start[k] + shift - self.lenshift[2*k]		# Adaptation of the new len_start
			self.len_stop[k] = self.len_stop[k] + shift + self.lenshift[2*k+1]		# Adaptation of the new len_stop

			if(lenshift[2*k]==1): self.shift[ self.len_start[k] - 1 ] = self.shift[ self.len_start[k] ] # If adding, copy the first particle
			if(lenshift[2*k+1]==1): self.shift[ self.len_stop[k] + 1 ] = self.shift[ self.len_stop[k] ] # If adding, copy the last particle

		for k in range(self.nparticles)

			self.pos[k] = self.shift[k]
