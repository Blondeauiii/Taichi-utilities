# - Here the mid pre-allocated version using shift

@ti.data_oriented
class Cortex2D:

	def __init__(self, nfil, lfil, nmax, l0, m):

		# - Init the python variables

		self.lfil = lfil                         			# Length of initial filaments
		self.l0 = l0										# Length of one segment
		self.nmax = nmax									# Number of available particles
		self.mass = m										# Mass of the one particle

		self.pseq = 1.0      								# - Probability of launching a sequence of polymerisation

		self.prate = 0.0									# - polymerisation rate
		self.unprate = 0.0									# - depolymerisation rate
		self.create = 0.0									# - emergence rate 

		# - Init the taichi variables

		self.nfil = ti.field(dtype=ti.i32, shape=())		
		self.nfil[None] = nfil									# Number of initial filaments

		self.nparticles = ti.field(dtype=ti.i32, shape=())	
		self.nparticles[None] = self.nfil[None]*self.lfil		# Number of polymerised particles

		self.nseg = ti.field(dtype=ti.i32, shape=())			
		self.nseg[None] = self.nfil[None] * (self.lfil-1)		# Number of segments 

		# - Init the taichi fields

		self.pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.nmax))
		self.forces = ti.Vector.field(2, dtype=ti.f32, shape=(self.nmax))
	
		

		self.link0 = ti.field(dtype=ti.i32, shape=(self.nmax))		# to the max diectly
		self.link1 = ti.field(dtype=ti.i32, shape=(self.nmax))		# to the max diectly

		self.shift = ti.Vector.field(2, dtype=ti.f32, shape=(self.nmax)) 	# - This is the tool of copy to permite the shifting
		

		# - Allocation of dynamic fields

		S1 = ti.root.dynamic(ti.i, self.nmax, chunk_size=nfil)
		S2 = ti.root.dynamic(ti.i, self.nmax, chunk_size=nfil)
		S3 = ti.root.dynamic(ti.i, self.nmax, chunk_size=nfil)

		self.len_start = ti.field(dtype=ti.i32)					# - supposed to evolve slowly
		self.len_stop = ti.field(dtype=ti.i32)					# - supposed to evolve slowly
		self.lenshift = ti.field(dtype=ti.i32)					# - Same thing for the register

		S1.place(self.len_start)
		S2.place(self.len_stop)
		S3.place(self.lenshift)

		#S = ti.root.dynamic(ti.i, 1024, chunk_size=32)
		#x = ti.field(int)
		#S.place(x)


	def grow(self):

		if rd.random() < self.pseq :

			self.rd_polym()

			self.shift_lists()
	
	@ti.kernel
	def reinit_forces(self):
		for k in range(self.nparticles):
			self.forces[k] = [0.0 for _ in range(2)]

	@ti.kernel
	def rd_polym(self):

		self.lenshift.fill(0)
		self.shift.fill(0)

		for k in range(self.nfil[None]):

			if ti.random() < self.prate: 	# Polymerisation at the starting point of the filament
				
				self.lenshift[2*k] += 1
				self.nparticles[None] += 1
				self.nseg[None] += 1

			if ti.random() < self.prate: 	# Polymerisation at the stopping point of the filament
				
				self.lenshift[2*k+1] += 1
				self.nparticles[None] += 1
				self.nseg[None] += 1

			if ti.random() < self.unprate: 	# Depolymerisation at the starting point of the filament

				self.lenshift[2*k] -= 1
				self.nparticles[None] -= 1
				self.nseg[None] += 1

			if ti.random() < self.unprate: 	# Depolymerisation at the stopping point of the filament

				self.lenshift[2*k+1] -= 1
				self.nparticles[None] -= 1
				self.nseg[None] += 1

		if ti.random() < self.create: 		# Polymerisation of a new filament
	
			self.nfil[None] += 1
			self.nparticles[None] += 1

	@ti.kernel
	def shift_lists(self):

		for k in range(self.nfil[None]): 			# For each filament independantly
	
		# -------------------------------------------------------------------------------------------

			shift = 0						# For the total shifting

			for i in range(2*k):          	# Do the sum of the shifting

				shift += self.lenshift[i] 

			shift += self.lenshift[2*k]  	# Take account of the shift of the beginning of the filament

			ti.sync()

			# -----------------------------------------------

			for i in range(self.len_start[k], self.len_stop[k]): 

				self.shift[k+shift] = self.pos[i] 		# Pickup the filament and apply the shift

			# -----------------------------------------------

			self.len_start[k] = self.len_start[k] + shift - self.lenshift[2*k]		# Adaptation of the new len_start
			self.len_stop[k] = self.len_stop[k] + shift + self.lenshift[2*k+1]		# Adaptation of the new len_stop

			if(self.lenshift[2*k]==1): self.shift[ self.len_start[k] - 1 ] = self.shift[ self.len_start[k] ] # If adding, copy the first particle
			if(self.lenshift[2*k+1]==1): self.shift[ self.len_stop[k] + 1 ] = self.shift[ self.len_stop[k] ] # If adding, copy the last particle

			ti.sync()

		ti.sync()

		# -------------------------------------------------------------------------------------------

		for k in range(self.nparticles[None]):

			self.pos[k] = self.shift[k]

		ti.sync()

		for k in range(self.nfil[None]):

			start = self.len_start[k] - k # Taking account of the number of segment, not the number of particles

			for i in range(self.len_start[k], self.len_stop[k]-1):

				self.link0[ self.len_start[k]-k ] = i
				self.link1[ self.len_start[k]-k ] = i + 1

		ti.sync()
