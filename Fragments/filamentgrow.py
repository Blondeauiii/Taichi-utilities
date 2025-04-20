import taichi as ti

@ti.data_oriented
class Cortex2D_test:

	def __init__(self, nfil, lfil, nmax, l0, m):

		# - Init the python variables

		self.lfil = lfil                         			# Length of initial filaments
		self.l0 = l0										# Length of one segment
		self.nmax = nmax									# Number of available particles
		self.mass = m										# Mass of the one particle

		self.pseq = 1.0      								# - Probability of launching a sequence of polymerisation

		self.prate = 0.3									# - polymerisation rate
		self.unprate = 0.1									# - depolymerisation rate
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
		S4 = ti.root.dynamic(ti.i, self.nmax, chunk_size=nfil)

		self.len_start = ti.field(dtype=ti.i32)					# - supposed to evolve slowly
		self.len_stop = ti.field(dtype=ti.i32)					# - supposed to evolve slowly
		self.lenshift = ti.field(dtype=ti.i32)					# - Same thing for the register
		self.filshift = ti.field(dtype=ti.i32)					# - In case a filament disappear

		S1.place(self.len_start)
		S2.place(self.len_stop)
		S3.place(self.lenshift)
		S4.place(self.filshift)

		self.init_startstop()

	@ti.kernel
	def rdplace(self, spacedim: float, cx: float, cy: float):
		
		for k in range(self.nfil[None]):
			ti.loop_config(serialize=True)
			alpha = ti.random()*2*3.14
			pos = [ti.random()*spacedim+cx, ti.random()*spacedim+cy]
			for l in range(self.lfil):

				self.pos[k*self.lfil+l] = [ pos[0] + l*ti.math.cos(alpha) , pos[1] + l*ti.math.sin(alpha) ]

				if l > 0:
					self.link0[k*(self.lfil-1)+l-1] = k*self.lfil+l-1
					self.link1[k*(self.lfil-1)+l-1] = k*self.lfil+l

	@ti.kernel
	def reinit_forces(self):
		for k in range(self.nparticles[None]):
			self.forces[k] = [0.0 for _ in range(2)]

	def grow(self, t):

		if rd.random() < self.pseq :

			self.filshift.fill(0)
			self.lenshift.fill(0)
			self.shift.fill(0)

			self.rd_polym(t)
			ti.sync()

			self.shift_lists()
			ti.sync()

	@ti.kernel
	def init_startstop(self):

		ti.loop_config(serialize=True)
		for k in range(self.nfil[None]):
			
			self.len_start.append( k * self.lfil )
			self.len_stop.append( (k+1) * self.lfil -1 )
			self.filshift.append( 0 )

		for k in range(2*self.nfil[None]):

			self.lenshift.append( 0 )

	@ti.kernel
	def rd_polym(self, t: int):

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
				self.nseg[None] -= 1

			if ti.random() < self.unprate: 	# Depolymerisation at the stopping point of the filament

				self.lenshift[2*k+1] -= 1
				self.nparticles[None] -= 1
				self.nseg[None] -= 1

			# In the case the filament disappear
			if self.len_start[k] - self.len_stop[k] >= self.lenshift[2*k+1] + self.lenshift[2*k] :

				self.filshift[k] -= 1
				self.nseg[None] += 1 # We delete a single particle

				# just make sure to don't apply -2 on a 1 length filament, we repair the system
				if self.len_start[k] - self.len_stop[k] == 1 and self.lenshift[2*k+1] + self.lenshift[2*k] == -2: 

					self.lenshift[2*k] = -1
					self.lenshift[2*k+1] = 0
					self.nparticles[None] += 1
					self.nseg[None] -= 1

		if ti.random() < self.create: 		# Polymerisation of a new filament
	
			self.nfil[None] += 1
			self.nparticles[None] += 1

	@ti.kernel
	def shift_lists(self):

		new_nfil = self.nfil[None]

		for k in range(self.nfil[None]): 			# For each filament independantly
	
		# -------------------------------------------------------------------------------------------

			shift = 0									# For the total shifting
			fshift = 0 									# For the total filament shifting

			for i in range(2*k):          				# Do the sum of the shifting of the previous fils

				shift += self.lenshift[i] 

			for i in range(k):

				fshift += self.filshift[i]

			rstart = self.len_start[k]      					# Start for reading
			rstop = self.len_stop[k] + 1						# Stop for reading

			self.len_start[k] += shift                        							# Start for the writting
			self.len_stop[k] += shift + self.lenshift[2*k] + self.lenshift[2*k+1] 		# Stop for the writting

			# Take account of the shift of the beginning of the filament

			if self.lenshift[2*k] == -1 : rstart+=1  			# Reduce the range of reading
			if self.lenshift[2*k+1] == -1 : rstop-=1 			# Reduce the range of reading

			if rstop - rstart < 1: new_nfil -= 1

			# -----------------------------------------------

			for i in range(rstart, rstop): 

				self.shift[ i + shift + self.lenshift[2*k] ] = self.pos[i] 		# Pickup the filament and apply the shift

			ti.sync()

			# -----------------------------------------------
			if self.filshift[k] < 0:

				self.len_start[k + fshift] = self.len_start[k] + shift - self.lenshift[2*k]		# Adaptation of the new len_start
				self.len_stop[k + fshift] = self.len_stop[k] + shift + self.lenshift[2*k+1]		# Adaptation of the new len_stop

			if(self.lenshift[2*k]==1):  														# If adding, copy the first particle
				self.shift[ self.len_start[k] ] = self.shift[ self.len_start[k]+1  ]
				self.shift[ self.len_start[k] ] += 1 * ( self.shift[ self.len_start[k]+1 ] - self.shift[ self.len_start[k]+2 ] )
			if(self.lenshift[2*k+1]==1): 														# If adding, copy the last particle
				self.shift[ self.len_stop[k] ] = self.shift[ self.len_stop[k]-1  ] 
				self.shift[ self.len_stop[k] ] += 1 * ( self.shift[ self.len_stop[k]-1 ] - self.shift[ self.len_stop[k]-2 ] )

		# -------------------------------------------------------------------------------------------

		for k in range(self.nmax):

			if k < self.nparticles[None]:
				self.pos[k] = self.shift[k]
			else:
				self.pos[k] = 0
		
		for k in range(self.nfil[None]):

			start = self.len_start[k] #- k # Taking account of the number of segment, not the number of particles

			for i in range(start, self.len_stop[k]):

				self.link0[ i-k ] = i
				self.link1[ i-k ] = i + 1

		ti.sync()

		self.nfil[None] = new_nfil
