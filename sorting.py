@ti.data_oriented
class Cell_reg:

    def __init__(self, parts, dim, spacedim):  # Necessary for cell listing

        self.particlesystem = parts
        self.nparticles = parts.nparticles

        self.padded_size = 1 << (self.nparticles - 1).bit_length()  # Prochaine puissance de 2
        self.logsize = np.log2(self.padded_size)

        self.idx = ti.field( shape=self.nparticles, dtype=ti.i32)
        self.hashlist = ti.field( shape=self.nparticles, dtype=ti.i32)
        self.select_neighbors = ti.field( shape=self.nparticles, dtype=ti.i32)
        self.start_idx = ti.field( shape=(dim+2)**3, dtype=ti.i32)

        self.n2idx = ti.field(dtype=ti.f32, shape=self.padded_size)
        self.n2hash = ti.field(dtype=ti.f32, shape=self.padded_size)

        self.dim = dim + 2  # particles should not be in the extremities
        self.max_hash = self.dim**3 + 1
        self.spacedim = spacedim  
        self.redcell = ( dim - 2 ) / ( self.spacedim  * 2) # use of dim to keep particles inside

        for k in range(self.idx.shape[0]): self.idx[k] = k


            # Necessary for Mesh-Particle interaction

    @ti.kernel
    def bitonic_sort_hash(self, up: ti.i32, Nl: ti.i32):
        k = 2
        while k <= Nl:
            j = k // 2
            while j > 0:
                ti.loop_config(parallelize=int(self.logsize))
                for i in range(Nl):
                    ixj = i ^ j
                    if ixj > i:
                        if ((i & k) == 0 and up == 1) or ((i & k) != 0 and up == 0):
                            if (self.n2hash[i] > self.n2hash[ixj]):

                                self.n2hash[i], self.n2hash[ixj] = self.n2hash[ixj], self.n2hash[i]
                                self.n2idx[i], self.n2idx[ixj] = self.n2idx[ixj], self.n2idx[i]
                                #self.start_idx[int(self.n2hash[ixj])] = i

                        else:
                            if (self.n2hash[i] < self.n2hash[ixj]):

                                self.n2hash[i], self.n2hash[ixj] = self.n2hash[ixj], self.n2hash[i]
                                self.n2idx[i], self.n2idx[ixj] = self.n2idx[ixj], self.n2idx[i]
                                #self.start_idx[int(self.n2hash[ixj])] = i
                j = j // 2
            k = k * 2

    @ti.kernel
    def fill_start_idx(self):

        for k in range(self.start_idx.shape[0]):

            self.start_idx[k] = -1#self.max_hash


    @ti.kernel
    def solve_start_idx(self):

        for k in range(self.idx.shape[0]):

            if k <= self.start_idx[ int(self.n2hash[k]) ] or self.start_idx[ int(self.n2hash[k]) ] == -1:

                self.start_idx[ int(self.n2hash[k]) ] = k



    def update(self): # Main calling for the Register
        #print(self.nparticles)
        self.cell_list()
        self.copy_to_n2()
        self.bitonic_sort_hash( 1 , self.padded_size )  # 1 pour trier en ordre croissant
        self.copy_from_n2()
        self.fill_start_idx()
        self.solve_start_idx()

    @ti.kernel
    def copy_to_n2(self):

        for i in range(self.nparticles):
            self.n2idx[i] = self.idx[i]
            self.n2hash[i] = self.hashlist[i]

        for i in range(self.nparticles, self.padded_size):
            self.n2idx[i] = self.max_hash + 1
            self.n2hash[i] = self.max_hash + 1

    @ti.kernel
    def copy_from_n2(self):

        for i in range(self.nparticles):
            self.idx[i] = int(self.n2idx[i])
            self.hashlist[i] = int(self.n2hash[i])


    @ti.kernel
    def cell_list(self):

        obj = self.particlesystem

        for k in range(self.nparticles):

            part = obj.pos[k]

            xred = int((part[0] + self.spacedim) * self.redcell) + 1
            yred = int((part[1] + self.spacedim) * self.redcell) + 1
            zred = int((part[2] + self.spacedim) * self.redcell) + 1

            hashcode = xred * self.dim * self.dim + yred * self.dim + zred
            self.hashlist[k] = hashcode

        return
