import taichi as ti
import numpy as np
import time

ti.init(arch=ti.gpu)

# - Screen parameters

screen_res = (800, 800)              # here a grid of 80 * 80
screen_to_world_ratio = 10.0
boundary = (
    screen_res[0] / screen_to_world_ratio,
    screen_res[1] / screen_to_world_ratio,
)

bg_color = 0x000000
particle_color = 0x068587
boundary_color = 0xEBACA2
particle_radius = 5.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# - Simulations primal parameters

nparticles = 4000
ntriangles = 1
dt = 0.01
T = 1

k_spring = 1
l0_spring = 10
sigma = 1.0
e0 = 1.0
rad = -2.0
rlim_lj = 100
epsilon = 1

ncells = 10

dim = 2
boundary_x = (0,screen_res[0])
boundary_y = (0,screen_res[1])
redcell_x = ( dim - 2 ) / (screen_res[0]  * 2)
redcell_y = ( dim - 2 ) / (screen_res[1]  * 2)
maxhash = (dim+2)**2

# - Initialisation of datas

	# -- Particletype 1

p_pos = ti.Vector.field(dim, float)
p_vel = ti.Vector.field(dim, float)
p_force = ti.Vector.field(dim, float)

p_idx = ti.field(ti.i32)
p_hashlist = ti.field(ti.i32)
p_start_idx = ti.field(ti.i32)

	# -- Particletype 2

t_link_0 = ti.field(int)
t_link_1 = ti.field(int)

t_pos = ti.Vector.field(dim, float)
t_vel = ti.Vector.field(dim, float)
t_force = ti.Vector.field(dim, float)

t_idx = ti.field(ti.i32)
t_hashlist = ti.field(ti.i32)
t_start_idx = ti.field(ti.i32)

# - Allocation of memory

ti.root.dense(ti.i, nparticles).place(p_pos, p_vel, p_force)
ti.root.dense(ti.j, ntriangles*3).place(t_pos, t_vel, t_force)
ti.root.dense(ti.j, ntriangles*3).place(t_link_0, t_link_1)

ti.root.dense(ti.i, nparticles).place(p_idx, p_hashlist)
ti.root.dense(ti.i, nparticles).place(t_idx, t_hashlist)
ti.root.dense(ti.i, maxhash).place(p_start_idx, t_start_idx)

# - Utility functions

@ti.data_oriented
class Cell_reg: # - useful for a particlesystem object

    def __init__(self, parts, dim, spacedim):  # Necessary for cell listing

        self.pos = parts
        self.nparticles = self.pos.shape[0]
        self.spacedim = spacedim
        self.dim = dim + 2  # particles should not be in the extremities

        self.padded_size = 1 << (self.nparticles - 1).bit_length()  # Prochaine puissance de 2
        self.logsize = np.log2(self.padded_size)

        self.idx = ti.field( shape=self.nparticles, dtype=ti.i32)
        self.hashlist = ti.field( shape=self.nparticles, dtype=ti.i32)
        self.start_idx = ti.field( shape=(self.dim)**3, dtype=ti.i32)

        self.n2idx = ti.field(dtype=ti.f32, shape=self.padded_size)
        self.n2hash = ti.field(dtype=ti.f32, shape=self.padded_size)

        self.max_hash = self.dim**3 + 1
        self.redcell_x = ( dim ) / ( self.spacedim[0] ) # use of dim to keep particles inside
        self.redcell_y = ( dim ) / ( self.spacedim[1] )

        for k in range(self.idx.shape[0]): self.idx[k] = k


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

                        else:
                            if (self.n2hash[i] < self.n2hash[ixj]):

                                self.n2hash[i], self.n2hash[ixj] = self.n2hash[ixj], self.n2hash[i]
                                self.n2idx[i], self.n2idx[ixj] = self.n2idx[ixj], self.n2idx[i]

                j = j // 2
            k = k * 2

    @ti.kernel
    def fill_start_idx(self):

        for k in range(self.start_idx.shape[0]):

            self.start_idx[k] = -1 #self.max_hash

    @ti.kernel
    def solve_start_idx(self):

        for k in range(self.idx.shape[0]):

            if k <= self.start_idx[ int(self.hashlist[k]) ] or self.start_idx[ int(self.hashlist[k]) ] == -1:

                self.start_idx[ int(self.hashlist[k]) ] = k

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
    def cell_list2d(self):

        for k in range(self.nparticles):

            part = self.pos[k]

            xred = int((part[0] + 0.0) * self.redcell_x) + 1
            yred = int((part[1] + 0.0) * self.redcell_y) + 1

            hashcode = xred * self.dim + yred
            self.hashlist[k] = hashcode

        return

    def update(self, parts): # Main calling for the Register
        self.pos = parts
        self.cell_list2d()
        self.copy_to_n2()
        self.bitonic_sort_hash( 1 , self.padded_size )  # 1 pour trier en ordre croissant
        self.copy_from_n2()
        self.fill_start_idx()
        self.solve_start_idx()

            # Necessary for Mesh-Particle interaction

p_reg = Cell_reg(p_pos, ncells, boundary)
t_reg = Cell_reg(t_pos, ncells, boundary)

# - Utility ti functions

@ti.func
def dist(pos1, pos2):

    drx = pos1[0] - pos2[0]
    dry = pos1[1] - pos2[1]

    r2 = drx*drx + dry*dry
    r = ti.sqrt(r2)

    return r, r2, drx, dry

@ti.func
def lj_f(r, rad, sigma, e0):
	r += rad
	intensity = - e0 * ( 48*(sigma/r)**12 - 24*(sigma/r)**6 )
	return intensity

@ti.func
def hashToCell2d_ti(hashcode, dim):

    xpos = hashcode//dim
    ypos = hashcode%dim

    return xpos, ypos

@ti.func
def cellToHash2d_ti(xpos, ypos, dim):

    hashcode = xpos*dim + ypos

    return ti.cast(hashcode, int)

# - Kernel functions

@ti.kernel
def reinit_forces():

	p_force.fill(0.0)
	t_force.fill(0.0)


@ti.kernel
def springs():

	for k in t_link_0:

		r, r2, drx, dry = dist( t_pos[ t_link_0[k] ] , t_pos[ t_link_1[k] ] )
		intensity = - (r - l0_spring) * k_spring

		t_force[ t_link_0[k] ][0] += intensity * (drx/r)
		t_force[ t_link_0[k] ][1] += intensity * (dry/r)

		t_force[ t_link_1[k] ][0] -= intensity * (drx/r)
		t_force[ t_link_1[k] ][1] -= intensity * (dry/r)

@ti.kernel
def integrate():

	for k in range(nparticles):
		p_pos[k] += p_force[k] * dt

	for k in range(ntriangles*3):
		t_pos[k] += t_force[k] * dt

@ti.kernel
def add_noise():

	for k in p_force:
		p_force[k][0] += (ti.random()-0.5)*T
		p_force[k][1] += (ti.random()-0.5)*T
	for k in t_force:
		t_force[k][0] += (ti.random()-0.5)*T
		t_force[k][1] += (ti.random()-0.5)*T

@ti.kernel
def add_gravity():
	for k in p_force:
		p_force[k][1] += -10


@ti.kernel
def add_centerforce(x: float,y: float):
	for k in range(p_force.shape[0]):
		p_force[k][0] -= p_pos[k][0] - x
		p_force[k][1] -= p_pos[k][1] - y

@ti.kernel
def lj_force(n: int, plist: ti.template(), forces: ti.template(), reg: ti.template()):

    #ti.loop_config(serialize=True)
    for k in range(n):  # Loop for all particles

        xp, yp = hashToCell2d_ti(reg.hashlist[k], ncells+2)
        #print("original cell = ",end=" ")
        #print(xp,yp)

        for i in range(xp-1,xp+2):
            for j in range(yp-1,yp+2):

                hash_part = cellToHash2d_ti(i,j,ncells+2)

                #print("target cell = ",end=" ")
                #print(i,j,reg.start_idx[hash_part])

                if(i>=0 and j>=0 and i<ncells+2 and j<ncells+2 and reg.start_idx[hash_part]!=-1):

                    pos = int( reg.start_idx[hash_part] ) # index in the hashlist corresponding
                    hashcode_init = reg.hashlist[ pos ]
                    #print("condition passed",end=" ")
                    #print(pos)

                    while pos < nparticles and reg.hashlist[pos] == hashcode_init:

                        if(k!=reg.idx[pos]):

                    # -----------------------------------------------

                            r, r2, drx, dry = dist(plist[ reg.idx[pos] ],plist[k])

                            #print("r = ")
                            #print(r)
                            #print(k,pos)
                            if(r<rlim_lj):

                            	intensity = lj_f(r, rad, sigma, e0)
                            	if intensity > 100 or intensity < -100 :
                            		intensity = -10
                            	#print(r,intensity)

                            	forces[p_idx[pos]][0] -= intensity * (drx/r)
                            	forces[p_idx[pos]][1] -= intensity * (dry/r)

                            	forces[k][0] += intensity * (drx/r)
                            	forces[k][1] += intensity * (dry/r)

                    # -----------------------------------------------

                        pos += 1 # go to the next particle
                        if(pos == n): break
                        hashcode_current = reg.hashlist[ pos ]


@ti.kernel
def apply_boundary():

	for k in range(nparticles):

		if p_pos[k][0] <= boundary_x[0]:
			p_pos[k][0] += boundary_x[0] + epsilon * ti.random()

		if p_pos[k][0] >= boundary_x[1]:
			p_pos[k][0] -= boundary_x[1] + epsilon * ti.random()

		if p_pos[k][1] <= boundary_y[0]:
			p_pos[k][1] += boundary_y[0] + epsilon * ti.random()

		if p_pos[k][1] >= boundary_y[1]:
			p_pos[k][1] -= boundary_y[1] - epsilon * ti.random()

@ti.kernel
def init_rdparticles():

	for k in range(nparticles):
		for c in ti.static(range(dim)):
			p_pos[k][c] = (ti.random()) * boundary[0]

	for k in range(ntriangles*3):
		for c in ti.static(range(dim)):
			t_pos[k][c] = (ti.random()) * boundary[0]

	for k in range(ntriangles):
		t_link_0[3*k] = k*3
		t_link_1[3*k] = k*3 + 1
		t_link_0[3*k+1] = k*3 + 1
		t_link_1[3*k+1] = k*3 + 2
		t_link_0[3*k+2] = k*3 + 2
		t_link_1[3*k+2] = k*3

# -  Rendering function

def render(gui):

    gui.clear(bg_color)
    p_pos_np = p_pos.to_numpy()
    t_pos_np = t_pos.to_numpy()
    t_lines_0 = t_link_0.to_numpy()
    t_lines_1 = t_link_1.to_numpy()

    gui.rect(topleft=(boundary_x[0],boundary_y[1]), bottomright=(boundary_x[1],boundary_y[0]), color=0xFFFFFF)

    for j in range(dim):
        p_pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
        t_pos_np[:, j] *= screen_to_world_ratio / screen_res[j]

    for j in range(len(t_lines_0)):
        coord0 = t_pos_np[t_lines_0[j]]
        coord1 = t_pos_np[t_lines_1[j]]
        gui.line(coord0[:2], coord1[:2], radius=1, color=boundary_color)

    gui.circles(p_pos_np, radius=particle_radius, color=particle_color)
    gui.circles(t_pos_np, radius=particle_radius, color=boundary_color)
    gui.show()

# - logs functions

logfile = open("logfile.txt", "w")

def print_logs(logfile):

    log_hashlist = p_reg.hashlist.to_numpy()
    logfile.write("Hashlist \n")
    logfile.write(str(log_hashlist) + "\n")

    log_idx = p_reg.idx.to_numpy()
    logfile.write("Idx \n")
    logfile.write(str(log_idx) + "\n")

    log_pos = p_pos.to_numpy()
    logfile.write("Pos \n")
    #logfile.write(str(log_pos) + "\n")

    log_start_idx = p_reg.start_idx.to_numpy()
    logfile.write("Start_Idx \n")
    for k in range( len(log_start_idx) ):
        logfile.write(str(log_start_idx[k])+" ")
        if(k%20==0):
            logfile.write("\n")
    logfile.write("\n")

# --- Main function --- #

def main():
    print("main starting")
    init_rdparticles()
    p_reg.update(p_pos)
    t_reg.update(t_pos)

    gui = ti.GUI("Particles system", screen_res)
    print("timeloop starting")
    #for t in range(100):
    while gui.running and not gui.get_event(gui.ESCAPE):
        step()
        if gui.frame % 20 == 1:
            print_logs(logfile)

        render(gui)
    logfile.close()

def step():

	reinit_forces()
	p_reg.update(p_pos)
	#t_reg.update(t_pos)
	#springs()
	lj_force(nparticles, p_pos, p_force, p_reg)
	#add_gravity()
	add_centerforce(40.0,40.0)
	add_noise()
	integrate()
	apply_boundary()

main()
