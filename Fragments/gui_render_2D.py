@ti.data_oriented
class Renderer_2D:

    def __init__(self, resx, resy, stwr):

        self.screen_res = (resx, resy)
        self.stwr = stwr
        self.gui = ti.GUI("Particles system", self.screen_res)

    def clear_screen(self):
        bg_color = 0x000000 # Black
        self.gui.clear(bg_color)

    def render_particles(self,obj,prad,pc):
      
        p_pos_np = obj.pos.to_numpy()
      
        for j in range(2): p_pos_np[:, j] *= self.stwr / self.screen_res[j]
          
        self.gui.circles(p_pos_np, radius=prad, color=pc)

    @ti.kernel
    def fill_plinks(self, obj: ti.template(), pos0: ti.template(), pos1: ti.template()):
      
        for i in range(obj.nseg):
          
            pos0[i] = obj.pos[ obj.link0[i] ]
            pos1[i] = obj.pos[ obj.link1[i] ]
		
    def render_lines(self,obj,rad,sc):

        pos0 = ti.Vector.field(2, shape=obj.nseg, dtype=ti.f32)
        pos1 = ti.Vector.field(2, shape=obj.nseg, dtype=ti.f32)

        self.fill_plinks(obj, pos0, pos1)
	
        link0 = pos0.to_numpy()
        link1 = pos1.to_numpy()

        for j in range(2):
        	link0[:, j] *= self.stwr / self.screen_res[j]
        	link1[:, j] *= self.stwr / self.screen_res[j]

        self.gui.lines(link0,link1, radius=rad, color=sc)
        #self.gui.circles(link0, radius=rad, color=sc)

    def show(self):
        self.gui.show()
