import pyglet

class ArmEnv(object):
    def __init__(self) -> None:
        pass

    def step(self, action):
        pass

    def reset(self,):
        pass

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer()
        self.viewer.render()

class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_indo):
        #画出手臂等
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)

        #窗口背景颜色
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.batch = pyglet.graphics.Batch()

        self.point = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [50, 50,
                     50, 100,
                     100, 100,
                     100, 50]),
            ('c3B', (86, 109, 249) * 4)
        )

        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4)
        )

        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,
                     100, 160,
                     200, 160,
                     200, 150]),
            ('c3B', (249, 86, 86) * 4)
        )
    
    def render(self):
        #刷新并呈现在屏幕上
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
    
    def on_draw(self):
        #刷新手臂等位置
        self.clear()
        self.batch.draw()
    
    def _update_arm(self):
        #更新手臂的位置信息
        pass


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()