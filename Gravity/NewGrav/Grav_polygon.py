class Gravity_Polygon:
    def __init__(self, x, z):
        self.obs_N = 15
        self.thick = constant64(2)
        self.x_obv = tf.linspace(constant64(-70.), constant64(70.), obs_N)
        y_obv = tf.zeros(tf.shape(self.x_obv), dtype=tf.float64)
