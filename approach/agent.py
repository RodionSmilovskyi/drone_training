import numpy as np
import tensorflow as tf


class PolicyNetwork(tf.keras.Model):
    def __init__(self, env):
        super(PolicyNetwork, self).__init__()
        self.env = env
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(env.action_space.n, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)

class Agent:
    def __init__(self, env, learning_rate=0.01):
        self.env = env
        self.policy = PolicyNetwork(env)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state)
        return np.random.choice(self.env.action_space.n, p=probs[0].numpy())

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + (self.env.discount_factor ** pw) * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
        log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

        loss = -tf.reduce_mean(discounted_rewards * log_probs)

        with tf.GradientTape() as tape:
            tape.watch(self.policy.trainable_variables)
            grads = tape.gradient(loss, self.policy.trainable_variables)
            
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))