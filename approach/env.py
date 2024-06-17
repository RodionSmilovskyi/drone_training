import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data
import datetime
import os
from PIL import Image

class Env(gym.Env):
    def __init__(self, record_steps = False):
        self._agent_location = np.array([0, 0, 0], dtype=np.int32)
        self._target_location = np.array([10, 10, 10], dtype=np.int32)
        self._record_steps = record_steps
        # self.observation_space = gym.spaces.Dict({
        #     "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        # })

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]),  # thrust
            1: np.array([-1, 1]),  # yaw
            2: np.array([-1, 1]),  # roll
            3: np.array([-1, 1]),  # pitch
        }

        self.world_space = gym.spaces.Box(low=np.array([-20, -20, 0]), high=np.array([20, 20, 10]), dtype=np.float32)

        self._client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)

        for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self._client)


    def step(self, action):
        for i in range(4):
            p.applyExternalForce(self.drone_id, i, forceObj=[0, 0, 2.45], posObj=[0, 0, 0], flags=p.LINK_FRAME)

        if self._record_steps == True:
            self._save_step_view()
        
        self._step_number = self._step_number + 1

        p.stepSimulation()
        return None, 0, self._step_number == 2, False

    def render(self, mode = 'human'):
        pass # to implement

    def close(self):
        p.disconnect()
        pass # to implement

    def reset(self, seed : int = None, options : dict = None):
        p.resetSimulation()

        self._run_label = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._step_number = 0
        self._scene_setup()
       
        pass # to implement

    def _scene_setup(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.drone_id = p.loadURDF("drone.urdf", basePosition = [0, 0, 1])

        self._place_target()

        self._look_at(self.drone_id, self.target_id)


    def _place_target(self):
        random_position = self.world_space.sample()

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="assets/a_cube.obj")
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="assets/a_cube.obj")
        self.target_id = p.createMultiBody(baseCollisionShapeIndex = collisionShapeId, baseVisualShapeIndex = visualShapeId, basePosition = random_position.tolist())

        pass

    def _look_at(self, source_id, target_id):
        source_pos, _ = p.getBasePositionAndOrientation(source_id)
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        direction = np.array(target_pos) - np.array(source_pos)
        direction /= np.linalg.norm(direction)

        yaw = np.arctan2(direction[1], direction[0])
        pitch = np.arctan2(-direction[2], np.sqrt(direction[0]**2 + direction[1]**2))
        roll = np.arctan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, pitch, yaw])
        p.resetBasePositionAndOrientation(source_id, source_pos, quat)

    def _save_step_view(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        target = np.dot(rot_mat,np.array([self.world_space.high[0], 0, 0])) + np.array(pos)

        drone_cam_view = p.computeViewMatrix(cameraEyePosition=pos, cameraTargetPosition=target, cameraUpVector=[0, 0, 1])
        drone_cam_pro =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=0,
                                                      farVal=np.max(self.world_space.high)
                                                      )
        [width, height, rgbImg, dep, seg] = p.getCameraImage(width=256,
                                            height=256,
                                            shadow=1,
                                            viewMatrix=drone_cam_view,
                                            projectionMatrix=drone_cam_pro,
                                            )

        # Convert the image data to a numpy array
        image = np.array(rgbImg, dtype=np.uint8).reshape((height, width, 4))

        # Create a PIL image from the numpy array
        img = Image.fromarray(image, 'RGBA')

        dirname = os.path.join(os.getcwd(), "approach", "records", self._run_label)
        os.makedirs(dirname, exist_ok=True)
        img.save(os.path.join(dirname, f"{self._step_number}.png"))

    


