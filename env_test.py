import pybullet as p
import pybullet_data
import time

# Start PyBullet
client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeId = p.loadURDF("plane.urdf", physicsClientId=client)
# Load the URDF file
quadcopter_id = p.loadURDF("drone.urdf")

collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents = [0.1,0.1,0.1])
visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents = [0.1,0.1,0.1])
cube_id = p.createMultiBody(baseCollisionShapeIndex = collisionShapeId, baseVisualShapeIndex = visualShapeId, basePosition = [1,1,1])
# # Set gravity
p.setGravity(0, 0, -9.8, physicsClientId=client)

# # Simulate thrust
for _ in range(10000):
    # Apply a force to each of the four propellers
    for i in range(4):
        p.applyExternalForce(quadcopter_id, i, forceObj=[0, 0, 2.45], posObj=[0, 0, 0], flags=p.LINK_FRAME)
    
    # Step simulation
    p.stepSimulation()
    
    # Pause for a bit
    time.sleep(0.01)

# # Disconnect from PyBullet
p.disconnect()
