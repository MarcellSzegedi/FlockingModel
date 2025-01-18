
##################################################
## I would like to get some feedback on my work ##
##################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mpl.rcParams['animation.embed_limit'] = 1500


class Flocking:

    acceleration_scaler = 10
    max_velocity = 5
    wall_sensibility = 0.4

    def __init__(
            self,
            n_boids: int,
            time_step: float,
            starting_area: np.ndarray,
            neighbour_radius: float,
            rule_weights: np.ndarray=np.array([1/3, 1/3, 1/3])
    ):
        self.n_boids = n_boids
        self.time_step = time_step
        self.starting_area = starting_area
        self.midpoint_vector = np.array([starting_area[0] / 2, starting_area[1] / 2, starting_area[2] / 2])
        self.neighbour_radius = neighbour_radius
        self.rule_weights = rule_weights
        self.positions = None
        self.velocities = None

    def initialise_flocking(self):
        self.positions = np.concatenate((
            np.random.uniform(self.starting_area[0] * 0.3, self.starting_area[0] * 0.7, self.n_boids).reshape(self.n_boids, 1),
            np.random.uniform(self.starting_area[1] * 0.3, self.starting_area[1] * 0.7, self.n_boids).reshape(self.n_boids, 1),
            np.random.uniform(self.starting_area[2] * 0.3, self.starting_area[2] * 0.7, self.n_boids).reshape(self.n_boids, 1)), axis=1)
        self.velocities = np.random.uniform(-self.max_velocity * 0.5, self.max_velocity * 0.5, (self.n_boids, 3))

    def update_boid_attributes(self):
        new_accelerations = self.calc_new_accelerations() * self.acceleration_scaler * self.time_step
        self.velocities += new_accelerations
        self.positions += self.velocities * self.time_step

        self.pos_boundary_check()
        self.vel_boundary_check()

    def calc_new_accelerations(self):
        boids_in_vicinity = self.collect_boids_in_vicinity()
        boid_close_to_wall = self.get_wall_close_boids()
        acc_weights = self.calc_rule_weights(boid_close_to_wall)
        raw_acceleration = (acc_weights[:, 0][:, np.newaxis] * self.calc_separation_rule(boids_in_vicinity) +
                            acc_weights[:, 1][:, np.newaxis] * self.calc_alignment_rule(boids_in_vicinity) +
                            acc_weights[:, 2][:, np.newaxis] * self.calc_cohesion_rule() +
                            acc_weights[:, 3][:, np.newaxis] * self.calc_wall_rule(boid_close_to_wall))

        return raw_acceleration / np.linalg.norm(raw_acceleration)

    def collect_boids_in_vicinity(self):
        boids_in_vicinity = []
        for boid in self.positions:
            boid_distances = np.linalg.norm(boid - self.positions, axis=1, keepdims=True)
            is_boid_in_vicinity = np.where((0 < boid_distances) & (boid_distances <= self.neighbour_radius))[0]
            boid_in_vic_info = np.stack((self.positions[is_boid_in_vicinity], self.velocities[is_boid_in_vicinity]), axis=0)
            boids_in_vicinity.append(boid_in_vic_info)

        return boids_in_vicinity

    def calc_rule_weights(self, boid_close_to_wall: np.ndarray):
        wall_rule_weight = np.zeros(len(boid_close_to_wall))
        wall_distance, wall_idx = self.calc_boid_wall_dist()
        wall_rule_scaler = 1 - np.power(wall_distance / self.starting_area[wall_idx % 3] * self.wall_sensibility, 5)
        wall_rule_weight[boid_close_to_wall] = wall_rule_scaler[boid_close_to_wall]

        acceleration_weights = np.hstack((
            self.rule_weights * np.ones((len(boid_close_to_wall), 3)) * (1 - wall_rule_weight[:, np.newaxis]),
            wall_rule_weight[:, np.newaxis]))

        return acceleration_weights

    def calc_boid_wall_dist(self):
        zero_col = np.zeros(len(self.positions))[:, np.newaxis]
        one_col =  np.ones(len(self.positions))[:, np.newaxis]
        wall_1_dist = np.hstack((self.positions[:, [0, 1]], zero_col))
        wall_2_dist = np.hstack((self.positions[:, 0][:, np.newaxis], zero_col, self.positions[:, 2][:, np.newaxis]))
        wall_3_dist = np.hstack((zero_col, self.positions[:, [1, 2]]))
        wall_4_dist = np.hstack((self.positions[:, [0, 1]], one_col * self.starting_area[0]))
        wall_5_dist = np.hstack((self.positions[:, 0][:, np.newaxis], one_col * self.starting_area[1], self.positions[:, 2][:, np.newaxis]))
        wall_6_dist = np.hstack((one_col * self.starting_area[2], self.positions[:, [1, 2]]))

        wall_distances = np.hstack((
            np.linalg.norm(self.positions - wall_1_dist, axis=1)[:, np.newaxis],
            np.linalg.norm(self.positions - wall_2_dist, axis=1)[:, np.newaxis],
            np.linalg.norm(self.positions - wall_3_dist, axis=1)[:, np.newaxis],
            np.linalg.norm(self.positions - wall_4_dist, axis=1)[:, np.newaxis],
            np.linalg.norm(self.positions - wall_5_dist, axis=1)[:, np.newaxis],
            np.linalg.norm(self.positions - wall_6_dist, axis=1)[:, np.newaxis]))

        return np.min(wall_distances, axis=1), np.argmin(wall_distances, axis=1)

    def get_wall_close_boids(self):
        x_mask = (self.positions[:, 0] < self.starting_area[0] * self.wall_sensibility) | (
                self.positions[:, 0] > self.starting_area[0] * (1 - self.wall_sensibility))
        y_mask = (self.positions[:, 1] < self.starting_area[1] * self.wall_sensibility) | (
                self.positions[:, 1] > self.starting_area[1] * (1 - self.wall_sensibility))
        z_mask = (self.positions[:, 2] < self.starting_area[2] * self.wall_sensibility) | (
                self.positions[:, 2] > self.starting_area[2] * (1 - self.wall_sensibility))

        boid_idx_close_wall = x_mask & y_mask & z_mask
        return boid_idx_close_wall

    def calc_separation_rule(self, boids_in_vicinity):
        new_accelerations = []
        for boid, boid_neighbours in zip(self.positions, boids_in_vicinity):
            if boid_neighbours.size > 0:
                neighbour_vectors = boid - boid_neighbours[0]
                normalised_neighbour_vectors = neighbour_vectors / np.linalg.norm(neighbour_vectors, axis=1, keepdims=True)

                new_raw_acceleration_vector = np.sum(normalised_neighbour_vectors, axis=0)
                new_acc_vector = (new_raw_acceleration_vector / np.linalg.norm(new_raw_acceleration_vector)
                                  if np.linalg.norm(new_raw_acceleration_vector) != 0 else np.array([0, 0, 0]))
                new_accelerations.append(new_acc_vector)
            else:
                new_accelerations.append(np.array([0, 0, 0]))

        return np.array(new_accelerations)

    def calc_alignment_rule(self, boids_in_vicinity):
        new_accelerations = []
        for boid_neighbours in boids_in_vicinity:
            if boid_neighbours.size > 0:
                average_velocity = np.sum(boid_neighbours[1], axis=0)
                new_acc_vector = (average_velocity / np.linalg.norm(average_velocity)
                                  if np.linalg.norm(average_velocity) > 0 else np.array([0, 0, 0]))
                new_accelerations.append(new_acc_vector)
            else:
                new_accelerations.append(np.array([0, 0, 0]))

        return np.array(new_accelerations)

    def calc_cohesion_rule(self):
        center_of_flocking = np.mean(self.positions, axis=0)
        cohesion_vector = center_of_flocking - self.positions
        return cohesion_vector / np.linalg.norm(cohesion_vector, axis=1, keepdims=True)

    def calc_wall_rule(self, boid_close_to_wall: np.ndarray):
        new_accelerations = np.zeros_like(self.positions)
        goal_vector = (self.midpoint_vector + np.mean(self.positions, axis=0)) / 2
        new_accelerations[boid_close_to_wall] = ((goal_vector - self.positions[boid_close_to_wall]) / np.linalg.norm(
            goal_vector - self.positions[boid_close_to_wall], axis=1, keepdims=True))
        return new_accelerations

    def pos_boundary_check(self):
        for dim in range(3):
            min_violated = self.positions[:, dim] < 0
            max_violated = self.positions[:, dim] > self.starting_area[dim]

            if np.any(min_violated):
                self.positions[min_violated, dim] = 0
                self.velocities[min_violated, dim] *= -1 / 2

            if np.any(max_violated):
                self.positions[max_violated, dim] = self.starting_area[dim]
                self.velocities[max_violated, dim] *= -1 / 2

    def vel_boundary_check(self):
        v_max_violated = np.linalg.norm(self.velocities, axis=1, keepdims=True)[:, 0] > self.max_velocity
        if any(v_max_violated):
            self.velocities[v_max_violated] = (self.velocities[v_max_violated]
                                               * self.max_velocity
                                               / np.linalg.norm(self.velocities[v_max_violated], axis=1, keepdims=True))


def run_flocking_simulation(
        n_boids: int,
        time: float,
        time_step: float,
        starting_area: np.ndarray,
        neighbour_radius: float,
        rule_weights: np.ndarray
):
    flocking_model = Flocking(n_boids, time_step, starting_area, neighbour_radius, rule_weights)
    flocking_model.initialise_flocking()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, starting_area[0])
    ax.set_ylim(0, starting_area[1])
    ax.set_zlim(0, starting_area[2])

    # Initialize quiver plot
    normalized_velocities = flocking_model.velocities / np.linalg.norm(flocking_model.velocities, axis=1, keepdims=True)
    quiver = ax.quiver(
        flocking_model.positions[:, 0],
        flocking_model.positions[:, 1],
        flocking_model.positions[:, 2],
        normalized_velocities[:, 0],
        normalized_velocities[:, 1],
        normalized_velocities[:, 2],
        length=2.3,  # Increased from 2.0 to 5.0
        normalize=True,
        color='blue',
        linewidth=1.2,  # Added linewidth parameter
        arrow_length_ratio=1  # Controls the size of the arrow head
    )

    def update_attributes(frame):
        flocking_model.update_boid_attributes()

        # Remove previous arrows
        ax.cla()

        # Set axis limits again since we cleared the axis
        ax.set_xlim(0, starting_area[0])
        ax.set_ylim(0, starting_area[1])
        ax.set_zlim(0, starting_area[2])

        # Normalize velocities for consistent arrow lengths
        normalized_velocities = flocking_model.velocities / np.linalg.norm(flocking_model.velocities, axis=1,
                                                                           keepdims=True)

        # Create new quiver plot
        quiver = ax.quiver(
            flocking_model.positions[:, 0],
            flocking_model.positions[:, 1],
            flocking_model.positions[:, 2],
            normalized_velocities[:, 0],
            normalized_velocities[:, 1],
            normalized_velocities[:, 2],
            length=2.3,  # Increased from 2.0 to 5.0
            normalize=True,
            color='blue',
            linewidth=1.2,  # Added linewidth parameter
            arrow_length_ratio=1  # Controls the size of the arrow head
        )

        return quiver,

    frames = int(time // time_step) + 1
    interval = time_step * 1000

    animation = FuncAnimation(fig, update_attributes, frames=frames, interval=interval)

    html = animation.to_jshtml()
    with open('flocking_simulation.html', 'w') as f:
        f.write(html)

    plt.show()
    return animation


#########################################################
#########################################################

# Model Settings
n_boids = 500
time_step = 0.1
time = 250
starting_area = np.array([100, 100, 100])
neighbour_radius = np.mean(starting_area) / 10
rule_weights = np.array([0.15, 0.45, 0.4])

#########################################################
#########################################################

animation = run_flocking_simulation(n_boids, time, time_step, starting_area, neighbour_radius, rule_weights)
