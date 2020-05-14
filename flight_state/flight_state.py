

class FlightState:
    def __init__(self,
                 analysis_type=None,
                 aircraft=None, #Aerosandbox object
                 ):
        if analysis_type = "simple":
            pass
        elif analysis_type = "alpha_sweep":
            pass
        elif analysis_type = "stability_sweep":
            pass
    def define_stability(self):
        pass
    def is_stable(self):
        pass
    _
import numpy as np
from numpy import linalg as LA
import pandas as pd

t_0_5 = lambda miu : 0.69 / np.abs(miu)
class Stability:
    def __init__(self, aircraft_deck, aero_deck, atm_deck, flight_point):
        self.aircraft_deck = aircraft_deck
        self.aero_deck = aero_deck
        self.atm_deck = atm_deck
        self.stability_modes_eig = self.find_eigenvalues()
        self.alpha = flight_point["alpha"]
        self.beta = flight_point["beta"]
        self.p = flight_point["p"]
        self.q = flight_point["q"]
        self.r = flight_point["r"]


    def find_eigenvalues(self):
        G = self.atm_deck["G"]
        RO = self.atm_deck["ro"]

        inertia_tensor = self.aircraft_deck["inertia_tensor"]
        i_x = inertia_tensor[0][0]
        i_y = inertia_tensor[1][1]
        i_z = inertia_tensor[2][2]
        i_xy = inertia_tensor[0][1]
        i_xz = inertia_tensor[0][2]
        i_zy = inertia_tensor[2][1]

        cog = self.aircraft_deck["cog_mac"]
        wing_a = self.aircraft_deck["wing_area"]
        span = self.aircraft_deck["span"]
        chord = self.aircraft_deck["chord"]
        mass = self.aircraft_deck["mass"]

        u_0 = self.aero_deck["u_0"]
        mach = self.aero_deck["mach"]
        cd_0 = self.aero_deck["cd_0"]
        cd_u = self.aero_deck["cd_u"]
        cl_u = self.aero_deck["cl_u"]
        cl_0 = self.aero_deck["cl_0"]
        cm_u = self.aero_deck["cm_u"]
        cd_a = self.aero_deck["cd_a"]
        cl_a = self.aero_deck["cl_a"]
        cm_a = self.aero_deck["cm_a"]
        cm_a_dot = self.aero_deck["cm_a_dot"]
        cz_q = self.aero_deck["cz_q"]
        cm_q = self.aero_deck["cm_q"]
        cy_b = self.aero_deck["cy_b"]
        cl_b = self.aero_deck["cl_b"]
        cn_b = self.aero_deck["cn_b"]
        cy_p = self.aero_deck["cy_p"]
        cl_p = self.aero_deck["cl_p"]
        cn_p = self.aero_deck["cn_p"]
        cy_r = self.aero_deck["cy_r"]
        cl_r = self.aero_deck["cl_r"]
        cn_r = self.aero_deck["cn_r"]

        q_dyn = 0.5 * RO * u_0 ** 2

        # U Derivatives
        X_u = -(cd_u + 2 * cd_0) * (q_dyn * wing_a) / (u_0 * mass)

        Z_u = -(cl_u + 2 * cl_0) * (q_dyn * wing_a) / (u_0 * mass)

        M_u = cm_u * (q_dyn * wing_a * chord) / (u_0 * i_y)

        # W Derivatives
        X_w = -(cd_a - cl_0) * (q_dyn * wing_a) / (u_0 * mass)

        Z_w = -(cl_a + cd_0) * (q_dyn * wing_a) / (u_0 * mass)

        M_w = cm_a * (q_dyn * wing_a * chord) / (u_0 * i_y)

        # W_dot derivatives
        X_w_dot = 0

        Z_w_dot = 0

        M_w_dot = cm_a_dot * (chord / (2 * u_0)) * (q_dyn * wing_a * chord) / (u_0 * i_y)

        # q derivatives
        X_q = 0

        Z_q = cz_q * (chord / (2 * u_0)) * (q_dyn * wing_a) / mass

        M_q = cm_q * (chord / (2 * u_0)) * (q_dyn * wing_a * chord) / i_y

        A_long = np.array([[X_u, X_w, 0, -G],
                           [Z_u, Z_w, u_0, 0],
                           [M_u + M_w_dot * Z_u, M_w + M_w_dot * Z_w, M_q + M_w_dot * u_0, 0],
                           [0, 0, 1, 0]])

        longitudinal_eig, vec_long = LA.eig(A_long)

        # Beta derivativatives
        Y_b = q_dyn * wing_a * cy_b / mass

        L_b = q_dyn * wing_a * span * cl_b / i_x

        N_b = q_dyn * wing_a * span * cn_b / i_z

        # P derivatives
        Y_p = q_dyn * wing_a * span * cy_p / (2 * mass * u_0)

        L_p = q_dyn * wing_a * (span ** 2) * cl_p / (2 * i_x * u_0)

        N_p = q_dyn * wing_a * (span ** 2) * cn_p / (2 * i_z * u_0)

        # R derivatives
        Y_r = q_dyn * wing_a * span * cy_r / (2 * mass * u_0)

        L_r = q_dyn * wing_a * (span ** 2) * cl_r / (2 * i_x * u_0)

        N_r = q_dyn * wing_a * (span ** 2) * cn_r / (2 * i_z * u_0)

        theta = 5
        theta = np.radians([theta])
        A_lat = np.array([[Y_b / u_0, Y_p / u_0, -(1 - (Y_r / u_0)), (G / u_0) * np.cos(theta)[0]],
                          [L_b, L_p, L_r, 0],
                          [N_b, N_p, N_r, 0],
                          [0, 1, 0, 0]])

        lateral_eig, vec_lat = LA.eig(A_lat)

        stability_modes_eig = self.define_modes(lateral_eig, longitudinal_eig)

        return stability_modes_eig

    def define_modes(self, lateral_eig, longitudinal_eig):

        # Defining Longitudinal modes
        max_eig = 0
        min_eig = 0
        for eigenvalue in longitudinal_eig:
            if np.abs(np.real(eigenvalue)) > max_eig - 0.000001:
                max_eig = np.abs(np.real(eigenvalue))
                short_period_eig = eigenvalue
            elif (min_eig > 0 and np.abs(np.real(eigenvalue)) < min_eig - 0.000001) or min_eig == 0:
                phugoid_eig = eigenvalue
                min_eig = np.abs(np.real(eigenvalue))

        # Defining Longitudinal modes
        max_eig = 0

        for i, eigenvalue in enumerate(lateral_eig):
            for j, eigenvalue_2 in enumerate(lateral_eig):
                if i != j:
                    if np.imag(eigenvalue) != 0:
                        if np.abs(np.abs(np.imag(eigenvalue)) - np.abs(np.imag(eigenvalue_2))) < 0.00001:
                            dutch_roll_eig = eigenvalue
                        else:
                            continue
            if "dutch_roll_eig" in locals():
                if np.abs(np.real(eigenvalue)) > max_eig and not eigenvalue == dutch_roll_eig:
                    max_eig = np.abs(np.real(eigenvalue))
                    roll_eig = eigenvalue
                elif not eigenvalue == dutch_roll_eig:
                    spiral_eig = eigenvalue
                else:
                    continue
            else:
                if np.abs(np.real(eigenvalue)) > max_eig:
                    max_eig = np.abs(np.real(eigenvalue))
                    roll_eig = eigenvalue
                else:
                    spiral_eig = eigenvalue

        eigenvalue_list = [["short_period", short_period_eig],
                           ["phugoid", phugoid_eig],
                           ["spiral", spiral_eig],
                           ["dutch_roll", dutch_roll_eig],
                           ["roll", roll_eig]]
        stability_modes_eig = pd.DataFrame(eigenvalue_list, columns=["mode", "eigenvalue"])

        return stability_modes_eig
