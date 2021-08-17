"""
    GUI Creator: Suchet Sapre
    Last Revision Date: 06/25/20
"""

import math
import numpy as np
from scipy.linalg import expm
from tkinter import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import time

''' TODO: Convert global variables to class variables '''
t_interval = 45  # number of days per time point
drug_optimization_interval = 0.5  # is actually 0.01 in the PNAS paper, currently keeping at 0.1-0.2 to keep run-time
# short
total_cells = 1e9
death_threshold = 1e13
fig, ax_arr = None, None


def U(A: np.array, x: np.array) -> np.array:
    """ Heavyside step function
        TODO: Make it more concise.
        This function is not used.
    """
    B = A[:, :]  # creating a copy of the original A matrix to change
    for i in range(len(x)):
        if x[i] < 1:
            B[i, :] = 0
    return B


def diag(x: np.array) -> np.array:
    """ Places vector components on the diagonal of a zero-matrix.
        TODO: Can probably replace all function calls with the numpy function. Using diag makes it more readable.
    """
    return np.diagflat(x)


class Patient:
    """ The Patient class contains all of the functions that allow for the different strategies to be simulated.
        Once a Patient object is instantiated with certain initial conditions, it can currently be simulated with a
        single strategy using the simulate() function. A graph of the various cell populations will automatically be
        projected and saved to the filename specified above.
    """

    def __init__(self, x, weeks, g0, Sa, T, strategy_num, patient_name, graph_output_filename, paired):
        """ Parameters of the Patient class. """
        self.x = x
        self.original_x = x
        self.weeks = weeks
        self.days = weeks * 7
        self.completed_days = t_interval
        self.g0 = g0
        self.Sa = Sa
        self.T = T
        self.strategy_num = strategy_num
        self.patient_name = patient_name
        self.steps_between_time_points = 45
        self.graph_output_filename = graph_output_filename
        self.death_day = 0
        self.set_death_day = False

        if self.strategy_num == 0:
            ''' Check to see whether the R1 cells are the dominating cell population and adjust the drug levels 
                accordingly. 
            '''
            if self.x[1] / sum(list(self.x)) <= 0.5:
                self.d = np.transpose(np.array([1.0, 0.0]))
            else:
                self.d = np.transpose(np.array([1.0, 0.0]))
        else:
            ''' Place to add any other drug initializations for the other strategies. '''
            self.d = np.transpose(np.array([0.5, 0.5]))
        self.surviving = True
        self.flipped = 0  # to check if the drug has alrdy been switched
        ''' All cell populations for this Patient object are stored in different lists to make plotting easier. I.E. 
            the S cell population vs. time data list is in self.S. 
        '''
        self.tot = [sum(list(x))]
        self.S = [x[0]]
        self.R1 = [x[1]]
        self.R2 = [x[2]]
        self.R12 = [x[3]]
        self.dList = []
        self.paired = paired  # 1 is for the first of two patients, 2 is for the second of two patients, -1 is for stand-alone patient (this indicates which subplot the matplotlib function should plot in.

    def list_of_lists_to_csv(self, to_csv: list, filename: str) -> None:
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(to_csv)

    def delete_fractional(self) -> None:
        for i in range(len(self.x)):
            if self.x[i] < 1:
                self.x[i] = 0

    def update_cell_type_lists(self) -> None:
        self.S.append(self.x[0])
        self.R1.append(self.x[1])
        self.R2.append(self.x[2])
        self.R12.append(self.x[3])
        self.tot.append(sum(list(self.x)))
        self.dList.append(self.d)

    def check_surviving(self) -> None:
        if sum(list(self.x)) >= death_threshold: self.surviving = False

    def predict_cell_population(self, true_t_interval=float(t_interval)) -> np.array:
        """ This function predicts the cell population at the next t_interval using the equation x(t) = x(0)*e^(B*true_t_inverval).
            Here, the B matrix is defined akin to the B matrix on page 3 of the supporting information of the PNAS paper.
            TODO: Account for single-cell boundary crossings. Find a way to have the argument "true_d=self.d" in the function header so that you don't have to change self.d everytime.
        """
        A = np.subtract((np.multiply(np.add(np.identity(self.T.shape[0]), self.T), self.g0)),
                        (diag(np.matmul(self.Sa, self.d))))
        B = A  # U(A, self.x)
        exponentiated = expm(np.multiply(B, true_t_interval))  # computes e^(Bt)
        predicted = np.matmul(exponentiated, self.x)  # computes e^(Bt) * x(0) #check whether you need to transpose this
        return predicted

    def predict_cell_population_with_intermediate_steps(self, num_steps=-1) -> None:
        if num_steps == -1:
            num_steps = self.steps_between_time_points
        for i in range(num_steps):
            if self.surviving:
                self.x = self.predict_cell_population(true_t_interval=t_interval / num_steps)
                self.delete_fractional()
                self.check_surviving()
            elif self.set_death_day is False:
                self.death_day = self.completed_days - t_interval + (i + 1) * (t_interval / num_steps)
                self.set_death_day = True

    def best_drug_combo_minimize_total_cellpop(self) -> np.array:
        """ TODO: d1 and d2 might have to sum to 1, check on this later. """
        original_d = self.d
        min_total_cellpop = total_cells
        best_drug_combo = np.array([0.0, 0.0])
        D1 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
        D2 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
        ''' Determine which drug dosage levels minimize the predicted total cell population '''
        for d1 in D1:
            for d2 in D2:
                if d1 + d2 > 1: continue
                self.d = np.array([d1, d2])
                pred_total_cellpop = sum(list(self.predict_cell_population()))
                if pred_total_cellpop < min_total_cellpop:
                    min_total_cellpop = pred_total_cellpop
                    best_drug_combo = np.array([d1, d2])
        self.d = original_d
        return best_drug_combo

    def best_drug_combo_minimize_R12(self) -> np.array:
        original_d = self.d
        min_total_R12 = total_cells
        best_drug_combo = np.array([0.0, 0.0])
        D1 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
        D2 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
        ''' Determine which drug dosage combination minimizes the predicted R12 cell population '''
        for d1 in D1:
            for d2 in D2:
                if d1 + d2 > 1:
                    continue
                self.d = np.array([d1, d2])
                pred_R12 = list(self.predict_cell_population())[3]
                if pred_R12 < min_total_R12:
                    min_total_R12 = pred_R12
                    best_drug_combo = np.array([d1, d2])
        self.d = original_d
        return best_drug_combo

    def t_inc(self) -> int:
        """ Calculates the predicted number of days until incurability. """
        for t in range(1, self.days, self.days // 20):  # remove "total_sim_days//20" in final version
            pred = self.predict_cell_population(true_t_interval=t)
            if pred[3] >= 1:
                return t
        return self.days

    def t_X(self, index: int) -> int:
        ''' Calculates the predicted number of days until cell population X (either 0, 1, 2, 3 for each of the respective
            components in the vector self.x) goes above the death threshold (1e13). '''
        for t in range(1, self.days, self.days // 20):  # remove "total_sim_days//20" in final version
            pred = self.predict_cell_population(true_t_interval=t)
            if pred[index] >= death_threshold:
                return t
        return self.days

    def simulate(self, verbose=True) -> None:
        if verbose:
            print("Started Simulating Patient " + self.patient_name + " with Strategy %s for %i weeks" % (
                str(round(self.strategy_num, 1)), self.weeks))
        for t_interval_count in range(1, self.days // t_interval):
            ''' Reevaluates strategy0 at every t_interval '''
            # print(self.x)
            if self.surviving:
                ''' Each strategy should simply update self.x to the next time point. '''
                if self.strategy_num == 0:
                    self.strategy0()
                elif self.strategy_num == 1:
                    self.strategy1()
                elif self.strategy_num == 2:
                    self.strategy2_1()
                elif self.strategy_num == 2.1:
                    self.strategy2_1()
                elif self.strategy_num == 2.2:
                    self.strategy2_2()
                elif self.strategy_num == 3:
                    self.strategy3()
                elif self.strategy_num == 4:
                    self.strategy4()
                elif self.strategy_num == 5:
                    self.strategy5()
                else:
                    self.strategy0()
            elif self.set_death_day is False:
                self.death_day = self.completed_days - t_interval
                self.set_death_day = True
            self.check_surviving()
            self.delete_fractional()
            self.completed_days += t_interval
            self.update_cell_type_lists()
            if self.strategy_num == 5:
                self.plot_cell_populations(save_fig=False)
        if verbose:
            print("Finished Simulating Patient " + self.patient_name + " with Strategy %s for %i weeks" % (
                str(round(self.strategy_num, 1)), self.weeks))

    def strategy0(self) -> None:
        """ This function will update a patient with strategy0 to the next time point similar to all the other
        strategyN() functions. """
        current_cellpop = sum(list(self.x))
        original_cellpop = sum(list(self.original_x))
        ''' This if statement addresses progression. '''
        if self.flipped == 0 and (current_cellpop > 1e9 or current_cellpop > 2 * original_cellpop):
            self.flipped += 1
            if self.d[0] == 1.0:
                self.d = np.transpose(np.array([0.0, 1.0]))  # switch the drug being used
            else:
                self.d = np.transpose(np.array([1.0, 0.0]))
        # precision: this number tells you what fraction of the t_interval it is simulating -- can replace this code
        # block with a function
        # self.x = self.predict_cell_population()
        self.predict_cell_population_with_intermediate_steps()
        '''for i in range(self.steps_between_time_points):
            self.x = self.predict_cell_population(t_interval/self.steps_between_time_points)
            self.delete_fractional()'''

    def strategy1(self) -> None:
        best_drug_combo = self.best_drug_combo_minimize_total_cellpop()
        self.d = best_drug_combo
        # self.x = self.predict_cell_population()
        self.predict_cell_population_with_intermediate_steps()

    def strategy2(self, threshold: float) -> None:
        current_cell_pop = sum(list(self.x))
        ''' If there is an immediate threat of mortality revert to strategy 1. '''
        if current_cell_pop > threshold:
            self.strategy1()
        else:
            best_drug_combo = self.best_drug_combo_minimize_R12()
            self.d = best_drug_combo
            # self.x = self.predict_cell_population()
            self.predict_cell_population_with_intermediate_steps()

    def strategy2_1(self) -> None:
        self.strategy2(1e9)

    def strategy2_2(self) -> None:
        self.strategy2(1e11)

    def is_R12_curable(self) -> bool:
        return (self.g0 - self.Sa[3, 0] - self.Sa[3, 1]) <= 0

    def strategy3(self) -> None:
        pred_R12 = list(self.predict_cell_population())[3]
        current_R12 = list(self.x)[3]
        ''' If there is no immediate threat of R12 cells, revert to strategy1. '''
        if pred_R12 < 1 or (current_R12 >= 1 and self.is_R12_curable() is False):
            self.strategy1()
        else:
            best_drug_combo = self.best_drug_combo_minimize_R12()
            self.d = best_drug_combo
            # self.x = self.predict_cell_population()
            self.predict_cell_population_with_intermediate_steps(num_steps=5)

    def strategy4(self) -> None:
        current_R12 = list(self.x)[3]
        if current_R12 < 1 or self.is_R12_curable():
            ''' case: the current R12 population is less than 1 or the R12 cells are curable'''
            max_min_all = 0
            constraint = 0
            best_drug_combo = np.array([0.0, 0.0])
            D1 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
            D2 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
            for d1 in D1:
                for d2 in D2:
                    if d1 + d2 > 1: continue
                    self.d = np.array([d1, d2])
                    t_incurable = self.t_inc()
                    t_S = self.t_X(0)
                    t_R1 = self.t_X(1)
                    t_R2 = self.t_X(2)
                    t_R12 = self.t_X(3)
                    curr_min = min([t_incurable, t_S, t_R1, t_R2, t_R12])
                    if curr_min > max_min_all:
                        max_min_all = curr_min
                        constraint = min([t_S, t_R1, t_R2, t_R12])
                        best_drug_combo = self.d
            if constraint > 45:
                ''' case: min(t_S, t_R1, t_R2, t_R12) > 45'''
                self.d = best_drug_combo
                # self.x = self.predict_cell_population()
                self.predict_cell_population_with_intermediate_steps(num_steps=5)
            else:
                ''' case: min(t_S, t_R1, t_R2, t_R12) <= 45'''
                max_min_all_but_t_inc = 0
                best_drug_combo = np.array([0.0, 0.0])
                D1 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
                D2 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
                for d1 in D1:
                    for d2 in D2:
                        if d1 + d2 > 1:
                            continue
                        self.d = np.array([d1, d2])
                        t_S = self.t_X(0)
                        t_R1 = self.t_X(1)
                        t_R2 = self.t_X(2)
                        t_R12 = self.t_X(3)
                        curr_min = min([t_S, t_R1, t_R2, t_R12])
                        if curr_min > max_min_all_but_t_inc:
                            max_min_all_but_t_inc = curr_min
                            best_drug_combo = self.d
                self.d = best_drug_combo
                # self.x = self.predict_cell_population()
                self.predict_cell_population_with_intermediate_steps(num_steps=5)
        else:
            '''case: current_R12 >= 1 and self.is_R12_curable is False'''
            max_min_all_but_t_inc = 0
            best_drug_combo = np.array([0.0, 0.0])
            D1 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
            D2 = np.arange(0, 1 + drug_optimization_interval, drug_optimization_interval)
            for d1 in D1:
                for d2 in D2:
                    if d1 + d2 > 1: continue
                    self.d = np.array([d1, d2])
                    t_S = self.t_X(0)
                    t_R1 = self.t_X(1)
                    t_R2 = self.t_X(2)
                    t_R12 = self.t_X(3)
                    curr_min = min([t_S, t_R1, t_R2, t_R12])
                    if curr_min > max_min_all_but_t_inc:
                        max_min_all_but_t_inc = curr_min
                        best_drug_combo = self.d
            self.d = best_drug_combo
            # self.x = self.predict_cell_population()
            self.predict_cell_population_with_intermediate_steps(num_steps=5)

    def replace_fractional_with_poisson(self, x):
        for i in range(len(x)):
            if x[i] < 1:
                x[i] = 1 - math.exp(-x[i])
        return x

    def strategy5(self) -> None:
        """ This is essentially the create-your-own drug combination strategy. """
        global fig
        global ax_arr

        index = -1
        if self.paired == -1:
            index = 0
        elif self.paired == 1:
            index = 0
        elif self.paired == 2:
            index = 1

        self.d = np.transpose(np.array([1.0, 0.0]))
        pre_poisson_drug_one_cell_pop = self.predict_cell_population()
        drug_one_cell_pop = self.replace_fractional_with_poisson(pre_poisson_drug_one_cell_pop)
        self.d = np.transpose(np.array([0.0, 1.0]))
        pre_poisson_drug_two_cell_pop = self.predict_cell_population()
        drug_two_cell_pop = self.replace_fractional_with_poisson(pre_poisson_drug_two_cell_pop)
        self.d = np.transpose(np.array([0.5, 0.5]))
        pre_poisson_drug_mixed_cell_pop = self.predict_cell_population()
        drug_mixed_cell_pop = self.replace_fractional_with_poisson(pre_poisson_drug_mixed_cell_pop)

        ''' weeks_arr holds the x-values that you want to additionally plot'''
        current_week = (self.completed_days - t_interval) / 7
        next_week = self.completed_days / 7
        weeks_arr = [current_week, next_week]

        drug_one_S_cell_arr = [self.S[-1], drug_one_cell_pop[0]]
        drug_one_R1_cell_arr = [self.R1[-1], drug_one_cell_pop[1]]
        drug_one_R2_cell_arr = [self.R2[-1], drug_one_cell_pop[2]]
        drug_one_R12_cell_arr = [self.R12[-1], drug_one_cell_pop[3]]
        ax_arr[index].plot(weeks_arr, drug_one_S_cell_arr, c="lightgrey", label="Drug 1 Prediction")
        ax_arr[index].plot(weeks_arr, drug_one_R1_cell_arr, c="lightgrey")
        ax_arr[index].plot(weeks_arr, drug_one_R2_cell_arr, c="lightgrey")
        ax_arr[index].plot(weeks_arr, drug_one_R12_cell_arr, c="lightgrey")

        drug_two_S_cell_arr = [self.S[-1], drug_two_cell_pop[0]]
        drug_two_R1_cell_arr = [self.R1[-1], drug_two_cell_pop[1]]
        drug_two_R2_cell_arr = [self.R2[-1], drug_two_cell_pop[2]]
        drug_two_R12_cell_arr = [self.R12[-1], drug_two_cell_pop[3]]
        ax_arr[index].plot(weeks_arr, drug_two_S_cell_arr, c="black", label="Drug 2 Prediction")
        ax_arr[index].plot(weeks_arr, drug_two_R1_cell_arr, c="black")
        ax_arr[index].plot(weeks_arr, drug_two_R2_cell_arr, c="black")
        ax_arr[index].plot(weeks_arr, drug_two_R12_cell_arr, c="black")

        drug_mixed_S_cell_arr = [self.S[-1], drug_mixed_cell_pop[0]]
        drug_mixed_R1_cell_arr = [self.R1[-1], drug_mixed_cell_pop[1]]
        drug_mixed_R2_cell_arr = [self.R2[-1], drug_mixed_cell_pop[2]]
        drug_mixed_R12_cell_arr = [self.R12[-1], drug_mixed_cell_pop[3]]
        ax_arr[index].plot(weeks_arr, drug_mixed_S_cell_arr, c="grey", label="Mixed Dosage Prediction")
        ax_arr[index].plot(weeks_arr, drug_mixed_R1_cell_arr, c="grey")
        ax_arr[index].plot(weeks_arr, drug_mixed_R2_cell_arr, c="grey")
        ax_arr[index].plot(weeks_arr, drug_mixed_R12_cell_arr, c="grey")

        self.plot_cell_populations(save_fig=False)

        drug_num = float(input("Enter \"1\" or \"2\" or \"1.5\" to choose which drug to continue with: "))

        if drug_num == 1:
            self.d = np.transpose(np.array([1.0, 0.0]))
        elif drug_num == 2:
            self.d = np.transpose(np.array([0.0, 1.0]))
        elif drug_num == 1.5:
            self.d = np.transpose(np.array([0.5, 0.5]))

        self.x = self.predict_cell_population()

    def convert_and_plot_drug_list(self, weeks_arr: list) -> None:
        """
        drug one plotted in
        """
        global fig
        global ax_arr
        index = -1
        if self.paired == -1:
            index = 0
        elif self.paired == 1:
            index = 0
        elif self.paired == 2:
            index = 1

        min_S = min(self.S)
        min_R1 = min(self.R1)
        min_R2 = min(self.R2)
        min_val = min(min_S, min(min_R1, min_R2))
        min_val = max(min_val, 1)

        drug_one = [(weeks_arr[i], min_val / 2 + 10) for i in range(len(weeks_arr) - 1) if
                    self.dList[i].tolist() == [1, 0]]
        drug_two = [(weeks_arr[i], min_val / 2 + 10) for i in range(len(weeks_arr) - 1) if
                    self.dList[i].tolist() == [0, 1]]
        drug_mixed = [(weeks_arr[i], min_val / 2 + 10) for i in range(len(weeks_arr) - 1) if
                      self.dList[i].tolist() != [1, 0] and self.dList[i].tolist() != [0, 1]]

        if len(drug_one) != 0:
            drug_one_x, drug_one_y = zip(*drug_one)
            ax_arr[index].scatter(drug_one_x, drug_one_y, c="lightgrey", label="Drug 1", marker='>')
        if len(drug_two) != 0:
            drug_two_x, drug_two_y = zip(*drug_two)
            ax_arr[index].scatter(drug_two_x, drug_two_y, c="black", label="Drug 2", marker='>')
        if len(drug_mixed) != 0:
            drug_mixed_x, drug_mixed_y = zip(*drug_mixed)
            ax_arr[index].scatter(drug_mixed_x, drug_mixed_y, c="grey", label="Mixed Dosage", marker='>')

    def convert_float_to_string(self, flt, num_sigfigs=5) -> str:
        return '{:g}'.format(float('{:.{p}g}'.format(flt, p=num_sigfigs)))

    def convert_patient_parameters_to_string(self) -> str:
        """ "R1: " + self.convert_float_to_string(self.x[1]) + " R2: " + self.convert_float_to_string(self.x[2]) + "
        R12: " + self.convert_float_to_string(self.x[3]) + """
        return "Patient Parameters: " + "\n Growth Rate: " + self.convert_float_to_string(
            self.g0) + "\n Ss_1: " + self.convert_float_to_string(self.Sa[0][0]) + \
               "\n Ss_2: " + self.convert_float_to_string(self.Sa[0][1]) + "\n Sr1_1: " + self.convert_float_to_string(
            self.Sa[1][0]) + "\n Sr1_2: " + self.convert_float_to_string(
            self.Sa[1][1]) + "\n Sr2_1: " + self.convert_float_to_string(
            self.Sa[2][0]) + "\n Sr2_2: " + self.convert_float_to_string(self.Sa[2][1]) \
               + "\n Sr12_1: " + self.convert_float_to_string(
            self.Sa[3][0]) + "\n Sr12_2: " + self.convert_float_to_string(
            self.Sa[3][1]) + "\n Ts_r1: " + self.convert_float_to_string(
            self.T[1][0]) + "\n Ts_r2: " + self.convert_float_to_string(self.T[2][0])

    def plot_cell_populations(self, convert_to_spreadsheet=False, show_plot=True, save_fig=True) -> None:
        """ Plots the cell populations of this patient over the time of the therapy. """
        global fig
        global ax_arr
        index = -1
        if self.paired == -1:
            index = 0
        elif self.paired == 1:
            index = 0
        elif self.paired == 2:
            index = 1
        if self.paired == 1:
            fig, ax_arr = plt.subplots(1, 2, figsize=(16, 6))
        elif self.paired == -1:
            fig, ax_arr = plt.subplots(1, 1, figsize=(8, 6))
            ax_arr = [ax_arr]

        weeks_arr = [(i * t_interval) / 7 for i in range(0,
                                                         self.completed_days // t_interval)]  # changed from
        # self.weeks_arr; can remove t_interval to make graph more compressed.
        one_arr = [1 for _ in range(len(weeks_arr))]
        ax_arr[index].plot(weeks_arr, self.S, "c", label="S")
        ax_arr[index].plot(weeks_arr, self.R1, "g", label="R1")
        ax_arr[index].plot(weeks_arr, self.R2, "y", label="R2")
        ax_arr[index].plot(weeks_arr, self.R12, "r", label="R12")
        ax_arr[index].axvline(x=self.death_day / 7, c="m",
                              label="Patient Death: Day %s" % str(round(self.death_day, 2)))
        ax_arr[index].scatter(weeks_arr, one_arr, c="orange", label="Single Cell Boundary", marker="_")
        # plt.plot(weeks_arr, self.tot, "b", label="Total Tumor Cell Population") #for now dont plot this
        self.convert_and_plot_drug_list(weeks_arr)
        ax_arr[index].set_yscale('log')
        ax_arr[index].legend()
        ax_arr[index].set_xlabel("Time (Weeks) \n\n\n .")
        ax_arr[index].set_ylabel("Number of Cells")
        ax_arr[index].set_title(
            self.patient_name + " - Strategy " + str(round(self.strategy_num, 1)) + ": Number of Cells vs. Time")
        caption = self.convert_patient_parameters_to_string()
        plt.figtext(0.5, -0.3, caption, wrap=True, horizontalalignment='center', fontsize=10, bbox={'facecolor': 'grey',
                                                                                                    'alpha': 0.3,
                                                                                                    'pad': 2})
        if show_plot:
            plt.show()
        if len(self.graph_output_filename) != 0 and save_fig:
            plt.savefig(self.graph_output_filename, bbox_inches='tight')
            plt.clf()
            plt.close(fig)
        if convert_to_spreadsheet:
            to_csv = [["Days", "S", "R1", "R2", "R12"]]
            # reconverted to days for csv file
            for i in range(0, self.completed_days // t_interval):
                to_csv.append([(i * t_interval), self.S[i], self.R1[i], self.R2[i], self.R12[i]])
            self.list_of_lists_to_csv(to_csv, self.patient_name + "_data.csv")
            print("Successfully created csv file of " + self.patient_name + " data")


class Cohort:
    """ The Cohort class allows the user to conduct batch simulations. """

    def __init__(self, g0_params, R1_params, R2_params, R12_params, Ss_1_params, Ss_2_params, Sr1_1_params,
                 Sr1_2_params, Sr2_1_params, Sr2_2_params, Sr12_1_params, Sr12_2_params, Ts_r1_params, Ts_r2_params,
                 cohort_name, strategy_num, csv_filename, use_csv, weeks):
        self.weeks = weeks
        self.cohort_name = cohort_name
        os.makedirs(self.cohort_name)
        self.working_directory = "./" + self.cohort_name + "/"
        self.strategy_num = strategy_num
        self.csv_filename = csv_filename
        self.csv_cohort_parameter_data = None
        self.csv_num_patients = None

        if len(self.csv_filename) != 0:
            self.csv_cohort_parameter_data = pd.read_csv(self.csv_filename)
            self.csv_num_patients = self.csv_cohort_parameter_data.shape[0]

        self.use_csv = use_csv
        self.g0_params = g0_params
        self.R1_params = R1_params
        self.R2_params = R2_params
        self.R12_params = R12_params
        self.Ss_1_params = Ss_1_params
        self.Ss_2_params = Ss_2_params
        self.Sr1_1_params = Sr1_1_params
        self.Sr1_2_params = Sr1_2_params
        self.Sr2_1_params = Sr2_1_params
        self.Sr2_2_params = Sr2_2_params
        self.Sr12_1_params = Sr12_1_params
        self.Sr12_2_params = Sr12_2_params
        self.Ts_r1_params = Ts_r1_params
        self.Ts_r2_params = Ts_r2_params
        self.batch_num_patients = len(g0_params) * len(R1_params) * len(R2_params) * len(R12_params) * len(
            Ss_1_params) * len(Ss_2_params) * len(Sr1_1_params) * len(Sr1_2_params) * len(Sr2_1_params) * len(
            Sr2_2_params) * len(Sr12_1_params) * len(Sr12_2_params) * len(Ts_r1_params) * len(Ts_r2_params)

        if self.use_csv:
            print("Number of Patients in Cohort: %i" % self.csv_num_patients)
        else:
            print("Number of Patients in Cohort: %i" % self.batch_num_patients)

    def simulate_cohort(self) -> None:
        print("Started Simulating Cohort")
        start = time.time()
        if len(self.csv_filename) != 0 and self.use_csv:
            for i in range(self.csv_num_patients):
                if self.strategy_num == 0:
                    curr_patient = self.create_patient_from_csv_row(i, 0, -1)
                    curr_patient.simulate(verbose=False)
                    curr_patient.plot_cell_populations(show_plot=False, save_fig=True)
                else:
                    '''simulates patient with strategy 0'''
                    default_strategy = 0
                    curr_patient_one = self.create_patient_from_csv_row(i, default_strategy, 1)
                    print(curr_patient_one.patient_name)
                    curr_patient_one.simulate(verbose=False)
                    curr_patient_one.plot_cell_populations(show_plot=False, save_fig=False)
                    print("Patient with Strategy 0 Death Day: %f" % curr_patient_one.death_day)
                    # print(curr_patient_one.dList)

                    '''simulates patient with strategy X (not zero)'''
                    curr_patient_two = self.create_patient_from_csv_row(i, self.strategy_num, 2)
                    curr_patient_two.simulate(verbose=False)
                    curr_patient_two.plot_cell_populations(show_plot=False, save_fig=True)
                    print("Patient with Strategy 2.2 Death Day: %f" % curr_patient_two.death_day)
                    # print(curr_patient_two.dList)

                    death_default = curr_patient_one.death_day
                    death_selected = curr_patient_two.death_day

                    selected_showed_improvement = False

                    if death_selected - death_default > 60 and death_selected / death_default > 1.25:
                        selected_showed_improvement = True

                    print("The Selected Strategy Showed Improvement: %r" % selected_showed_improvement)
                    print()

                curr_time = time.time()
                curr_elapsed_time = curr_time - start
                estimated_time_remaining = int(
                    round(curr_elapsed_time * (1 / ((i + 1) / self.csv_num_patients)) - curr_elapsed_time))
                # print('\rPatient %i out of %i [%d%%] -- Estimated time left: %i sec' % (
                #    i + 1, self.csv_num_patients,
                #    100 * (i + 1) / self.csv_num_patients, estimated_time_remaining), end="")
            end = time.time()
            print("\nFinished Simulating Cohort")
            print("Average Simulation Time Per Patient: %f sec" % ((end - start) / self.csv_num_patients))
        else:
            S = -1
            count = 0
            for R1 in self.R1_params:
                for R2 in self.R2_params:
                    for R12 in self.R12_params:
                        S = total_cells - R1 - R2 - R12
                        x0 = np.transpose(np.array([S, R1, R2, R12]))
                        for Ss_1 in self.Ss_1_params:
                            for Ss_2 in self.Ss_2_params:
                                for Sr1_1 in self.Sr1_1_params:
                                    for Sr1_2 in self.Sr1_2_params:
                                        for Sr2_1 in self.Sr2_1_params:
                                            for Sr2_2 in self.Sr2_2_params:
                                                for Sr12_1 in self.Sr12_1_params:
                                                    for Sr12_2 in self.Sr12_2_params:
                                                        Sa = np.array([[Ss_1, Ss_2],
                                                                       [Sr1_1, Sr1_2],
                                                                       [Sr2_1, Sr2_2],
                                                                       [Sr12_1, Sr12_2]])
                                                        for Ts_r1 in self.Ts_r1_params:
                                                            for Ts_r2 in self.Ts_r2_params:
                                                                for g0 in self.g0_params:
                                                                    T = np.array([[0, 0, 0, 0],
                                                                                  [Ts_r1, 0, 0, 0],
                                                                                  [Ts_r2, 0, 0, 0],
                                                                                  [0, Ts_r2, Ts_r1, 0]])
                                                                    weeks = self.weeks
                                                                    graph_output_filename = self.working_directory + str(
                                                                        count) + ".png"
                                                                    patient_name = "Patient %i" % count
                                                                    curr_patient_default = Patient(x0, weeks, g0, Sa, T,
                                                                                                   0,
                                                                                                   patient_name,
                                                                                                   graph_output_filename,
                                                                                                   1)
                                                                    curr_patient_default.simulate(verbose=False)
                                                                    curr_patient_default.plot_cell_populations(
                                                                        show_plot=False, save_fig=False)

                                                                    curr_patient_actual = Patient(x0, weeks, g0, Sa, T,
                                                                                                  self.strategy_num,
                                                                                                  patient_name,
                                                                                                  graph_output_filename,
                                                                                                  2)
                                                                    curr_patient_actual.simulate(verbose=False)
                                                                    curr_patient_actual.plot_cell_populations(
                                                                        show_plot=False, save_fig=True)

                                                                    curr_time = time.time()
                                                                    curr_elapsed_time = curr_time - start
                                                                    estimated_time_remaining = int(round(
                                                                        curr_elapsed_time * (1 / ((
                                                                                                          count + 1) / self.batch_num_patients)) - curr_elapsed_time))
                                                                    print(
                                                                        '\rPatient %i out of %i [%d%%] -- Estimated time left: %i sec' % (
                                                                            count + 1, self.batch_num_patients,
                                                                            100 * (count + 1) / self.batch_num_patients,
                                                                            estimated_time_remaining), end="")
                                                                    count += 1
            end = time.time()
            print("\nFinished Simulating Cohort")
            print("Average Simulation Time Per Patient: %f sec" % ((end - start) / self.batch_num_patients))

    def create_patient_from_csv_row(self, row_number: int, strategy_num: float, paired: int) -> Patient:
        patient_data = self.csv_cohort_parameter_data.iloc[row_number, :]
        patient_id = str(int(patient_data["paramID"]))
        R1 = patient_data["R1"]
        R2 = patient_data["R2"]
        R12 = patient_data["R12"]
        S = total_cells - R1 - R2 - R12
        g0 = patient_data["GrowthRate"]

        ''' Sa_b is the sensitivity of a to drug b
            Ta_b is the transition rate of cell type a to b 
        '''
        Ss_1 = patient_data["Ss_1"]
        Ss_2 = patient_data["Ss_2"]
        Sr1_1 = patient_data["Sr1_1"]
        Sr1_2 = patient_data["Sr1_2"]
        Sr2_1 = patient_data["Sr2_1"]
        Sr2_2 = patient_data["Sr2_2"]
        Sr12_1 = patient_data["Sr12_1"]
        Sr12_2 = patient_data["Sr12_2"]
        Ts_r1 = patient_data["Ts_r1"]
        Ts_r2 = patient_data["Ts_r2"]
        Tr1_r12 = Ts_r2  # 4e-7
        Tr2_r12 = Ts_r1  # 4e-9
        weeks = self.weeks  # 255; when using strategy4 make sure to keep the number of weeks low (~20) because of
        # high run-time
        graph_output_filename = self.working_directory + patient_id + ".png"

        x0 = np.transpose(np.array([S, R1, R2, R12]))
        T = np.array([[0, 0, 0, 0],
                      [Ts_r1, 0, 0, 0],
                      [Ts_r2, 0, 0, 0],
                      [0, Tr1_r12, Tr2_r12, 0]])
        Sa = np.array([[Ss_1, Ss_2],
                       [Sr1_1, Sr1_2],
                       [Sr2_1, Sr2_2],
                       [Sr12_1, Sr12_2]])

        patient_name = "Patient " + str(patient_id)
        test_patient = Patient(x0, weeks, g0, Sa, T, strategy_num, patient_name, graph_output_filename, paired)
        return test_patient


def main():
    """ Initialized the main GUI interface
        TODO: Fix all of the different input options that the user has. Should all of the transition matrix values and add multiple patients option.
        be changeable to the user? Should all sensitivity matrix values be changeable to the user? etc.
    """

    def clicked_sim() -> None:
        cohort_name = str(cohort_name_entry.get())
        strategy_num = float(strategy_number_entry.get())
        csv_filename = str(csv_filename_entry.get())
        use_csv = (str(use_csv_entry.get()) == "Y")
        weeks = int(weeks_entry.get())

        g0_inputs = [float(g0_LB.get()), float(g0_UB.get()), int(g0_n_vals.get()), str(g0_scale.get()),
                     str(g0_manual.get())]
        R1_inputs = [float(R1_LB.get()), float(R1_UB.get()), int(R1_n_vals.get()), str(R1_scale.get()),
                     str(R1_manual.get())]
        R2_inputs = [float(R2_LB.get()), float(R2_UB.get()), int(R2_n_vals.get()), str(R2_scale.get()),
                     str(R2_manual.get())]
        R12_inputs = [float(R12_LB.get()), float(R12_UB.get()), int(R12_n_vals.get()), str(R12_scale.get()),
                      str(R12_manual.get())]
        Ss_1_inputs = [float(Ss_1_LB.get()), float(Ss_1_UB.get()), int(Ss_1_n_vals.get()), str(Ss_1_scale.get()),
                       str(Ss_1_manual.get())]
        Ss_2_inputs = [float(Ss_2_LB.get()), float(Ss_2_UB.get()), int(Ss_2_n_vals.get()), str(Ss_2_scale.get()),
                       str(Ss_2_manual.get())]
        Sr1_1_inputs = [float(Sr1_1_LB.get()), float(Sr1_1_UB.get()), int(Sr1_1_n_vals.get()), str(Sr1_1_scale.get()),
                        str(Sr1_1_manual.get())]
        Sr1_2_inputs = [float(Sr1_2_LB.get()), float(Sr1_2_UB.get()), int(Sr1_2_n_vals.get()), str(Sr1_2_scale.get()),
                        str(Sr1_2_manual.get())]
        Sr2_1_inputs = [float(Sr2_1_LB.get()), float(Sr2_1_UB.get()), int(Sr2_1_n_vals.get()), str(Sr2_1_scale.get()),
                        str(Sr2_1_manual.get())]
        Sr2_2_inputs = [float(Sr2_2_LB.get()), float(Sr2_2_UB.get()), int(Sr2_2_n_vals.get()), str(Sr2_2_scale.get()),
                        str(Sr2_2_manual.get())]
        Sr12_1_inputs = [float(Sr12_1_LB.get()), float(Sr12_1_UB.get()), int(Sr12_1_n_vals.get()),
                         str(Sr12_1_scale.get()), str(Sr12_1_manual.get())]
        Sr12_2_inputs = [float(Sr12_2_LB.get()), float(Sr12_2_UB.get()), int(Sr12_2_n_vals.get()),
                         str(Sr12_2_scale.get()), str(Sr12_2_manual.get())]
        Ts_r1_inputs = [float(Ts_r1_LB.get()), float(Ts_r1_UB.get()), int(Ts_r1_n_vals.get()), str(Ts_r1_scale.get()),
                        str(Ts_r1_manual.get())]
        Ts_r2_inputs = [float(Ts_r2_LB.get()), float(Ts_r2_UB.get()), int(Ts_r2_n_vals.get()), str(Ts_r2_scale.get()),
                        str(Ts_r2_manual.get())]

        cohort_inputs = [g0_inputs, R1_inputs, R2_inputs, R12_inputs, Ss_1_inputs, Ss_2_inputs, Sr1_1_inputs,
                         Sr1_2_inputs, Sr2_1_inputs, Sr2_2_inputs, Sr12_1_inputs, Sr12_2_inputs, Ts_r1_inputs,
                         Ts_r2_inputs]

        cohort_params = [[], [], [], [], [], [], [],
                         [], [], [], [], [], [],
                         []]

        for i in range(len(cohort_inputs)):
            if cohort_inputs[i][2] == 1 and len(cohort_inputs[i][-1]) == 0:
                cohort_params[i] = [cohort_inputs[i][0]]
            elif len(cohort_inputs[i][-1]) == 0:
                scale_function = ''
                if cohort_inputs[i][3] == "arithmetic":
                    scale_function = np.linspace
                elif cohort_inputs[i][3] == "logarithmic":
                    scale_function = np.geomspace
                small_delta = 1e-50
                cohort_params[i] = scale_function(cohort_inputs[i][0] + small_delta, cohort_inputs[i][1],
                                                  num=cohort_inputs[i][2]).tolist()
                for j in range(len(cohort_params[i])):
                    if cohort_params[i][j] == small_delta:
                        cohort_params[i][j] = 0
            else:
                cohort_params[i] = [float(param.strip()) for param in cohort_inputs[i][-1].split(",")]

        g0_params, R1_params, R2_params, R12_params, Ss_1_params, Ss_2_params, Sr1_1_params, Sr1_2_params, Sr2_1_params, Sr2_2_params, Sr12_1_params, Sr12_2_params, Ts_r1_params, Ts_r2_params = cohort_params

        ''' Print out the params and confirm whether or not the user wants to proceede through console input. '''
        if len(csv_filename) == 0 or use_csv is False:
            print("------Cohort Parameters------")
            print("Growth Rate: " + str(g0_params))
            print("R1: " + str(R1_params))
            print("R2: " + str(R2_params))
            print("R12: " + str(R12_params))
            print("Ss_1: " + str(Ss_1_params))
            print("Ss_2: " + str(Ss_2_params))
            print("Sr1_1: " + str(Sr1_1_params))
            print("Sr1_2: " + str(Sr1_2_params))
            print("Sr2_1: " + str(Sr2_1_params))
            print("Sr2_2: " + str(Sr2_2_params))
            print("Sr12_1: " + str(Sr12_1_params))
            print("Sr12_2: " + str(Sr12_2_params))
            print("Ts_r1: " + str(Ts_r1_params))
            print("Ts_r2: " + str(Ts_r2_params))

            desired = input("Are these the desired parameters? Y/N")
            if desired == "N":
                print("Please reenter the desired parameters into the GUI.")
                return
            print("\n")

        test_cohort = Cohort(g0_params, R1_params, R2_params, R12_params, Ss_1_params, Ss_2_params, Sr1_1_params,
                             Sr1_2_params, Sr2_1_params, Sr2_2_params, Sr12_1_params, Sr12_2_params, Ts_r1_params,
                             Ts_r2_params, cohort_name, strategy_num, csv_filename, use_csv, weeks)
        test_cohort.simulate_cohort()

    ''' User Interface Code Below '''

    window = Tk()
    window.title("DPM GUI")
    window.geometry("1250x750")
    title_label = Label(window, text="Cancer Therapy Simulation GUI", font=('Helvetica', 18, 'bold'))
    title_label.grid(column=3, row=0)

    ''' BATCH SIMULATION MODE '''
    batch_label = Label(window, text="Batch Simulation Mode", font=('Helvetica', 16, 'bold'))
    batch_label.grid(column=3, row=1)

    parameters_label = Label(window, text="Parameter", font=('Helvetica', 14, 'bold'))
    parameters_label.grid(column=0, row=2)
    lower_bound_label = Label(window, text="Lower Bound", font=('Helvetica', 14, 'bold'))
    lower_bound_label.grid(column=1, row=2)
    upper_bound_label = Label(window, text="Upper Bound", font=('Helvetica', 14, 'bold'))
    upper_bound_label.grid(column=2, row=2)
    n_vals_label = Label(window, text="Number of Values", font=('Helvetica', 14, 'bold'))
    n_vals_label.grid(column=3, row=2)
    scale_label = Label(window, text="Scale Type (Arithmetic or Logarithmic)", font=('Helvetica', 14, 'bold'))
    scale_label.grid(column=4, row=2)
    manual_scale_label = Label(window, text="Manual Parameter Value Entry", font=('Helvetica', 14, 'bold'))
    manual_scale_label.grid(column=5, row=2)

    '''-----------R1------------'''
    R1_label = Label(window, text="R1")
    R1_label.grid(column=0, row=3)

    R1_LB = Entry(window)
    R1_LB.insert(0, "1e3")
    R1_LB.grid(column=1, row=3)

    R1_UB = Entry(window)
    R1_UB.insert(0, "1e5")
    R1_UB.grid(column=2, row=3)

    R1_n_vals = Entry(window)
    R1_n_vals.insert(0, "3")
    R1_n_vals.grid(column=3, row=3)

    R1_scale = Entry(window)
    R1_scale.insert(0, "logarithmic")
    R1_scale.grid(column=4, row=3)

    R1_manual = Entry(window)
    R1_manual.insert(0, "")
    R1_manual.grid(column=5, row=3)

    '''-----------R2------------'''
    R2_label = Label(window, text="R2")
    R2_label.grid(column=0, row=4)

    R2_LB = Entry(window)
    R2_LB.insert(0, "0")
    R2_LB.grid(column=1, row=4)

    R2_UB = Entry(window)
    R2_UB.insert(0, "0")
    R2_UB.grid(column=2, row=4)

    R2_n_vals = Entry(window)
    R2_n_vals.insert(0, "2")
    R2_n_vals.grid(column=3, row=4)

    R2_scale = Entry(window)
    R2_scale.insert(0, "arithmetic")
    R2_scale.grid(column=4, row=4)

    R2_manual = Entry(window)
    R2_manual.insert(0, "1e2, 1e3")
    R2_manual.grid(column=5, row=4)

    '''-----------R12------------'''
    R12_label = Label(window, text="R12")
    R12_label.grid(column=0, row=5)

    R12_LB = Entry(window)
    R12_LB.insert(0, "0")
    R12_LB.grid(column=1, row=5)

    R12_UB = Entry(window)
    R12_UB.insert(0, "0")
    R12_UB.grid(column=2, row=5)

    R12_n_vals = Entry(window)
    R12_n_vals.insert(0, "1")
    R12_n_vals.grid(column=3, row=5)

    R12_scale = Entry(window)
    R12_scale.insert(0, "arithmetic")
    R12_scale.grid(column=4, row=5)

    R12_manual = Entry(window)
    R12_manual.insert(0, "")
    R12_manual.grid(column=5, row=5)

    '''-----------g0------------'''
    g0_label = Label(window, text="g0")
    g0_label.grid(column=0, row=6)

    g0_LB = Entry(window)
    g0_LB.insert(0, "0.05")
    g0_LB.grid(column=1, row=6)

    g0_UB = Entry(window)
    g0_UB.insert(0, "0.05")
    g0_UB.grid(column=2, row=6)

    g0_n_vals = Entry(window)
    g0_n_vals.insert(0, "1")
    g0_n_vals.grid(column=3, row=6)

    g0_scale = Entry(window)
    g0_scale.insert(0, "arithmetic")
    g0_scale.grid(column=4, row=6)

    g0_manual = Entry(window)
    g0_manual.insert(0, "")
    g0_manual.grid(column=5, row=6)

    '''-----------Ss_1------------'''
    Ss_1_label = Label(window, text="Ss_1")
    Ss_1_label.grid(column=0, row=7)

    Ss_1_LB = Entry(window)
    Ss_1_LB.insert(0, "0.16878")
    Ss_1_LB.grid(column=1, row=7)

    Ss_1_UB = Entry(window)
    Ss_1_UB.insert(0, "0.16878")
    Ss_1_UB.grid(column=2, row=7)

    Ss_1_n_vals = Entry(window)
    Ss_1_n_vals.insert(0, "1")
    Ss_1_n_vals.grid(column=3, row=7)

    Ss_1_scale = Entry(window)
    Ss_1_scale.insert(0, "arithmetic")
    Ss_1_scale.grid(column=4, row=7)

    Ss_1_manual = Entry(window)
    Ss_1_manual.insert(0, "")
    Ss_1_manual.grid(column=5, row=7)

    '''-----------Ss_2------------'''
    Ss_2_label = Label(window, text="Ss_2")
    Ss_2_label.grid(column=0, row=8)

    Ss_2_LB = Entry(window)
    Ss_2_LB.insert(0, "0.087737")
    Ss_2_LB.grid(column=1, row=8)

    Ss_2_UB = Entry(window)
    Ss_2_UB.insert(0, "0.087737")
    Ss_2_UB.grid(column=2, row=8)

    Ss_2_n_vals = Entry(window)
    Ss_2_n_vals.insert(0, "1")
    Ss_2_n_vals.grid(column=3, row=8)

    Ss_2_scale = Entry(window)
    Ss_2_scale.insert(0, "arithmetic")
    Ss_2_scale.grid(column=4, row=8)

    Ss_2_manual = Entry(window)
    Ss_2_manual.insert(0, "")
    Ss_2_manual.grid(column=5, row=8)

    '''-----------Sr1_1------------'''
    Sr1_1_label = Label(window, text="Sr1_1")
    Sr1_1_label.grid(column=0, row=9)

    Sr1_1_LB = Entry(window)
    Sr1_1_LB.insert(0, "4e-5")
    Sr1_1_LB.grid(column=1, row=9)

    Sr1_1_UB = Entry(window)
    Sr1_1_UB.insert(0, "3e-2")
    Sr1_1_UB.grid(column=2, row=9)

    Sr1_1_n_vals = Entry(window)
    Sr1_1_n_vals.insert(0, "2")
    Sr1_1_n_vals.grid(column=3, row=9)

    Sr1_1_scale = Entry(window)
    Sr1_1_scale.insert(0, "arithmetic")
    Sr1_1_scale.grid(column=4, row=9)

    Sr1_1_manual = Entry(window)
    Sr1_1_manual.insert(0, "")
    Sr1_1_manual.grid(column=5, row=9)

    '''-----------Sr1_2------------'''
    Sr1_2_label = Label(window, text="Sr1_2")
    Sr1_2_label.grid(column=0, row=10)

    Sr1_2_LB = Entry(window)
    Sr1_2_LB.insert(0, "0.168780")
    Sr1_2_LB.grid(column=1, row=10)

    Sr1_2_UB = Entry(window)
    Sr1_2_UB.insert(0, "0.16878")
    Sr1_2_UB.grid(column=2, row=10)

    Sr1_2_n_vals = Entry(window)
    Sr1_2_n_vals.insert(0, "1")
    Sr1_2_n_vals.grid(column=3, row=10)

    Sr1_2_scale = Entry(window)
    Sr1_2_scale.insert(0, "arithmetic")
    Sr1_2_scale.grid(column=4, row=10)

    Sr1_2_manual = Entry(window)
    Sr1_2_manual.insert(0, "")
    Sr1_2_manual.grid(column=5, row=10)

    '''-----------Sr2_1------------'''
    Sr2_1_label = Label(window, text="Sr2_1")
    Sr2_1_label.grid(column=0, row=11)

    Sr2_1_LB = Entry(window)
    Sr2_1_LB.insert(0, "0.44")
    Sr2_1_LB.grid(column=1, row=11)

    Sr2_1_UB = Entry(window)
    Sr2_1_UB.insert(0, "0.44")
    Sr2_1_UB.grid(column=2, row=11)

    Sr2_1_n_vals = Entry(window)
    Sr2_1_n_vals.insert(0, "1")
    Sr2_1_n_vals.grid(column=3, row=11)

    Sr2_1_scale = Entry(window)
    Sr2_1_scale.insert(0, "arithmetic")
    Sr2_1_scale.grid(column=4, row=11)

    Sr2_1_manual = Entry(window)
    Sr2_1_manual.insert(0, "")
    Sr2_1_manual.grid(column=5, row=11)

    '''-----------Sr2_2------------'''
    Sr2_2_label = Label(window, text="Sr2_2")
    Sr2_2_label.grid(column=0, row=12)

    Sr2_2_LB = Entry(window)
    Sr2_2_LB.insert(0, "0.0020402")
    Sr2_2_LB.grid(column=1, row=12)

    Sr2_2_UB = Entry(window)
    Sr2_2_UB.insert(0, "0.0020402")
    Sr2_2_UB.grid(column=2, row=12)

    Sr2_2_n_vals = Entry(window)
    Sr2_2_n_vals.insert(0, "1")
    Sr2_2_n_vals.grid(column=3, row=12)

    Sr2_2_scale = Entry(window)
    Sr2_2_scale.insert(0, "arithmetic")
    Sr2_2_scale.grid(column=4, row=12)

    Sr2_2_manual = Entry(window)
    Sr2_2_manual.insert(0, "")
    Sr2_2_manual.grid(column=5, row=12)

    '''-----------Sr12_1------------'''
    Sr12_1_label = Label(window, text="Sr12_1")
    Sr12_1_label.grid(column=0, row=13)

    Sr12_1_LB = Entry(window)
    Sr12_1_LB.insert(0, "0.00015505")
    Sr12_1_LB.grid(column=1, row=13)

    Sr12_1_UB = Entry(window)
    Sr12_1_UB.insert(0, "0.00015505")
    Sr12_1_UB.grid(column=2, row=13)

    Sr12_1_n_vals = Entry(window)
    Sr12_1_n_vals.insert(0, "1")
    Sr12_1_n_vals.grid(column=3, row=13)

    Sr12_1_scale = Entry(window)
    Sr12_1_scale.insert(0, "arithmetic")
    Sr12_1_scale.grid(column=4, row=13)

    Sr12_1_manual = Entry(window)
    Sr12_1_manual.insert(0, "")
    Sr12_1_manual.grid(column=5, row=13)

    '''-----------Sr12_2------------'''
    Sr12_2_label = Label(window, text="Sr12_2")
    Sr12_2_label.grid(column=0, row=14)

    Sr12_2_LB = Entry(window)
    Sr12_2_LB.insert(0, "0.1357")
    Sr12_2_LB.grid(column=1, row=14)

    Sr12_2_UB = Entry(window)
    Sr12_2_UB.insert(0, "0.1357")
    Sr12_2_UB.grid(column=2, row=14)

    Sr12_2_n_vals = Entry(window)
    Sr12_2_n_vals.insert(0, "1")
    Sr12_2_n_vals.grid(column=3, row=14)

    Sr12_2_scale = Entry(window)
    Sr12_2_scale.insert(0, "arithmetic")
    Sr12_2_scale.grid(column=4, row=14)

    Sr12_2_manual = Entry(window)
    Sr12_2_manual.insert(0, "")
    Sr12_2_manual.grid(column=5, row=14)

    '''-----------Ts_r1------------'''
    Ts_r1_label = Label(window, text="Ts_r1")
    Ts_r1_label.grid(column=0, row=15)

    Ts_r1_LB = Entry(window)
    Ts_r1_LB.insert(0, "2.154E-06")
    Ts_r1_LB.grid(column=1, row=15)

    Ts_r1_UB = Entry(window)
    Ts_r1_UB.insert(0, "2.154E-06")
    Ts_r1_UB.grid(column=2, row=15)

    Ts_r1_n_vals = Entry(window)
    Ts_r1_n_vals.insert(0, "1")
    Ts_r1_n_vals.grid(column=3, row=15)

    Ts_r1_scale = Entry(window)
    Ts_r1_scale.insert(0, "arithmetic")
    Ts_r1_scale.grid(column=4, row=15)

    Ts_r1_manual = Entry(window)
    Ts_r1_manual.insert(0, "")
    Ts_r1_manual.grid(column=5, row=15)

    '''-----------Ts_r2------------'''
    Ts_r2_label = Label(window, text="Ts_r2")
    Ts_r2_label.grid(column=0, row=16)

    Ts_r2_LB = Entry(window)
    Ts_r2_LB.insert(0, "4.642E-05")
    Ts_r2_LB.grid(column=1, row=16)

    Ts_r2_UB = Entry(window)
    Ts_r2_UB.insert(0, "4.642E-05")
    Ts_r2_UB.grid(column=2, row=16)

    Ts_r2_n_vals = Entry(window)
    Ts_r2_n_vals.insert(0, "1")
    Ts_r2_n_vals.grid(column=3, row=16)

    Ts_r2_scale = Entry(window)
    Ts_r2_scale.insert(0, "arithmetic")
    Ts_r2_scale.grid(column=4, row=16)

    Ts_r2_manual = Entry(window)
    Ts_r2_manual.insert(0, "")
    Ts_r2_manual.grid(column=5, row=16)

    ''' CSV INPUT MODE '''
    csv_label = Label(window, text="CSV Input Mode", font=('Helvetica', 16, 'bold'))
    csv_label.grid(column=3, row=17)

    csv_filename_label = Label(window, text="CSV Filename")
    csv_filename_label.grid(column=2, row=18)

    csv_filename_entry = Entry(window)
    csv_filename_entry.insert(0, "shortParameterInputForTesting.csv")
    csv_filename_entry.grid(column=2, row=19)

    use_csv_label = Label(window, text="Use CSV File? (Y/N)")
    use_csv_label.grid(column=4, row=18)

    use_csv_entry = Entry(window)
    use_csv_entry.insert(0, "Y")
    use_csv_entry.grid(column=4, row=19)

    ''' OVERALL COHORT INFORMATION '''

    overall_cohort_label = Label(window, text="General Cohort Information", font=('Helvetica', 16, 'bold'))
    overall_cohort_label.grid(column=3, row=20)

    cohort_name_label = Label(window, text="Cohort Name")
    cohort_name_label.grid(column=2, row=21)

    cohort_name_entry = Entry(window)
    cohort_name_entry.grid(column=2, row=22)

    weeks_label = Label(window, text="Weeks")
    weeks_label.grid(column=3, row=21)

    weeks_entry = Entry(window)
    weeks_entry.insert(0, "255")
    weeks_entry.grid(column=3, row=22)

    strategy_number_label = Label(window, text="Strategy Number")
    strategy_number_label.grid(column=4, row=21)

    strategy_number_entry = Entry(window)
    strategy_number_entry.insert(0, "2.2")
    strategy_number_entry.grid(column=4, row=22)

    ''' Simulation button. '''
    btn = Button(window, text="Simulate Patient", bg="orange", fg="red", command=clicked_sim)
    btn.grid(column=3, row=23)

    window.mainloop()


main()
